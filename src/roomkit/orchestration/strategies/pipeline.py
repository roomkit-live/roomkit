"""Pipeline orchestration strategy.

Composes ``ConversationPipeline``, ``ConversationRouter``, and
``HandoffHandler`` into a declarative strategy that can be passed to
``RoomKit(orchestration=...)`` or ``create_room(orchestration=...)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.handoff import (
    HandoffHandler,
    build_handoff_tool,
    setup_handoff,
)
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import (
    ConversationState,
    set_conversation_state,
)

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.pipeline")


class Pipeline(Orchestration):
    """Linear pipeline orchestration strategy.

    Agents are chained in order: the first agent is the entry point,
    each subsequent agent is reachable via handoff from its predecessor.

    Example::

        kit = RoomKit(
            orchestration=Pipeline(
                agents=[triage, support, billing],
            ),
        )
        room = await kit.create_room()
    """

    def __init__(
        self,
        agents: list[Agent],
        routing: dict[str, list[str]] | None = None,
        *,
        supervisor_id: str | None = None,
        voice_channel_id: str | None = None,
        greet_on_handoff: bool = False,
        greeting_prompt: str | None = None,
        farewell_prompt: str | None = None,
    ) -> None:
        """Initialise the pipeline strategy.

        Args:
            agents: Ordered list of agents. First agent is the entry
                point. Stages chain linearly (each agent hands off to
                the next).
            routing: Optional keyword hints injected into agent system
                prompts (not runtime rules). Keys are agent channel IDs,
                values are keyword lists describing when to route there.
            supervisor_id: Optional supervisor agent ID that receives
                all events for monitoring.
            voice_channel_id: Voice channel ID for voice-aware handoffs.
            greet_on_handoff: Whether agents greet on handoff.
            greeting_prompt: Custom greeting prompt template.
            farewell_prompt: Custom farewell prompt template.
        """
        self._agents = list(agents)
        self._routing = routing or {}
        self._supervisor_id = supervisor_id
        self._voice_channel_id = voice_channel_id
        self._greet_on_handoff = greet_on_handoff
        self._greeting_prompt = greeting_prompt
        self._farewell_prompt = farewell_prompt

    def agents(self) -> list[Agent]:
        """Return all pipeline agents."""
        return list(self._agents)

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire pipeline routing and handoff into the room."""
        stages = self._build_stages()
        cp = ConversationPipeline(
            stages=stages,
            supervisor_id=self._supervisor_id,
        )

        router = cp.to_router()
        handler = HandoffHandler(
            kit=kit,
            router=router,
            phase_map=cp.get_phase_map(),
            allowed_transitions=cp.get_allowed_transitions(),
        )
        handler.greeting_map = {
            a.channel_id: g
            for a in self._agents
            if (g := getattr(a, "greeting", None)) is not None
        }
        handler.agents = {a.channel_id: a for a in self._agents}

        # Install router as room-scoped BEFORE_BROADCAST hook
        kit.hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=router.as_hook(),
                priority=-100,
                name=f"pipeline_router_{room_id}",
            ),
        )

        # Wire handoff tools per agent
        self._wire_handoff(self._agents, handler, cp)

        # Set initial conversation state
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase=stages[0].phase,
            active_agent_id=self._agents[0].channel_id,
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)

    def _build_stages(self) -> list[PipelineStage]:
        """Build pipeline stages from agent list."""
        stages: list[PipelineStage] = []
        for i, agent in enumerate(self._agents):
            phase = agent.channel_id
            next_phase = self._agents[i + 1].channel_id if i + 1 < len(self._agents) else None
            stages.append(
                PipelineStage(
                    phase=phase,
                    agent_id=agent.channel_id,
                    next=next_phase,
                    description=getattr(agent, "description", None),
                )
            )
        return stages

    def _wire_handoff(
        self,
        agents: list[Agent],
        handler: HandoffHandler,
        cp: ConversationPipeline,
    ) -> None:
        """Set up per-agent handoff tools with constrained targets."""
        agent_map: dict[str, Agent] = {a.channel_id: a for a in agents}
        stage_by_agent: dict[str, PipelineStage] = {s.agent_id: s for s in cp.stages}

        for agent in agents:
            # Guard against double registration (shared Agent instances)
            if any(t.name == "handoff_conversation" for t in agent._injected_tools):
                continue

            stage = stage_by_agent.get(agent.channel_id)
            if stage is None:
                setup_handoff(agent, handler)
                continue

            reachable_phases: set[str] = set()
            if stage.next:
                reachable_phases.add(stage.next)
            reachable_phases.update(stage.can_return_to)

            targets: list[tuple[str, str | None]] = []
            for s in cp.stages:
                if s.phase in reachable_phases and s.agent_id != agent.channel_id:
                    target_agent = agent_map.get(s.agent_id)
                    desc = target_agent.description if target_agent else None
                    if desc is None:
                        desc = s.description
                    targets.append((s.agent_id, desc))

            tool = build_handoff_tool(targets)
            setup_handoff(agent, handler, tool=tool)
