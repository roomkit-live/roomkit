"""Pipeline helper for sequential agent workflows.

ConversationPipeline generates RoutingRules for structured multi-agent
workflows with optional loops (e.g., coder <-> reviewer).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from roomkit.channels.agent import Agent
from roomkit.channels.declared import owner_tools_of
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration._agents import held
from roomkit.orchestration._handoff_greeting import HandoffGreeting
from roomkit.orchestration._realtime_pipeline import (
    RealtimePipeline,
    refuse_agents_with_unserved,
)
from roomkit.orchestration.handoff import (
    HandoffHandler,
    build_handoff_tool,
    require_live_agents,
    setup_handoff,
)
from roomkit.orchestration.router import ConversationRouter, RoutingConditions, RoutingRule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from roomkit.core.framework import RoomKit
    from roomkit.orchestration._agents import HeldAgent

logger = logging.getLogger("roomkit.orchestration")


class PipelineStage(BaseModel):
    """A stage in a conversation pipeline."""

    phase: str
    agent_id: str
    next: str | None = None
    can_return_to: set[str] = Field(default_factory=set)
    description: str | None = None


class ConversationPipeline:
    """Generates routing rules for sequential agent workflows.

    Example::

        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="analysis", agent_id="agent-discuss", next="coding"),
                PipelineStage(phase="coding", agent_id="agent-coder", next="review"),
                PipelineStage(phase="review", agent_id="agent-reviewer",
                              next="report", can_return_to={"coding"}),
                PipelineStage(phase="report", agent_id="agent-writer", next=None),
            ],
            supervisor_id="agent-supervisor",
        )
        router = pipeline.to_router()
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        default_phase: str | None = None,
        supervisor_id: str | None = None,
    ) -> None:
        self._stages = stages
        self._default_phase = default_phase or (stages[0].phase if stages else None)
        self._supervisor_id = supervisor_id
        self._stage_map = {s.phase: s for s in stages}
        self._validate()

    def _validate(self) -> None:
        """Validate pipeline graph consistency."""
        phase_names = {s.phase for s in self._stages}
        for stage in self._stages:
            if stage.next and stage.next not in phase_names:
                msg = f"Stage '{stage.phase}' has next='{stage.next}' which is not a valid phase"
                raise ValueError(msg)
            for ret in stage.can_return_to:
                if ret not in phase_names:
                    msg = (
                        f"Stage '{stage.phase}' has can_return_to='{ret}' "
                        f"which is not a valid phase"
                    )
                    raise ValueError(msg)

    @property
    def stages(self) -> list[PipelineStage]:
        """The pipeline stages."""
        return list(self._stages)

    def to_router(self) -> ConversationRouter:
        """Generate a ConversationRouter from this pipeline."""
        rules = []
        for i, stage in enumerate(self._stages):
            rules.append(
                RoutingRule(
                    agent_id=stage.agent_id,
                    conditions=RoutingConditions(phases={stage.phase}),
                    priority=i,
                )
            )

        default_stage = self._stage_map.get(self._default_phase or "")
        default_agent = default_stage.agent_id if default_stage else None

        return ConversationRouter(
            rules=rules,
            default_agent_id=default_agent,
            supervisor_id=self._supervisor_id,
        )

    def get_phase_map(self) -> dict[str, str]:
        """Return agent_id -> default phase mapping for HandoffHandler."""
        return {s.agent_id: s.phase for s in self._stages}

    def get_allowed_transitions(self) -> dict[str, set[str]]:
        """Return phase -> allowed next phases for validation."""
        transitions: dict[str, set[str]] = {}
        for stage in self._stages:
            allowed: set[str] = set()
            if stage.next:
                allowed.add(stage.next)
            allowed.update(stage.can_return_to)
            transitions[stage.phase] = allowed
        return transitions

    def install(
        self,
        kit: RoomKit,
        agents: Sequence[Agent | str],
        *,
        agent_aliases: dict[str, str] | None = None,
        hook_priority: int = -100,
        greet_on_handoff: bool = False,
        voice_channel_id: str | None = None,
        greeting_prompt: str | None = None,
        farewell_prompt: str | None = None,
    ) -> tuple[ConversationRouter, HandoffHandler]:
        """Wire routing and handoff in one call.

        Creates a router from this pipeline, registers it as a
        ``BEFORE_BROADCAST`` sync hook, builds a ``HandoffHandler``
        with the pipeline's phase map and transition constraints,
        and calls ``setup_handoff`` on every agent, each live or named by the
        id of one the kit holds, by its declaration alone included (RFC
        §10.6.2). A realtime voice channel runs each agent's prompt and tools
        in its session: it takes live agents only.

        When *greet_on_handoff* is ``True``, two extra hooks are
        registered:

        - **ON_HANDOFF** (async): blocks the old agent's farewell via
          a ``BEFORE_TTS`` flag, then sends an ``INSTRUCTION`` addressed
          to the new agent on *voice_channel_id* to prompt it to greet —
          never stored as the caller's words (RFC §10.1.1).
        - **BEFORE_TTS** (sync): blocks TTS while a handoff is pending.

        Returns ``(router, handler)`` for further customisation.
        """
        if greet_on_handoff and not voice_channel_id:
            raise ValueError("greet_on_handoff=True requires voice_channel_id")
        held_agents = [held(kit, a) for a in agents]

        # Detect speech-to-speech mode
        is_realtime = False
        if voice_channel_id:
            is_realtime = isinstance(kit.channels.get(voice_channel_id), RealtimeVoiceChannel)
        live = [a for a in held_agents if isinstance(a, Agent)]
        if is_realtime:
            # Refused before anything is installed.
            require_live_agents(held_agents)
            refuse_agents_with_unserved(live, voice_channel_id)  # ty: ignore[invalid-argument-type]
        else:
            # Every agent read before anything is installed, as the router's install.
            for agent in held_agents:
                owner_tools_of(kit, agent)

        router = self.to_router()

        kit.hook(
            HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            priority=hook_priority,
        )(router.as_hook())

        handler = HandoffHandler(
            kit=kit,
            router=router,
            agent_aliases=agent_aliases,
            phase_map=self.get_phase_map(),
            allowed_transitions=self.get_allowed_transitions(),
            known_agents=({a.channel_id for a in held_agents} if is_realtime else None),
            event_channel_id=(voice_channel_id if is_realtime else None),
        )

        # Populate agent metadata from Agent fields
        handler.greeting_map = {
            a.channel_id: g for a in held_agents if (g := getattr(a, "greeting", None)) is not None
        }
        handler.agents = {a.channel_id: a for a in held_agents}

        if is_realtime:
            self._wire_realtime(
                kit,
                live,
                handler,
                voice_channel_id,  # ty: ignore[invalid-argument-type]
                greet_on_handoff=greet_on_handoff,
                greeting_prompt=greeting_prompt,
            )
        else:
            self._wire_handoff(held_agents, handler)

            if voice_channel_id:
                wire_voice_map(kit, held_agents, voice_channel_id)

            if greet_on_handoff:
                HandoffGreeting(
                    kit,
                    voice_channel_id,  # ty: ignore[invalid-argument-type]
                    greeting_prompt,
                    farewell_prompt,
                ).register(hook_priority)

        return router, handler

    def _wire_handoff(
        self,
        agents: Sequence[HeldAgent],
        handler: HandoffHandler,
    ) -> None:
        """Set up per-agent handoff tools with constrained target enums.

        Each agent's handoff tool ``target`` parameter is restricted to
        only the reachable agent IDs (with descriptions derived from
        ``Agent.description`` or ``PipelineStage.description``).
        """
        agent_map: dict[str, HeldAgent] = {a.channel_id: a for a in agents}
        stage_by_agent: dict[str, PipelineStage] = {s.agent_id: s for s in self._stages}

        for agent in agents:
            stage = stage_by_agent.get(agent.channel_id)
            if stage is None:
                # Agent not in pipeline — generic tool
                setup_handoff(agent, handler)
                continue

            # Compute reachable targets
            reachable_phases: set[str] = set()
            if stage.next:
                reachable_phases.add(stage.next)
            reachable_phases.update(stage.can_return_to)

            targets: list[tuple[str, str | None]] = []
            for s in self._stages:
                if s.phase in reachable_phases and s.agent_id != agent.channel_id:
                    # Prefer Agent.description, fall back to PipelineStage.description
                    target_agent = agent_map.get(s.agent_id)
                    desc = target_agent.description if target_agent else None
                    if desc is None:
                        desc = s.description
                    targets.append((s.agent_id, desc))

            tool = build_handoff_tool(targets)
            setup_handoff(agent, handler, tool=tool)

    def _wire_realtime(
        self,
        kit: RoomKit,
        agents: list[Agent],
        handler: HandoffHandler,
        rtv_channel_id: str,
        *,
        greet_on_handoff: bool = False,
        greeting_prompt: str | None = None,
    ) -> None:
        """Wire speech-to-speech orchestration on a RealtimeVoiceChannel.

        Builds per-agent configurations (system prompt with identity block,
        voice, handoff tools), serves the handoff and the agents' tools on the
        channel, starts each new session with its room's active agent, and
        reconfigures a room's sessions on its handoffs (RFC §19.5).
        """
        rtv = kit.channels.get(rtv_channel_id)
        if not isinstance(rtv, RealtimeVoiceChannel):
            msg = f"voice_channel_id={rtv_channel_id!r} does not point to a RealtimeVoiceChannel"
            raise TypeError(msg)

        default_stage = self._stage_map.get(self._default_phase or "")
        default_agent_id = default_stage.agent_id if default_stage else agents[0].channel_id
        wiring = RealtimePipeline(
            kit,
            rtv,
            agents,
            self._stages,
            handler,
            default_agent_id,
            greet_on_handoff=greet_on_handoff,
            greeting_prompt=greeting_prompt,
        )
        wiring.install()
        # on_handoff_complete: reconfigure the room's realtime sessions
        handler.on_handoff_complete = wiring.on_handoff_complete

        logger.info(
            "Wired speech-to-speech orchestration: %d agents on %s",
            len(agents),
            rtv_channel_id,
        )


def wire_voice_map(kit: RoomKit, agents: Sequence[HeldAgent], voice_channel_id: str) -> None:
    """Auto-wire voice map entries from Agent.voice fields."""
    auto_map: dict[str, str] = {a.channel_id: a.voice for a in agents if a.voice is not None}
    if not auto_map:
        return

    vc = kit.channels.get(voice_channel_id)
    if isinstance(vc, VoiceChannel):
        vc.update_voice_map(auto_map)
    else:
        logger.warning(
            "voice_channel_id=%r does not point to a VoiceChannel; skipping voice map auto-wiring",
            voice_channel_id,
        )
