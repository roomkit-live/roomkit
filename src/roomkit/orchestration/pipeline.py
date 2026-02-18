"""Pipeline helper for sequential agent workflows.

ConversationPipeline generates RoutingRules for structured multi-agent
workflows with optional loops (e.g., coder <-> reviewer).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.handoff import HandoffHandler, setup_handoff
from roomkit.orchestration.router import ConversationRouter, RoutingConditions, RoutingRule

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration")


class PipelineStage(BaseModel):
    """A stage in a conversation pipeline."""

    phase: str
    agent_id: str
    next: str | None = None
    can_return_to: set[str] = Field(default_factory=set)


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
        agents: list[AIChannel],
        *,
        agent_aliases: dict[str, str] | None = None,
        hook_priority: int = -100,
        greet_on_handoff: bool = False,
        voice_channel_id: str | None = None,
        greeting_prompt: str | None = None,
    ) -> tuple[ConversationRouter, HandoffHandler]:
        """Wire routing and handoff in one call.

        Creates a router from this pipeline, registers it as a
        ``BEFORE_BROADCAST`` sync hook, builds a ``HandoffHandler``
        with the pipeline's phase map and transition constraints,
        and calls ``setup_handoff`` on every agent.

        When *greet_on_handoff* is ``True``, two extra hooks are
        registered:

        - **ON_HANDOFF** (async): blocks the old agent's farewell via
          a ``BEFORE_TTS`` flag, then sends a synthetic inbound message
          on *voice_channel_id* to prompt the new agent to greet.
        - **BEFORE_TTS** (sync): blocks TTS while a handoff is pending.

        Returns ``(router, handler)`` for further customisation.
        """
        if greet_on_handoff and not voice_channel_id:
            raise ValueError("greet_on_handoff=True requires voice_channel_id")

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
        )
        for agent in agents:
            setup_handoff(agent, handler)

        if greet_on_handoff:
            self._register_greet_hooks(
                kit,
                voice_channel_id=voice_channel_id,  # type: ignore[arg-type]
                greeting_prompt=greeting_prompt,
                hook_priority=hook_priority,
            )

        return router, handler

    def _register_greet_hooks(
        self,
        kit: RoomKit,
        *,
        voice_channel_id: str,
        greeting_prompt: str | None,
        hook_priority: int,
    ) -> None:
        """Register ON_HANDOFF + BEFORE_TTS hooks for handoff greeting."""
        from roomkit.models.context import RoomContext
        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import RoomEvent, TextContent
        from roomkit.models.hook import HookResult

        prompt = greeting_prompt or (
            "[The caller has just been transferred to you — please introduce yourself briefly]"
        )

        # Rooms where a handoff is in flight — TTS blocked until greeting fires
        _handoff_pending: set[str] = set()

        @kit.hook(HookTrigger.ON_HANDOFF, execution=HookExecution.ASYNC)
        async def _on_handoff(event: RoomEvent, _ctx: RoomContext) -> None:
            meta = event.metadata or {}
            to_agent = meta.get("to_agent", "")
            from_agent = meta.get("from_agent", "")
            logger.info("greet_on_handoff: %s → %s", from_agent, to_agent)

            _handoff_pending.add(event.room_id)

            async def _trigger_greeting() -> None:
                _handoff_pending.discard(event.room_id)
                await kit.process_inbound(
                    InboundMessage(
                        channel_id=voice_channel_id,
                        sender_id="system",
                        content=TextContent(body=prompt),
                    ),
                    room_id=event.room_id,
                )

            loop = asyncio.get_running_loop()
            loop.create_task(_trigger_greeting(), context=contextvars.Context())

        @kit.hook(
            HookTrigger.BEFORE_TTS,
            execution=HookExecution.SYNC,
            priority=hook_priority - 1,
        )
        async def _block_farewell(
            text: str,
            ctx: RoomContext,  # noqa: ARG001
        ) -> HookResult:
            if ctx.room.id in _handoff_pending:
                return HookResult.block("handoff_transition")
            return HookResult.allow()
