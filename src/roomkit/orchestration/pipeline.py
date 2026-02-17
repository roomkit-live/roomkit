"""Pipeline helper for sequential agent workflows.

ConversationPipeline generates RoutingRules for structured multi-agent
workflows with optional loops (e.g., coder <-> reviewer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.handoff import HandoffHandler, setup_handoff
from roomkit.orchestration.router import ConversationRouter, RoutingConditions, RoutingRule

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.core.framework import RoomKit


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
    ) -> tuple[ConversationRouter, HandoffHandler]:
        """Wire routing and handoff in one call.

        Creates a router from this pipeline, registers it as a
        ``BEFORE_BROADCAST`` sync hook, builds a ``HandoffHandler``
        with the pipeline's phase map and transition constraints,
        and calls ``setup_handoff`` on every agent.

        Returns ``(router, handler)`` for further customisation.
        """
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

        return router, handler
