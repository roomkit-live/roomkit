"""Pipeline helper for sequential agent workflows.

ConversationPipeline generates RoutingRules for structured multi-agent
workflows with optional loops (e.g., coder <-> reviewer).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from roomkit.orchestration.router import ConversationRouter, RoutingConditions, RoutingRule


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
