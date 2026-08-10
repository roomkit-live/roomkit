"""Task-plan event payload (RFC §9.2 ``ON_PLAN_UPDATED``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class PlanUpdatedEvent:
    """An agent rewrote its structured task plan.

    Carried to ``ON_PLAN_UPDATED`` hooks. The same plan is also published as
    an ephemeral event for live UIs; the hook is for hosts that record or act
    on plan changes without a realtime backend.
    """

    room_id: str
    """The room whose plan changed."""

    channel_id: str
    """The agent that owns the plan."""

    tasks: list[dict[str, Any]] = field(default_factory=list)
    """The plan as the agent wrote it: ``title`` / ``status`` entries."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the plan was updated."""
