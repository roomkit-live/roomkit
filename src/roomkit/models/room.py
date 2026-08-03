"""Room model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import AgentResponsePolicy, RoomStatus


class RoomTimers(BaseModel):
    """Timer configuration for a room."""

    inactive_after_seconds: int | None = Field(default=None, ge=0)
    closed_after_seconds: int | None = Field(default=None, ge=0)
    last_activity_at: datetime | None = None


class Room(BaseModel):
    """A conversation room."""

    id: str
    organization_id: str | None = None
    status: RoomStatus = RoomStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    timers: RoomTimers = Field(default_factory=RoomTimers)
    # What an agent's own output solicits here (RFC §19.3.1). Room state, not
    # a construction option: broadcast happens in whichever worker owns the
    # delivery lane, and it must reach the same verdict as every other.
    agent_response_policy: AgentResponsePolicy = AgentResponsePolicy.AGENT_CHAIN
    metadata: dict[str, Any] = Field(default_factory=dict)
    event_count: int = Field(default=0, ge=0)
    latest_index: int = Field(default=0, ge=0)
    # Highest event index whose delivery set has executed (RFC §10.1 step 14);
    # -1 = none yet. Store-managed: advanced only through
    # ConversationStore.advance_delivered_index() — update_room() never
    # rewinds it from a caller's stale copy.
    delivered_index: int = Field(default=-1, ge=-1)
