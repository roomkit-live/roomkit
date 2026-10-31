"""Room model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from roomkit.models._keys import Id
from roomkit.models._time import UtcDatetime
from roomkit.models.enums import AgentResponsePolicy, RoomStatus


class RoomTimers(BaseModel):
    """Timer configuration for a room (RFC §5.1).

    An ACTIVE room pauses once ``inactive_after_seconds`` passed without
    activity, counted from ``last_activity_at``, or from the room's
    ``created_at`` while it has had none; a PAUSED room closes once
    ``closed_after_seconds`` passed from its ``paused_at``. Its activity is
    the events the pipeline stores DELIVERED, which also resume it when
    PAUSED.
    """

    inactive_after_seconds: int | None = Field(default=None, ge=0)
    closed_after_seconds: int | None = Field(default=None, ge=0)
    # The instant of the room's last activity; None while it has had none.
    last_activity_at: UtcDatetime | None = None


class Room(BaseModel):
    """A conversation room."""

    id: Id
    organization_id: str | None = None
    status: RoomStatus = RoomStatus.ACTIVE
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    # When the room was paused (RFC §5.1); None unless PAUSED. The close
    # timer counts from it.
    paused_at: UtcDatetime | None = None
    closed_at: UtcDatetime | None = None
    timers: RoomTimers = Field(default_factory=RoomTimers)
    # What an agent's own output solicits here (RFC §19.3.1). Room state, not
    # a construction option: broadcast happens in whichever worker owns the
    # delivery lane, and it must reach the same verdict as every other.
    agent_response_policy: AgentResponsePolicy = AgentResponsePolicy.AGENT_CHAIN
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Store-managed, like delivered_index: committing an event moves them, and
    # update_room() keeps the stored values over a caller's copy.
    event_count: int = Field(default=0, ge=0)
    # The highest index committed in the room; None while it has no event
    # (RFC §5.1, §8.1). 0 would read as the first event's index.
    latest_index: int | None = Field(default=None, ge=0)
    # Highest event index whose delivery set has executed (RFC §10.1 step 14);
    # -1 = none yet. Store-managed: advanced only through
    # ConversationStore.advance_delivered_index() — update_room() never
    # rewinds it from a caller's stale copy.
    delivered_index: int = Field(default=-1, ge=-1)

    @model_validator(mode="after")
    def _no_index_without_events(self) -> Room:
        # A room stored before latest_index was nullable recorded 0 while it
        # had no event; that 0 names no event. Written only then: a model
        # holding the room (a context) runs this on the store's snapshot.
        if self.event_count == 0 and self.latest_index is not None:
            self.latest_index = None
        return self
