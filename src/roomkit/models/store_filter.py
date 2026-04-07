"""Event filtering and persistence policy models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from roomkit.models.enums import ChannelType, EventType


class EventFilter(BaseModel):
    """Filter criteria for querying room events.

    Used with :meth:`ConversationStore.list_events` to select events by type,
    source, time range, or correlation group.
    """

    event_types: list[EventType] | None = None
    """Include only events of these types. ``None`` means all types."""

    exclude_types: list[EventType] | None = None
    """Exclude events of these types. Takes precedence over *event_types*."""

    visibility: str | None = None
    """Filter by visibility value (e.g. ``"all"``, ``"agents"``)."""

    source_channel_id: str | None = None
    """Filter by originating channel ID."""

    source_channel_type: ChannelType | None = None
    """Filter by originating channel type."""

    correlation_id: str | None = None
    """Return all events sharing this correlation ID (e.g. one AI response)."""

    participant_id: str | None = None
    """Filter by participant ID in the event source."""

    after_time: datetime | None = None
    """Return events created after this timestamp (exclusive)."""

    before_time: datetime | None = None
    """Return events created before this timestamp (exclusive)."""

    @model_validator(mode="after")
    def _validate_time_range(self) -> EventFilter:
        if (
            self.after_time is not None
            and self.before_time is not None
            and self.after_time >= self.before_time
        ):
            msg = "after_time must be before before_time"
            raise ValueError(msg)
        return self


class PersistencePolicy(BaseModel):
    """Controls which event types are persisted to the store.

    Configured on :class:`RoomKit` to filter events before they reach
    :meth:`ConversationStore.add_event`.

    When *persist_types* is ``None`` (default), all event types are persisted.
    *exclude_types* always takes precedence over *persist_types*.
    """

    persist_types: set[EventType] | None = None
    """Persist only these event types. ``None`` means persist all."""

    exclude_types: set[EventType] = Field(default_factory=set)
    """Never persist these event types. Takes precedence over *persist_types*."""

    def should_persist(self, event_type: EventType) -> bool:
        """Return whether an event of the given type should be persisted."""
        if event_type in self.exclude_types:
            return False
        if self.persist_types is not None:
            return event_type in self.persist_types
        return True
