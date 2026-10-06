"""Event filtering and persistence policy models."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from roomkit.models.enums import ChannelType, EventStatus, EventType
from roomkit.models.event import RoomEvent


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

    responds_to: str | None = None
    """Return the answers to this event: the events whose ``responds_to``
    equals this id (RFC §8.5)."""

    parent_event_id: str | None = None
    """Return the replies of this thread root — events whose
    ``parent_event_id`` equals this id (flat two-level threading)."""

    top_level_only: bool = False
    """Return only top-level events (``parent_event_id IS NULL``): thread roots
    and non-threaded messages, excluding replies. Mutually exclusive with
    *parent_event_id*."""

    after_time: datetime | None = None
    """Return events created after this timestamp (exclusive)."""

    before_time: datetime | None = None
    """Return events created before this timestamp (exclusive)."""

    include_blocked: bool = False
    """Serve the rows the room refused too. An event stored ``BLOCKED`` was
    delivered to nobody (RFC §10.1 step 10, §7.5 rule 2): a hook refused it,
    its source could not write, or a cap stopped it. By default
    :meth:`ConversationStore.list_events` returns only what the room received;
    set this for an audit, copy or deletion reader that must see the refused
    rows as well. A read by id (``get_event``) is never filtered."""

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

    @model_validator(mode="after")
    def _validate_thread_filters(self) -> EventFilter:
        if self.top_level_only and self.parent_event_id is not None:
            msg = "top_level_only and parent_event_id are mutually exclusive"
            raise ValueError(msg)
        return self


def includes_blocked(event_filter: EventFilter | None) -> bool:
    """Whether a timeline read asked for the rows the room refused.

    ``None``, the default of every read, asks for the received rows only; so
    does a filter that leaves :attr:`EventFilter.include_blocked` unset.
    """
    return event_filter is not None and event_filter.include_blocked


def received_events(events: Iterable[RoomEvent]) -> list[RoomEvent]:
    """The rows the room received: every event but those stored ``BLOCKED``.

    The one predicate behind the store's default (``include_blocked=False``,
    RFC §14.1) and behind the per-reader history filter
    (:func:`~roomkit.core.visibility.visible_events`, §7.5 rule 8), so the two
    cannot drift on what a refused row is. A SQL store expresses the same
    predicate as a condition on the row, ahead of the page cut.
    """
    return [e for e in events if e.status != EventStatus.BLOCKED]


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
