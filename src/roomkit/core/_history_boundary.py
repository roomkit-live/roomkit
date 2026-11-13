"""Keep the loaded window's boundary separate from its visible event indices."""

from __future__ import annotations

from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent


def bounded_history(
    events: list[RoomEvent], limit: int, start_index: int | None = 0
) -> tuple[list[RoomEvent], int | None]:
    """Retain the tail; a fresh read includes one sentinel beyond its limit.

    A carried tail passes its existing boundary, including ``None`` for an
    unknown one. Only dropping leading entries advances that boundary.
    """
    if limit <= 0:
        return [], None
    if len(events) <= limit:
        return events, start_index
    kept = events[-limit:]
    return kept, kept[0].index


def with_history_tail(context: RoomContext, events: list[RoomEvent], limit: int) -> RoomContext:
    """Append or replace events with the same hard cap and an honest boundary."""
    recent, start = bounded_history(events, limit, context.history_start_index)
    return context.model_copy(update={"recent_events": recent, "history_start_index": start})
