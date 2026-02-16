"""Sliding-window memory provider — returns the last N events."""

from __future__ import annotations

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent


class SlidingWindowMemory(MemoryProvider):
    """Return the most recent events from the room context.

    This replicates the original ``AIChannel`` behavior of slicing
    ``context.recent_events[-max_events:]``.
    """

    def __init__(self, max_events: int = 50) -> None:
        self._max_events = max_events

    @property
    def name(self) -> str:
        return "SlidingWindowMemory"

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        events = context.recent_events[-self._max_events :]
        return MemoryResult(events=events)
