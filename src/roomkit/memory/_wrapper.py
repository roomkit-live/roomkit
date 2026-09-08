"""Lifecycle delegation shared by memory decorators."""

from __future__ import annotations

from roomkit.memory.base import MemoryProvider
from roomkit.models.event import RoomEvent


class _MemoryWrapper(MemoryProvider):
    def __init__(self, inner: MemoryProvider) -> None:
        self._inner = inner

    @property
    def recent_events_window(self) -> int:
        return self._inner.recent_events_window

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        await self._inner.ingest(room_id, event, channel_id=channel_id)

    async def clear(self, room_id: str) -> None:
        await self._inner.clear(room_id)

    async def close(self) -> None:
        await self._inner.close()
