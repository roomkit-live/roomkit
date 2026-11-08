"""Lifecycle delegation shared by memory decorators."""

from __future__ import annotations

from roomkit.memory.base import MemoryProvider
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage


class _MemoryWrapper(MemoryProvider):
    def __init__(self, inner: MemoryProvider) -> None:
        self._inner = inner

    @property
    def recent_events_window(self) -> int:
        return self._inner.recent_events_window

    async def compact(
        self,
        room_id: str,
        history: list[AIMessage],
        *,
        usage: dict[str, int],
        channel_id: str | None = None,
    ) -> list[AIMessage] | None:
        return await self._inner.compact(room_id, history, usage=usage, channel_id=channel_id)

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        await self._inner.ingest(room_id, event, channel_id=channel_id)

    async def clear(self, room_id: str) -> None:
        await self._inner.clear(room_id)

    async def close(self) -> None:
        await self._inner.close()
