"""Await owner publications without changing the turn's asynchronous delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from roomkit import RoomKit
from roomkit.realtime.base import EphemeralEvent
from roomkit.turns import OwnerEvents


class _ObservedEvents(OwnerEvents):
    def __init__(self, given: OwnerEvents) -> None:
        self.given = given
        self.pending: set[asyncio.Future[None]] = set()

    async def _observe[T](self, operation: Awaitable[T]) -> T:
        done = asyncio.get_running_loop().create_future()
        self.pending.add(done)
        try:
            return await operation
        finally:
            done.set_result(None)
            self.pending.discard(done)

    async def reaches_room(self, room_id: str, channel_id: str) -> bool:
        return await self._observe(self.given.reaches_room(room_id, channel_id))

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        await self._observe(self.given.publish(channel, event))

    async def emit(
        self,
        event_type: str,
        *,
        room_id: str | None,
        channel_id: str | None,
        event_id: str | None,
        data: dict[str, Any],
    ) -> None:
        await self._observe(
            self.given.emit(
                event_type, room_id=room_id, channel_id=channel_id, event_id=event_id, data=data
            )
        )


class Publications:
    """Track each kit's owner event operations, including binding reads.

    A publisher outlives its turn and can await SQLite before publishing.
    Waiting for event-loop ticks or just the client's current turns misses
    that work. Operations remain in their original task and context.
    """

    def __init__(self) -> None:
        self._events: dict[RoomKit, _ObservedEvents] = {}

    def watch(self, kit: RoomKit) -> None:
        given = kit._turns._events
        assert given is not None
        events = _ObservedEvents(given)
        kit._turns._events = events
        self._events[kit] = events

    async def drain(self, kit: RoomKit) -> bool:
        pending = list(self._events[kit].pending)
        if not pending:
            return False
        # Cancelling the harness wait does not cancel production publications.
        await asyncio.wait(pending)
        return True
