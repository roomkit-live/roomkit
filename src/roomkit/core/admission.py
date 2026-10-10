"""Off-lock checks: admission tickets for ``needs_lock=False`` hooks (RFC §9.5.1).

A ``BEFORE_BROADCAST`` check that reads only the event (a PII scan, a
moderation call) runs before the room lock is taken, so the I/O of one
message's check does not hold the other events of the room. Arrival order,
which holding the lock across the check would give, comes from a per-room
ticket: checks of successive events overlap, but each event waits for every
earlier ticket of the room to be released before it takes the lock, so they
commit in arrival order.

Neither existing ordering mechanism fits. The room lock (:class:`RoomLockManager`)
is a mutex: holding it across the check is exactly the serialization being
removed. The delivery lanes (:mod:`roomkit.core.lanes`) order by committed
index, and at this point no index exists yet.

The ticket lives in process memory: arrival order holds within one process.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from roomkit.core.locks import _has_room_lock
from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine, SyncPipelineResult
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

# Rooms whose off-lock check the current task is running. An event injected
# from inside such a check (a hook calling ``send_event``) must not queue
# behind the ticket its own caller holds — that is a deadlock — so it skips
# admission and commits ahead of the event being checked. Tasks the check
# spawns inherit the exemption.
_checking_rooms: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_checking_rooms", default=frozenset()
)


class _Ticket:
    """One event's place in its room's admission order."""

    __slots__ = ("_ahead", "_released")

    def __init__(self, ahead: list[_Ticket]) -> None:
        self._ahead = ahead
        self._released = asyncio.Event()

    async def wait_turn(self) -> None:
        """Return once every ticket taken before this one is released."""
        for ticket in self._ahead:
            await ticket._released.wait()
        self._ahead = []

    def release(self) -> None:
        self._released.set()


class RoomAdmission:
    """Per-room admission tickets and the off-lock check they order."""

    def __init__(
        self,
        hook_engine: HookEngine,
        lock_manager: RoomLockManager,
        build_context: Callable[[str], Awaitable[RoomContext]],
    ) -> None:
        self._hook_engine = hook_engine
        self._lock_manager = lock_manager
        self._build_context = build_context
        self._queues: dict[str, deque[_Ticket]] = {}

    def _take(self, room_id: str) -> _Ticket:
        queue = self._queues.setdefault(room_id, deque())
        ticket = _Ticket(list(queue))
        queue.append(ticket)
        return ticket

    def _release(self, room_id: str, ticket: _Ticket) -> None:
        # Synchronous on purpose: it runs from ``finally`` on every exit path,
        # cancellation included, and must not be able to fail half-way.
        ticket.release()
        queue = self._queues.get(room_id)
        if queue is None:
            return
        queue.remove(ticket)
        if not queue:
            del self._queues[room_id]

    def pending(self, room_id: str) -> int:
        """How many tickets of *room_id* are held (for tests and metrics)."""
        return len(self._queues.get(room_id, ()))

    @asynccontextmanager
    async def admitted(
        self, room_id: str, event: RoomEvent, context: RoomContext | None
    ) -> AsyncGenerator[SyncPipelineResult | None, None]:
        """Run *event*'s off-lock check, then hold its turn until exit.

        Yields ``None`` — nothing taken, every hook left to the locked pass —
        when no off-lock hook matches the event, or when the caller could
        only wait on itself: it runs inside an off-lock check of the same
        room, or already holds the room lock (a locked hook, or code under
        the lock, injecting an event). Either way the ticket ahead of this
        event is the caller's own, released only once the caller returns.

        Otherwise yields the off-lock outcome once every earlier ticket of the
        room has been released. The caller takes the room lock and commits inside the
        ``async with``; the ticket is released on exit, whatever the path —
        committed, blocked, refused, failed, timed out or cancelled.
        """
        if (
            room_id in _checking_rooms.get()
            or _has_room_lock(room_id, self._lock_manager)
            or not self._hook_engine.has_off_lock_hooks(room_id, event)
        ):
            yield None
            return

        ticket = self._take(room_id)
        try:
            if context is None:
                context = await self._build_context(room_id)
            token = _checking_rooms.set(_checking_rooms.get() | {room_id})
            try:
                outcome = await self._hook_engine.run_sync_hooks(
                    room_id, HookTrigger.BEFORE_BROADCAST, event, context, needs_lock=False
                )
            finally:
                _checking_rooms.reset(token)
            await ticket.wait_turn()
            yield outcome
        finally:
            self._release(room_id, ticket)
