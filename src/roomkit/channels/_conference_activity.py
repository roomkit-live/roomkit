"""What a conference teardown must not overtake.

A conference channel does multi-step work against a room — join then announce,
join then record then announce, synthesize then publish — and every step is an
await into code the channel does not control: a backend's network call, an
integrator's hook, a synthesizer's next chunk. A detach can land inside any one
of them.

Re-reading the room's generation after each await is necessary but not
sufficient: between the read and the next await there is another window, and
closing them one at a time is how a channel accumulates five of them. What
closes the class is the other half of the pair — teardown *waits* for work that
is already in flight, so a step that has passed its check finishes before
anything contradicts it, and a step that has not passed its check sees the new
generation and abandons itself.

The one thing this must not do is wait for itself. A listener of the very
announcement being made may drive the teardown from inside it — a
``conference_started`` handler that detaches the channel is ordinary integrator
code — and there the teardown is downstream of the work it would wait for.
Waiting would be waiting for the caller to finish being the caller: a deadlock,
and one that only appears when an integrator writes a perfectly reasonable
handler.

Which activities the caller is *inside* is carried on a
:class:`~contextvars.ContextVar` rather than read off the running task. The
teardown rarely runs on the announcing task — the hook engine dispatches
lifecycle hooks onto tasks of their own — but a task inherits a copy of the
context that created it, so the marker follows the causal chain wherever the
framework happens to schedule it.

See RFC section 12.10.4.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from collections.abc import AsyncIterator, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("roomkit.channels.conference")

# Activities the current context is nested inside — the ones a drain running
# here must not wait for, because it is downstream of them.
_enclosing: ContextVar[frozenset[int]] = ContextVar(
    "roomkit_conference_activity", default=frozenset()
)

_ids = itertools.count()

# How long a teardown waits, in total, before going ahead without whatever is
# left. Bounded because the work ends in code the channel does not own — a
# backend publishing a chunk, an integrator handling an event — and a detach
# that waits forever on a wedged SFU is a worse failure than a detach that
# overtakes it.
#
# A *total* budget, not a budget per item: seven wedged activities at five
# seconds each is thirty-five, and the hook engine cancels a lifecycle hook at
# thirty. A drain must never be the reason that ceiling is reached.
DRAIN_TIMEOUT_S = 5.0

# How long a cancelled closing step gets to unwind before it is abandoned.
# Cancellation is a request, and a provider can swallow it — an STT stream
# that catches CancelledError, a backend that shields its network call. A
# step that has not honoured the request after this long is holding the
# shutdown hostage, so it is abandoned and reported instead of waited for
# again; nothing it was using is freed on its account.
CANCEL_GRACE_S = 0.5


def _budget(timeout: float | None) -> float:
    """The wait a caller asked for, or the default read at the time of asking.

    Read here rather than bound as a default argument so the deadline a
    deployment or a test sets on ``DRAIN_TIMEOUT_S`` is the one that applies. A
    default evaluated at import silently ignores both.
    """
    return DRAIN_TIMEOUT_S if timeout is None else timeout


@dataclass(eq=False)
class _Activity:
    """One in-flight step, identified so a nested drain can recognise it."""

    id: int
    done: asyncio.Event = field(default_factory=asyncio.Event)


class RoomActivity:
    """In-flight work per room, which teardown drains before it contradicts it.

    Used through :meth:`track`, which registers a step for as long as its block
    runs, and :meth:`drain`, which teardown awaits.
    """

    def __init__(self) -> None:
        self._rooms: dict[str, set[_Activity]] = {}

    @contextlib.asynccontextmanager
    async def track(self, room_id: str) -> AsyncIterator[None]:
        """Mark the block as in-flight work for a room.

        The generation check that guards the work belongs *inside* this block,
        not before it: a check outside would leave the window between passing it
        and being registered, which is the window this exists to remove.
        """
        activity = _Activity(id=next(_ids))
        self._rooms.setdefault(room_id, set()).add(activity)
        # Anything this block awaits — and any task spawned from it — inherits
        # the marker, which is how a teardown triggered from inside knows not
        # to wait for the block that triggered it.
        token = _enclosing.set(_enclosing.get() | {activity.id})
        try:
            yield
        finally:
            _enclosing.reset(token)
            self._done(room_id, activity)

    def spawn[T](self, room_id: str, work: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Run work that outlives its caller, registered before it starts.

        :meth:`track` registers when its block is *entered*, which for work on a
        task of its own is a turn after the task was created — and a teardown
        draining in that turn finds nothing to wait for. Worse, work spawned
        from inside a ``track`` block inherits nothing of it: the caller's
        registration ends when the caller does, so a cancelled caller leaves the
        work running against a room the drain has already declared quiet. A
        publication that finishes after ``leave()`` is that, and the drain
        exists precisely to make it impossible.

        So the room is holding this before the task exists, and goes on holding
        it until the work is done rather than until whoever asked for it is.

        The marker is set around the task's creation so the task inherits it: a
        teardown triggered from inside this work recognises it as enclosing and
        defers, exactly as it does for a ``track`` block.
        """
        activity = _Activity(id=next(_ids))
        self._rooms.setdefault(room_id, set()).add(activity)
        token = _enclosing.set(_enclosing.get() | {activity.id})
        try:
            return asyncio.ensure_future(self._held(room_id, activity, work))
        finally:
            _enclosing.reset(token)

    async def _held[T](self, room_id: str, activity: _Activity, work: Coroutine[Any, Any, T]) -> T:
        """Run work the room is already holding, and let go when it is done."""
        try:
            return await work
        finally:
            self._done(room_id, activity)

    def _done(self, room_id: str, activity: _Activity) -> None:
        """Stop holding one piece of in-flight work."""
        room = self._rooms.get(room_id)
        if room is not None:
            room.discard(activity)
            if not room:
                del self._rooms[room_id]
        activity.done.set()

    def enclosing(self, room_id: str) -> list[asyncio.Event]:
        """Completion signals for the room's work the caller is running inside.

        A teardown that finds any of these is re-entrant — it was triggered from
        within work it would otherwise wait for. It cannot wait here, but it
        must not simply carry on either: whatever it destroys becomes visible to
        the rest of that work's observers. What it can do is close admission now
        and finish destroying later, once these are set.
        """
        ids = _enclosing.get()
        return [a.done for a in self._rooms.get(room_id, set()) if a.id in ids]

    async def drain(self, room_id: str, *, timeout: float | None = None) -> None:
        """Wait for a room's in-flight work, except what the caller is inside.

        Admission is expected to be closed first (the generation bumped), so
        work that has not yet passed its check abandons itself rather than
        joining the queue this is draining.

        ``timeout`` is the budget for the whole wait, not for each item.

        Bounded because the work ends in code the channel does not own: a step
        that will not finish is reported and left behind. Note what that costs —
        past the deadline the ordering this exists to provide is no longer
        guaranteed, so a publication wedged in the backend may still land after
        ``leave()``. Nothing further starts (the abandoned flag is already set),
        but the chunk already in the backend's hands is beyond recall until a
        conference backend offers cancellation.
        """
        enclosing = _enclosing.get()
        pending = [a.done for a in self._rooms.get(room_id, set()) if a.id not in enclosing]
        await self._wait_all(pending, _budget(timeout), room_id)

    async def drain_all(self, *, timeout: float | None = None) -> None:
        """Wait for every room's in-flight work, on one shared budget."""
        enclosing = _enclosing.get()
        pending = [
            a.done
            for activities in self._rooms.values()
            for a in activities
            if a.id not in enclosing
        ]
        await self._wait_all(pending, _budget(timeout), "all rooms")

    async def wait_for(
        self, signals: list[asyncio.Event], *, timeout: float | None = None
    ) -> None:
        """Wait for named completion signals, on one shared budget.

        What a deferred teardown waits on: the signals :meth:`enclosing`
        handed it, which by then it is no longer nested inside.
        """
        await self._wait_all(list(signals), _budget(timeout), "deferred teardown")

    @staticmethod
    async def _wait_all(pending: list[asyncio.Event], timeout: float, what: str) -> None:
        """Wait for every signal, on one deadline shared between them."""
        if not pending:
            return
        waiters = [asyncio.ensure_future(event.wait()) for event in pending]
        try:
            _, unfinished = await asyncio.wait(waiters, timeout=timeout)
            if unfinished:
                logger.warning(
                    "Conference teardown for %s went ahead after %.0fs: %d piece(s) of "
                    "in-flight work have not finished, so anything they still emit or "
                    "publish arrives out of order",
                    what,
                    timeout,
                    len(unfinished),
                )
        finally:
            for waiter in waiters:
                waiter.cancel()
