"""What the kit's ``close()`` cuts (RFC §12.10.4): the calls into a room
under way, and the deliveries of the calls it took.

``close()`` stops admission at once: a call that waits for its room's lock
is cut then, and a call still running its hooks or its writes once the
close's grace ran out is cut then. :class:`DoorRuns` holds them. Every
delivery under way is cut at once too, its turns first: :class:`Cascades`
holds them.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any

from roomkit.core import task_utils
from roomkit.core._reentrance import ReentryBudget
from roomkit.core.exceptions import KitClosedError, TurnLinkError
from roomkit.core.lanes import DeliveryCascade
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import EventStatus
from roomkit.models.event import RoomEvent

logger = logging.getLogger("roomkit.framework")

KIT_CLOSED = KitClosedError.code


def cut_by_close() -> InboundResult:
    """The result of a call into a room the kit's close cut before its
    commit (RFC §12.10.4): nothing of it is stored."""
    return InboundResult(blocked=True, reason=KIT_CLOSED, cancellation_reason=KIT_CLOSED)


def ended_by_close(kit: Any, exc: BaseException) -> bool:
    """Whether *exc* is the kit's own close ending the work that raised it:
    a call refused or cut, the store it released, the turn backend it closed
    (RFC §12.10.4). Expected, so logged at debug, never as a failure."""
    if not kit._closed:
        return False
    if isinstance(exc, KitClosedError):
        return True
    if isinstance(exc, TurnLinkError) and exc.reason == "backend_closed":
        return True
    return isinstance(exc, RuntimeError) and kit._store_released


def refuse_closed(closed: bool, what: str = "call") -> None:
    """Refuse a *what* on a kit whose ``close()`` has started.

    Raises:
        KitClosedError: the kit is closing or closed.
    """
    if closed:
        raise KitClosedError(f"The kit is closing: no {what} is taken")


class Cascades:
    """The deliveries of the calls the kit took, each its cascade: the ones
    still under way are cut at once when the kit closes (RFC §12.10.4)."""

    def __init__(self) -> None:
        self._live: weakref.WeakSet[DeliveryCascade] = weakref.WeakSet()
        self._cut_turns: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()
        self.closed = False

    def new(self, room_id: str, *, reentry_budget: int | ReentryBudget) -> DeliveryCascade:
        """A cascade for a call into *room_id*; cut at once when the kit is
        closing already: its turns never start, its delivery is dropped."""
        cascade = DeliveryCascade(room_id, reentry_budget=reentry_budget)
        self._live.add(cascade)
        if self.closed:
            cascade.cancel(KIT_CLOSED)
        return cascade

    def cut(self, spared: set[asyncio.Task[Any]]) -> None:
        """Cut every cascade under way, its turns at once (the turn work in
        *spared*, the tasks the close runs as, left to end)."""
        self.closed = True
        for cascade in list(self._live):
            if cascade.in_flight and cascade.cancelled is None:
                self._cut_turns.update(cascade.turn_work() - spared)
                cascade.cancel(KIT_CLOSED, spared=spared, wake=False)

    def unwinding(self) -> set[asyncio.Task[Any]]:
        """The turns the close cut still ending: what a cut turn leaves (its
        text, its open calls' end rows) is written before the store is
        released, the turn a close's own call runs in included."""
        return {task for task in self._cut_turns if not task.done()}

    def running(self) -> set[asyncio.Task[Any]]:
        """The work of the cascades the close cut, still running: a cut turn
        storing what it streamed, a delivery set or its aftermath unwinding
        (a send, an observer that ignores its cut)."""
        return {
            task
            for cascade in list(self._live)
            if cascade.cancelled is not None
            for task in cascade.running()
        }


def _event_lost(record: _Run, committed: RoomEvent | None) -> bool:
    """Whether the commit *record* started stored no event it can read back."""
    return record.committing is not None and committed is None


class CallCutError(KitClosedError):
    """A call into a room the kit's close cut (RFC §12.10.4). *committed*
    is the event its commit stored, read back once it was cut, or ``None``
    when no event of it is stored; *blocked_reason*, the reason its hooks
    blocked it for, when its commit was the block's record."""

    def __init__(
        self,
        message: str,
        committed: RoomEvent | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.committed = committed
        self.blocked_reason = blocked_reason


class _Run:
    """One call into a room under way, the event its commit writes and, for
    a block's record, the reason its hooks blocked it for."""

    __slots__ = ("blocked_reason", "committing", "task")

    def __init__(self, task: asyncio.Task[Any]) -> None:
        self.task = task
        self.committing: RoomEvent | None = None
        self.blocked_reason: str | None = None


_RUN: contextvars.ContextVar[_Run | None] = contextvars.ContextVar(
    "roomkit_call_run", default=None
)
"""The call the current code runs for, the one its commit is; ``None`` in a
call made from inside another in the same task (a hook's), whose commit is
not the outer call's."""


class DoorRuns:
    """The calls the kit takes that write or deliver, from the call's start
    to its commit: what ``close()`` cuts (RFC §12.10.4). One waiting for a
    room's lock is cut at once; one still running (its hooks, its writes)
    once the close's grace ran out. A cut call ends with :class:`CallCutError`,
    never a ``CancelledError`` its caller did not ask for; one whose commit
    landed as it was cut names the event stored.

    *read_event* reads a stored event back by its id; *spared* names the
    work the close is called from."""

    def __init__(
        self,
        read_event: Callable[[str], Awaitable[RoomEvent | None]],
        spared: Callable[[], set[asyncio.Task[Any]]],
    ) -> None:
        self._read_event = read_event
        self._spared = spared
        self._runs: dict[asyncio.Task[Any], _Run] = {}
        self._running: set[asyncio.Task[Any]] = set()
        self._waiting: set[asyncio.Task[Any]] = set()
        self._unwinding: set[asyncio.Task[Any]] = set()
        # Each call the close cut, and when: its read-back ends with its unwind.
        self._cut: dict[asyncio.Task[Any], float] = {}
        self.closed = False

    @contextlib.asynccontextmanager
    async def run(self, what: str = "call", *, own: bool = False) -> AsyncGenerator[None]:
        """Within it, the current task runs a call, cut as the kit closes.
        One made inside another in the same task is part of the outer call's
        run: its commit is the outer call's when it carries out that call
        (``send_event`` writing through the framework's own door), not when
        it is a call of its *own* (a hook's, made from the outer call's
        pipeline). Refused on a closing kit.

        Raises:
            KitClosedError: the kit is closing.
            CallCutError: the close cut the call.
        """
        refuse_closed(self.closed, what)
        task = asyncio.current_task()
        nested = task is None or task in self._runs
        record = None
        if not nested:
            assert task is not None
            record = _Run(task)
            self._runs[task] = record
            self._running.add(task)
        token = _RUN.set(record) if record is not None or own else None
        try:
            yield
        except asyncio.CancelledError:
            cut_at = None if record is None else self._take_cut(task)
            if record is None or cut_at is None:
                raise
            # Its unwind: a commit that landed as it was cut is read back,
            # so the call reports it stored (RFC §10.1, §12.10.4). Its writes
            # run to their end under it (run_to_its_end), cut and unwound
            # first: the read comes after them.
            self._unwinding.add(record.task)
            committed = await self._read_back(record, cut_at)
            raise CallCutError(
                f"The kit closed before this {what} ended"
                # Only its event's fate is known: a room change, by then, may
                # have written what it writes before its first context.
                + (": its event is not stored" if _event_lost(record, committed) else ""),
                committed,
                record.blocked_reason,
            ) from None
        finally:
            if token is not None:
                _RUN.reset(token)
            if record is not None:
                self._leave(record.task)

    def committing(self, event: RoomEvent, *, blocked_reason: str | None = None) -> None:
        """The current call writes *event* as its commit, the record of a
        block for *blocked_reason* when given: cut from now on, it reads the
        event back to report it stored."""
        record = _RUN.get()
        if record is not None:
            record.committing = event
            record.blocked_reason = blocked_reason

    def before_commit(self) -> Callable[[], bool] | None:
        """For the call the current task runs: a test, read from elsewhere,
        of whether a section that ends it with itself may still cut it
        (:mod:`roomkit.core._nested_calls`): before its commit, and while
        the kit is not closing, whose close cuts it itself (RFC §12.10.4).
        ``None`` for a call made inside another in the same task, whose
        commit is not its own."""
        record = _RUN.get()
        if record is None or record.task is not asyncio.current_task():
            return None
        return lambda: record.committing is None and not self.closed

    def committed(self) -> None:
        """The current call is past its commit: the close no longer cuts it,
        its delivery is its cascade's. A nested call's commit is not its
        outer call's."""
        record = _RUN.get()
        if record is not None:
            self._running.discard(record.task)

    @contextlib.contextmanager
    def waiting_for_lock(self, room_id: str) -> Generator[None]:
        """Around the current code's wait for *room_id*'s lock: a call
        waiting so is cut at once as the kit closes, and one that reaches
        it once the close started is refused, but the work the close is
        called from. The caller asks it only for a lock the code does not
        hold already, checked against the lock's own manager: a room lock
        named like an identity's key is no hold of that identity's lock.

        Raises:
            KitClosedError: the kit is closing.
        """
        task = asyncio.current_task()
        if task is None or task not in self._running:
            yield
            return
        if self.closed and task not in self._spared():
            raise CallCutError(
                "The kit closed before this call took its room's lock: nothing of it is stored"
            )
        self._running.discard(task)
        self._waiting.add(task)
        try:
            yield
        finally:
            self._waiting.discard(task)
            if task in self._runs:
                self._running.add(task)

    def cut_waiting(self, spared: set[asyncio.Task[Any]]) -> None:
        """Stop admission (RFC §12.10.4): the calls waiting for their room's
        lock are cut at once."""
        self.closed = True
        for task in list(self._waiting - spared):
            self._cut_task(task)

    def cut_running(self, spared: set[asyncio.Task[Any]]) -> None:
        """Cut every call still running before its commit, but the ones
        *spared* (a call the close runs under) and the ones unwinding."""
        self.closed = True
        for task in list((self._running | self._waiting) - spared - self._unwinding):
            self._cut_task(task)

    def running(self) -> set[asyncio.Task[Any]]:
        """The calls still running before their commit, or unwinding."""
        return {
            task for task in self._running | self._waiting | self._unwinding if not task.done()
        }

    async def _read_back(self, record: _Run, cut_at: float) -> RoomEvent | None:
        """The event *record*'s commit stored, read by its id within the
        unwind the close gave the call from its cut at *cut_at*; ``None``
        when it is not stored, or cannot be read in time (logged: the call
        then reports nothing stored, whatever the store holds)."""
        event = record.committing
        if event is None:
            return None
        try:
            async with asyncio.timeout_at(cut_at + task_utils.CUT_UNWIND_S):
                stored = await self._read_event(event.id)
        except Exception as error:  # noqa: BLE001 - a read that fails reports nothing stored
            logger.warning(
                "The kit closed as a call committed event %s, and reading it back failed "
                "(%s): the call reports nothing stored",
                event.id,
                type(error).__name__,
            )
            return None
        if stored is None or stored.room_id != event.room_id:
            return None
        return stored

    def _cut_task(self, task: asyncio.Task[Any]) -> None:
        if task in self._cut or task.done():
            return
        self._cut[task] = asyncio.get_running_loop().time()
        task.cancel(KIT_CLOSED)

    def _take_cut(self, task: asyncio.Task[Any] | None) -> float | None:
        """When the close cut *task*, if the cancellation it raised is the
        close's own: taken back, it is no cancellation of the caller.
        ``None`` when the close did not cut it, or a cancellation from
        elsewhere besides stays the caller's."""
        if task is None or task not in self._cut:
            return None
        cut_at = self._cut.pop(task)
        if task.cancelling() and task.uncancel() != 0:
            return None
        return cut_at

    def _leave(self, task: asyncio.Task[Any]) -> None:
        """The call *task* ran ended: a cut that never reached it (the call
        ended first) is taken back, so it cuts nothing of its caller's."""
        self._runs.pop(task, None)
        self._running.discard(task)
        self._waiting.discard(task)
        self._unwinding.discard(task)
        if self._cut.pop(task, None) is not None and task.cancelling():
            task.uncancel()


def inbound_cut_result(cut: KitClosedError) -> InboundResult:
    """What a call returning an ``InboundResult`` reads once the kit's close
    cut it (RFC §12.10.4): its event when its commit landed, ``kit_closed``;
    else a refusal, nothing stored."""
    if not isinstance(cut, CallCutError) or cut.committed is None:
        return cut_by_close()
    stored = cut.committed
    blocked = stored.status == EventStatus.BLOCKED
    return InboundResult(
        event=stored,
        blocked=blocked,
        reason=(cut.blocked_reason or stored.blocked_by) if blocked else None,
        cancellation_reason=KIT_CLOSED,
    )


def event_cut_result(cut: KitClosedError) -> RoomEvent:
    """What a call returning the committed event reads once the kit's close
    cut it (RFC §12.10.4): its event when its commit landed.

    Raises:
        KitClosedError: nothing of the call is stored.
    """
    stored = cut.committed if isinstance(cut, CallCutError) else None
    if stored is None:
        raise cut
    return stored
