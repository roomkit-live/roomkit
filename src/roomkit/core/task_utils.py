"""Shared asyncio task utilities."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import weakref
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any, Protocol

from roomkit.core.exceptions import KitClosedError

logger = logging.getLogger("roomkit.tasks")


class _Closable(Protocol):
    _closed: bool


class _HoldsRuns(Protocol):
    _background_runs: set[asyncio.Task[None]]
    _closed: bool


# How long a channel's close waits for the work it cancelled (its calls, its
# scheduled tasks) before it goes on and says what is still running.
CLOSE_WAIT_S = 5.0

# The one grace the kit's close gives its running work (a hook, a store
# write, a transport send) before it cuts it (RFC §12.10.4).
CLOSE_GRACE_S = 5.0

# How long the kit's close lets the work it cut unwind (a turn storing what it
# streamed, a run posting its terminal entry) before it goes on.
CUT_UNWIND_S = 1.0


def check_open(kit: _Closable, what: str = "background run") -> None:
    """Refuse to start *what* on a closing *kit*: started now, it would
    outlive it (RFC §19.7.3). Every door that starts a worker's turn on its
    own asks it: a strategy's background run, a delegation.

    Raises:
        KitClosedError: *kit* is closing.
    """
    if kit._closed:
        raise KitClosedError(f"The kit is closing: no {what} starts")


def hold_task(
    kit: _HoldsRuns,
    work: Coroutine[Any, Any, None],
    *,
    context: contextvars.Context | None = None,
    awaited: bool = False,
    eager: bool = False,
) -> asyncio.Task[None]:
    """Run *work* as a task *kit* holds until it ends, so its ``close()``
    cancels it with the other background work and waits for its end (RFC
    §19.7.3, §23.3). Held once the kit's close began, or after it, it is cut
    at once, as that close cuts the runs it finds: its first step runs (a
    work that must not start on a closing kit checks it there, as a
    strategy's background run does), then its end, and it is never left
    running past the close. *eager* starts it at once (:func:`start_named`).
    An exception it ends on
    is logged, unless its caller *awaited* it and receives it. *context*,
    when given, is the one it runs in instead of a copy of the caller's."""
    task = start_named(work, context=context, eager=eager)
    runs = kit._background_runs
    runs.add(task)
    task.add_done_callback(runs.discard)
    if not awaited:
        task.add_done_callback(log_task_exception)
    if kit._closed:
        cut_once_started(task)
    return task


def cut_once_started(task: asyncio.Task[Any]) -> None:
    """Cancel *task* once it has run its first step. Cancelled before it, a
    coroutine never runs its body, its own end included: the cancellation
    handler that posts a run's terminal entry, the ``finally`` that runs a
    task's completion. A task not started yet has its first step scheduled
    already, so a cancellation scheduled now comes right after it.

    A task whose cut is under way (scheduled, or delivered and not over) is
    left to end: a second cut (a close's pass finding a run held on the
    closing kit) would cut the end the first one started."""
    if task.cancelling() or task in _CUTS_SCHEDULED:
        return
    work = task.get_coro()
    if inspect.iscoroutine(work) and inspect.getcoroutinestate(work) == inspect.CORO_CREATED:
        _CUTS_SCHEDULED.add(task)
        asyncio.get_running_loop().call_soon(_cut_scheduled, task)
    else:
        task.cancel()


def _cut_scheduled(task: asyncio.Task[Any]) -> None:
    """The cut :func:`cut_once_started` scheduled after *task*'s first step."""
    _CUTS_SCHEDULED.discard(task)
    task.cancel()


_CUTS_SCHEDULED: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()
"""The tasks whose cut waits for their first step to have run."""


def start_named[T](
    work: Coroutine[Any, Any, T],
    *,
    name: str | None = None,
    context: contextvars.Context | None = None,
    eager: bool = False,
) -> asyncio.Task[T]:
    """Start *work* as a task named in its own context, so the code it runs
    and every task it starts read it in :func:`held_runs`: a ``close()`` from
    under it spares it rather than wait for itself. *eager* runs its first
    step at once. *context*, when given, is the one it runs in instead of a
    copy of the caller's."""
    # Named before it starts (a cell, since the task does not exist until it
    # is created), so what runs under it reads it from its first step on.
    run_context = context if context is not None else contextvars.copy_context()
    cell: list[asyncio.Task[Any]] = []
    run_context.run(_HELD_RUNS.set, (*run_context.get(_HELD_RUNS, ()), cell))
    task = asyncio.Task(
        work,
        loop=asyncio.get_running_loop(),
        name=name,
        context=run_context,
        eager_start=eager,
    )
    cell.append(task)
    return task


async def run_to_its_end[T](
    work: Coroutine[Any, Any, T],
    *,
    held: set[asyncio.Task[Any]],
    name: str,
    expected: tuple[type[Exception], ...] = (),
) -> T:
    """Run *work* to its end whatever becomes of its caller, and return what
    it returns: the writes past a call's decision (RFC §13.6, A caller that
    stops waiting).

    A cancellation of the caller, a timeout around it included, waits for
    the work's end and is raised then, so that a room lock the caller holds
    is released only once the writes are all done; a second one waits too.
    The work runs in a copy of the caller's context, its lock's lease and its
    frames included, as a task *name* started at once and named in
    :func:`held_runs` (a ``close()`` called from under it spares it). Until
    it ends it is in *held*, which the kit's close gives its grace, then
    cuts: a caller nobody cancelled then gets :class:`KitClosedError`. Its
    exception is raised, as
    awaiting it would; to a cancelled caller its cancellation is, and the
    exception is logged: as an error, unless it is one of *expected*, an end
    the caller would have handled (a race another call won), at debug.
    """
    task = start_named(work, name=name, eager=True)
    if task.done():
        return task.result()
    held.add(task)
    task.add_done_callback(held.discard)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError as exc:
            # The caller's own: the writes finish first.
            cancelled = cancelled or exc
    if cancelled is None:
        if task.cancelled():
            # Cut by the kit's close once its grace ran out, the caller
            # cancelled by nobody (RFC §12.10.4).
            raise KitClosedError("The kit closed before these writes ended")
        return task.result()
    if not task.cancelled() and (error := task.exception()) is not None:
        level = logging.DEBUG if isinstance(error, expected) else logging.ERROR
        logger.log(
            level,
            "Writes a cancelled caller left failed: %s",
            describe_error(error),
            exc_info=error,
        )
    raise cancelled


def describe_error(error: BaseException) -> str:
    """*error* for a log line: its type, and its message when it has one (a
    ``TimeoutError`` often has none)."""
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def close_unstarted(*works: object) -> None:
    """Close each of *works* that is a coroutine never started: what was to
    await it ended first (a task cancelled before its first step, a start
    given up), and nothing will now. Python would warn, at an unrelated
    moment, that it was never awaited."""
    for work in works:
        if inspect.iscoroutine(work) and inspect.getcoroutinestate(work) == inspect.CORO_CREATED:
            work.close()


async def _finish_cleanup(coro: Coroutine[Any, Any, object]) -> None:
    """Finish releasing resources before propagating a caller's cancellation."""
    task = asyncio.create_task(coro, name="resource_cleanup")
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def cancel_and_wait(
    *tasks: asyncio.Future[Any] | None,
    log_errors_to: logging.Logger | None = None,
    once_started: bool = False,
) -> None:
    """Cancel *tasks* and wait until each has ended, without eating the caller's cancellation.

    ``task.cancel()`` then ``with suppress(CancelledError): await task`` also
    swallows a cancellation aimed at the caller: the ``CancelledError`` that
    reaches it while it waits cannot be told from the one the task raises, so
    the caller carries on as if nobody had cancelled it. ``asyncio.wait``
    never raises a task's outcome, so here the two are told apart: the tasks
    still end before the caller moves on, as awaiting them guaranteed, and
    the caller's cancellation is raised once they have.

    ``None`` entries and the tasks the caller runs as (:func:`acting_tasks`)
    are skipped: a task tearing itself down cannot wait for its own end, nor a
    shutdown for a caller waiting on it. One a caller that joins the shutdown
    meanwhile runs as, or under (:func:`spared_by_close`), stops being waited
    for. *once_started* cancels a task not
    started yet after its first step (:func:`cut_once_started`), so its own
    end runs. A task's own exception is raised, as
    awaiting it would, unless *log_errors_to* is given: it is then logged
    there, at debug, and the teardown goes on. A cancelled caller gets its
    cancellation, never a task's exception.

    Like a ``TaskGroup``'s exit, the teardown is not cut short: a second
    cancellation of the caller, or a timeout around it, waits for the tasks
    too, so a task that never ends holds its caller.
    """
    acting = acting_tasks()
    pending = [t for t in dict.fromkeys(tasks) if t is not None and t not in acting]
    for task in pending:
        if once_started and isinstance(task, asyncio.Task):
            cut_once_started(task)
        else:
            task.cancel()
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            # A task a caller that joined meanwhile runs as, or under, is
            # spared, not waited for: it waits for this teardown in turn.
            await wait_for_unspared(lambda: set(pending) - spared_by_close())
            break
        except asyncio.CancelledError as exc:
            # The caller's own: the teardown finishes first
            cancelled = cancelled or exc
    errors: list[BaseException] = [
        error
        for t in pending
        if t.done() and not t.cancelled() and (error := t.exception()) is not None
    ]
    if cancelled is not None:
        for error in errors:
            (log_errors_to or logger).debug(
                "Task failed before its cancellation: %s", error, exc_info=error
            )
        raise cancelled
    if errors and log_errors_to is None:
        raise errors[0]
    for error in errors:
        (log_errors_to or logger).debug(
            "Task failed before its cancellation: %s", error, exc_info=error
        )


_CLOSE_CALLERS: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()
"""The tasks a kit's ``close()`` is called from, until it returns."""


def calling_close() -> contextlib.AbstractContextManager[None]:
    """Around a kit's ``close()``: the calling task is one a turn that ends,
    or a channel that closes, never cuts (RFC §12.10.4): cutting it would
    cut the close."""
    return _CallingClose(asyncio.current_task())


class _CallingClose:
    __slots__ = ("_first", "_task")

    def __init__(self, task: asyncio.Task[Any] | None) -> None:
        self._task = task
        self._first = False

    def __enter__(self) -> None:
        if self._task is not None and self._task not in _CLOSE_CALLERS:
            self._first = True
            _CLOSE_CALLERS.add(self._task)

    def __exit__(self, *exc: object) -> None:
        if self._first and self._task is not None:
            _CLOSE_CALLERS.discard(self._task)


def close_callers() -> set[asyncio.Task[Any]]:
    """The tasks waiting for a kit's ``close()`` they called (:func:`calling_close`)."""
    return set(_CLOSE_CALLERS)


_HELD_RUNS: contextvars.ContextVar[tuple[list[asyncio.Task[Any]], ...]] = contextvars.ContextVar(
    "roomkit_held_runs", default=()
)
"""The kit-held runs the current code runs under, outermost first, each in
the cell :func:`hold_task` fills once it has created the run's task."""


def cancel_reason(cancelled: BaseException) -> str | None:
    """The reason a cancellation names, when its canceller gave one
    (``task.cancel(reason)``): what a turn tells its executor (RFC §10.6.8)."""
    reason = cancelled.args[0] if cancelled.args else None
    return reason if isinstance(reason, str) and reason else None


def under_no_held_run(context: contextvars.Context) -> contextvars.Context:
    """*context*, its code under no kit-held run: a task started in it is
    not the work of the run that started it, which does not wait for it,
    and a ``close()`` it calls waits for that run as for any other
    (RFC §12.10.4): an observer's, fired and forgotten."""
    context.run(_HELD_RUNS.set, ())
    return context


def held_runs() -> tuple[asyncio.Task[Any], ...]:
    """The kit-held runs (:func:`hold_task`) the current code runs under: a
    ``close()`` called from one of them must not wait for it."""
    return tuple(task for cell in _HELD_RUNS.get() for task in cell)


class Callers:
    """The callers of work every one of them waits for (the kit's shutdown,
    which every ``close()`` waits for): the one that started it and each one
    that joined it since, with the kit-held runs each runs under and the
    context it joined from. What the work waits for or cuts spares them all
    (:func:`acting_tasks`, :func:`spared_by_close`, and a rule that reads a
    marker of the code it is called from, :func:`callers_read`), as each
    caller's own code would spare itself, and its waits wake as one joins
    (:func:`wait_for_unspared`): the work of a caller that joins while it is
    waited for stops being waited for."""

    def __init__(self) -> None:
        self.acting: set[asyncio.Task[Any]] = set()
        self.under: set[asyncio.Task[Any]] = set()
        self.contexts: list[contextvars.Context] = []
        self._joined: asyncio.Future[None] | None = None

    def start[T](self, work: Coroutine[Any, Any, T], *, name: str) -> asyncio.Task[T]:
        """Join, then start *work* as a task acting for the callers, named
        in :func:`held_runs`. Not eager: its first step runs once the caller
        holds the task, so a call it makes back reads it."""
        self.join()
        context = contextvars.copy_context()
        context.run(_ACTS_FOR.set, self)
        return start_named(work, name=name, context=context)

    def join(self) -> None:
        """Count the current code among the callers: the tasks it runs as,
        the kit-held runs it runs under, and its context, whose markers (the
        turns it runs under, the realtime call it serves, the conference
        activity it runs inside) spare its work too (RMK-1019)."""
        acting, under = acting_tasks(), set(held_runs())
        if acting <= self.acting and under <= self.under:
            return
        self.acting |= acting
        self.under |= under
        self.contexts.append(contextvars.copy_context())
        if self._joined is not None:
            self._joined.set_result(None)
            self._joined = None

    def joined(self) -> asyncio.Future[None]:
        """A future done once the next caller joins."""
        if self._joined is None:
            self._joined = asyncio.get_running_loop().create_future()
        return self._joined


_ACTS_FOR: contextvars.ContextVar[Callers | None] = contextvars.ContextVar(
    "roomkit_acts_for", default=None
)
"""The callers the current code's task acts for (:meth:`Callers.start`)."""


def callers_read[T](marker: contextvars.ContextVar[T]) -> list[T]:
    """*marker* as the current code reads it and, inside work its callers
    wait for (:class:`Callers`), as each caller read it when it joined: a
    rule that spares the work a ``close()`` is called from by a marker of
    that code spares the work of every ``close()`` the shutdown answers, not
    only the first one's (RMK-1019). Read again at each wake of a wait
    (:func:`wait_for_unspared`), it counts a caller that joined meanwhile."""
    values = [marker.get()]
    callers = _ACTS_FOR.get()
    if callers is not None:
        values.extend(context.run(marker.get) for context in callers.contexts)
    return values


def acting_tasks() -> set[asyncio.Task[Any]]:
    """The tasks the current code runs as: the current one and, inside work
    started for callers that wait for it (:class:`Callers`), theirs. A
    ``close()`` called from here neither waits for nor cuts them."""
    current = asyncio.current_task()
    callers = _ACTS_FOR.get()
    return {
        *(() if current is None else (current,)),
        *(() if callers is None else callers.acting),
    }


def spared_by_close() -> set[asyncio.Task[Any]]:
    """What a ``close()`` called from the current code neither waits for nor
    cuts: the tasks it runs as (:func:`acting_tasks`) and the kit-held runs it
    runs under (:func:`held_runs`), its callers' included."""
    callers = _ACTS_FOR.get()
    return {*acting_tasks(), *held_runs(), *(() if callers is None else callers.under)}


async def wait_for_unspared(
    pending: Callable[[], Iterable[asyncio.Future[Any]]], *, timeout: float | None = None
) -> set[asyncio.Future[Any]]:
    """Wait until *pending*, read again at each wake, holds nothing still
    running, or until *timeout*: what it still holds then, never cut.

    It wakes as one of them ends and, inside work its callers wait for
    (:class:`Callers`), as a caller joins: *pending*, reading
    :func:`acting_tasks` or :func:`spared_by_close`, then drops the work of a
    caller that joined while it was waited for, which waits for this work in
    turn. A cancellation of the waiter is raised.
    """
    loop = asyncio.get_running_loop()
    until = None if timeout is None else loop.time() + timeout
    callers = _ACTS_FOR.get()
    while running := {future for future in pending() if not future.done()}:
        remaining = None if until is None else until - loop.time()
        if remaining is not None and remaining <= 0:
            return running
        wakes = running if callers is None else {*running, callers.joined()}
        await asyncio.wait(wakes, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
    return set()


def _cancellation_requests() -> int:
    """The current task's pending cancellation requests; 0 outside a task."""
    task = asyncio.current_task()
    return task.cancelling() if task is not None else 0


def cancels_current_task(error: BaseException) -> bool:
    """Whether *error* is a cancellation aimed at the current task, which it
    lets through, rather than a ``CancelledError`` the work it awaited
    raised of its own: that one is the work's failure."""
    return isinstance(error, asyncio.CancelledError) and _cancellation_requests() > 0


async def await_interruptible(task: asyncio.Future[Any]) -> bool:
    """Await *task*, which someone else may cancel; whether they did
    (:func:`await_unless_cut`)."""
    cut, _ = await await_unless_cut(task)
    return cut


async def await_unless_cut[T](work: Awaitable[T]) -> tuple[bool, T | None]:
    """Await *work*, which someone else may cut: ``(True, None)`` when they
    did, ``(False, its value)`` otherwise.

    A cancellation from elsewhere (a playback interrupt, the kit's close)
    ends the wait quietly. One aimed at the caller reaches the work too, as
    awaiting it does, and is raised once the work has ended, even when the
    work swallowed it and returned: comparing the caller's pending
    cancellation requests before and after tells the two apart. The work's
    own exception is raised, as awaiting it would.
    """
    requested = _cancellation_requests()
    try:
        value = await work
    except asyncio.CancelledError:
        if _cancellation_requests() > requested:
            raise
        return True, None
    if _cancellation_requests() > requested:
        raise asyncio.CancelledError
    return False, value


def log_task_exception(task: asyncio.Task[Any]) -> None:
    """Done-callback that logs unhandled exceptions from fire-and-forget tasks.

    Attach to any :func:`asyncio.create_task` result to prevent silent
    exception loss::

        task = loop.create_task(some_coro())
        task.add_done_callback(log_task_exception)

    Cancelled tasks are silently ignored.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Unhandled exception in task %s: %s",
            task.get_name(),
            exc,
            exc_info=exc,
        )


_SHIELDED_IN_FLIGHT: set[asyncio.Task[None]] = set()
"""Work a cancelled task shields: held here so none is collected mid-run."""


async def shielded(work: Coroutine[Any, Any, None]) -> None:
    """Run *work* to its end even if the task awaiting it is cancelled meanwhile."""
    task = asyncio.ensure_future(work)
    _SHIELDED_IN_FLIGHT.add(task)
    task.add_done_callback(_SHIELDED_IN_FLIGHT.discard)
    await asyncio.shield(task)
