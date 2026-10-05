"""Shared asyncio task utilities."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, Protocol

from roomkit.core.exceptions import RoomKitError

logger = logging.getLogger("roomkit.tasks")


class _Closable(Protocol):
    _closed: bool


# How long a channel's close waits for the work it cancelled (its calls, its
# scheduled tasks) before it goes on and says what is still running.
CLOSE_WAIT_S = 5.0


def check_open(kit: _Closable, what: str = "background run") -> None:
    """Refuse to start *what* on a closing *kit*: started now, it would
    outlive it (RFC §19.7.3). Every door that starts a worker's turn on its
    own asks it: a strategy's background run, a delegation.

    Raises:
        RoomKitError: *kit* is closing.
    """
    if kit._closed:  # noqa: SLF001
        raise RoomKitError(f"The framework is closing: no {what} starts")


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
    *tasks: asyncio.Future[Any] | None, log_errors_to: logging.Logger | None = None
) -> None:
    """Cancel *tasks* and wait until each has ended, without eating the caller's cancellation.

    ``task.cancel()`` then ``with suppress(CancelledError): await task`` also
    swallows a cancellation aimed at the caller: the ``CancelledError`` that
    reaches it while it waits cannot be told from the one the task raises, so
    the caller carries on as if nobody had cancelled it. ``asyncio.wait``
    never raises a task's outcome, so here the two are told apart: the tasks
    still end before the caller moves on, as awaiting them guaranteed, and
    the caller's cancellation is raised once they have.

    ``None`` entries and the current task are skipped: a task tearing itself
    down cannot wait for its own end. A task's own exception is raised, as
    awaiting it would, unless *log_errors_to* is given: it is then logged
    there, at debug, and the teardown goes on. A cancelled caller gets its
    cancellation, never a task's exception.

    Like a ``TaskGroup``'s exit, the teardown is not cut short: a second
    cancellation of the caller, or a timeout around it, waits for the tasks
    too, so a task that never ends holds its caller.
    """
    current = asyncio.current_task()
    pending = [t for t in dict.fromkeys(tasks) if t is not None and t is not current]
    for task in pending:
        task.cancel()
    cancelled: asyncio.CancelledError | None = None
    while any(not t.done() for t in pending):
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError as exc:
            # The caller's own: the teardown finishes first
            cancelled = cancelled or exc
    errors: list[BaseException] = [
        error for t in pending if not t.cancelled() and (error := t.exception()) is not None
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


def _cancellation_requests() -> int:
    """The current task's pending cancellation requests; 0 outside a task."""
    task = asyncio.current_task()
    return task.cancelling() if task is not None else 0


async def await_interruptible(task: asyncio.Future[Any]) -> bool:
    """Await *task*, which someone else may cancel; whether they did.

    A cancellation of *task* from elsewhere (a playback interrupt) ends the
    wait quietly and returns ``True``. One aimed at the caller reaches the
    task too, as awaiting it does, and is raised once the task has ended,
    even when the task swallowed it and returned: comparing the caller's
    pending cancellation requests before and after tells the two apart. The
    task's own exception is raised, as awaiting it would.
    """
    requested = _cancellation_requests()
    try:
        await task
    except asyncio.CancelledError:
        if _cancellation_requests() > requested:
            raise
        return True
    if _cancellation_requests() > requested:
        raise asyncio.CancelledError
    return False


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
