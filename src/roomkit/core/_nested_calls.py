"""The calls a task of a locked hook's body nests under the hook's section
(RFC §9.5.1 Reentrance, §13.6).

A task a hook's body spawns acts for the body while it runs
(:func:`~roomkit.core.locks.lent_while_it_runs`): a call it makes into the
room then is nested under the section that holds the lock, as a call the
body awaits is, and keeps the lock lent to it for as long as it runs
(:func:`~roomkit.core.locks.relent_to`); a call the task makes once the body
is over is a new arrival. The body's section waits for such a call before
the hook's outcome is used, however the body ended, so that it commits ahead
of the event; and it is bounded by the section's deadline (the event's
``process_timeout``). Not committed when that deadline passes, it is cut
there, refused ``process_timeout``, nothing of it stored and nothing
reported but the event's ``process_timeout``; so is a call nested in turn
under its own section. A section cancelled before its deadline, its event's
caller gone, holds its lock for the call until it ends or that deadline
passes, where it is cut, nothing of it reported, as roomkit-rs does.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from roomkit.core.locks import relent_to

if TYPE_CHECKING:
    from roomkit.core.locks import _Call


class NestedCall:
    """One call a task of a hook's body nested under the body's section:
    the task it runs in, whether it is still before its commit, and the
    section's deadline."""

    __slots__ = ("_before_commit", "_committing", "_cut", "deadline", "over", "task")

    def __init__(self, before_commit: Callable[[], bool], deadline: float | None) -> None:
        self.task = asyncio.current_task()
        self.over: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.deadline = deadline
        self._before_commit = before_commit
        self._committing = False
        self._cut = False

    def committing(self) -> None:
        """The call writes its record outside the pipeline's commit (past
        the reentry budget): from now on no section cuts it."""
        self._committing = True

    @property
    def cut_at_its_deadline(self) -> bool:
        """Whether its section cut it once the section's deadline passed."""
        return self._cut

    def cut(self, past_a_deadline: bool) -> None:
        """End the call with its section once the section's deadline passed,
        or the deadline of a section it is nested in (*past_a_deadline*),
        unless it reached its commit, which goes on to its end. A section
        that ends before its deadline leaves it to go on."""
        loop = asyncio.get_running_loop()
        due = past_a_deadline or (self.deadline is not None and loop.time() >= self.deadline)
        if not due or self.over.done() or self.task is None:
            return
        if self._committing or not self._before_commit():
            return
        self._cut = True
        self.task.cancel()

    def timed_out(self) -> bool:
        """Read in the call on its cancellation: whether it is its
        section's cut, taken back, so that the call is refused
        ``process_timeout``. Any other cancellation stays raised."""
        if not self._cut or self.task is None:
            return False
        return self.task.uncancel() == 0


_current: contextvars.ContextVar[NestedCall | None] = contextvars.ContextVar(
    "_roomkit_nested_call", default=None
)
"""The nested call the current code runs within, if any: a hook its section
runs, cut with it, cuts the calls its own tasks nested."""


@contextmanager
def nested_in(
    call: _Call | None, before_commit: Callable[[], bool] | None, deadline: float | None
) -> Generator[NestedCall | None, None, None]:
    """Run a call into the room as one nested under *call*'s section, a
    hook's body a task of which makes it while the body runs
    (``Entry.follows``): noted there until it is over, the lock lent to it
    meanwhile; *before_commit* tells whether it is still before its commit,
    *deadline* is the section's. ``None`` for a call no section waits for."""
    if call is None or before_commit is None:
        yield None
        return
    nested = NestedCall(before_commit, deadline)
    call.nested.add(nested)
    token = _current.set(nested)
    try:
        with relent_to(call):
            yield nested
    finally:
        _current.reset(token)
        call.nested.discard(nested)
        nested.over.set_result(None)


async def nested_calls_over(call: _Call | None) -> None:
    """Wait for the calls the tasks of *call*, a hook's body, nested under
    its section; cancelled meanwhile, settle them as the section ends
    (:func:`nested_calls_after_a_cancel`), then raise the cancellation."""
    if call is None:
        return
    try:
        while call.nested:
            await asyncio.wait([nested.over for nested in call.nested])
    except asyncio.CancelledError:
        await nested_calls_after_a_cancel(call)
        raise


async def nested_calls_after_a_cancel(call: _Call | None) -> None:
    """Settle the calls the tasks of *call* nested under its section, the
    section cancelled: past its deadline, cut those not committed yet; before
    it (its event's caller gone), wait for them, the section's lock still
    held, up to that deadline, past which they are cut, nothing of them
    reported, as roomkit-rs's operation runs them on under its lock within
    the event's deadline.

    Cancelled again meanwhile (a cancel scope that repeats its cancellation,
    the event's own ``process_timeout`` falling at that deadline), it waits
    on all the same and raises the first such cancellation once they are
    settled: the wait sets no timer of its own that cancels the task."""
    if call is None or not call.nested:
        return
    deadline = next(iter(call.nested)).deadline
    within = _current.get()
    past = within is not None and within.cut_at_its_deadline
    loop = asyncio.get_running_loop()
    if past or (deadline is not None and loop.time() >= deadline):
        cut_nested_calls(call)
        return
    cancelled: asyncio.CancelledError | None = None
    while call.nested:
        left = None if deadline is None else deadline - loop.time()
        if left is not None and left <= 0:
            cut_nested_calls(call, past_the_deadline=True)
            break
        try:
            await asyncio.wait([nested.over for nested in call.nested], timeout=left)
        except asyncio.CancelledError as cancel:
            cancelled = cancelled or cancel
    if cancelled is not None:
        raise cancelled


def cut_nested_calls(call: _Call | None, *, past_the_deadline: bool = False) -> None:
    """Cut, as their section ends, the calls the tasks of *call* nested
    under it that are not committed yet, once the section's deadline
    passed (*past_the_deadline*, when its wait said so) or the nested call
    the current code runs within was cut at its own."""
    if call is None:
        return
    within = _current.get()
    past_a_deadline = past_the_deadline or (within is not None and within.cut_at_its_deadline)
    for nested in list(call.nested):
        nested.cut(past_a_deadline)
