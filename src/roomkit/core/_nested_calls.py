"""The calls a task of a locked hook's body nests under the hook's section
(RFC §9.5.1 Reentrance, §13.6).

A task a hook's body spawns acts for the body while it runs
(:func:`~roomkit.core.locks.lent_while_it_runs`): a call it makes into the
room is nested under the section that holds the lock, as a call the body
awaits is. The body waits for it before its hook's result is used, so it
commits ahead of the event, under the lock its section holds; and it ends
with that section. Not committed yet when the section ends, it is cut there,
nothing of it stored, and nothing reports it but what ended the section:
cut once the section's deadline passed (the event's ``process_timeout``), it
is refused ``process_timeout``; cut otherwise (the event's caller gone), it
reads that cancellation, as a call the body awaited would.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.locks import _Call


class NestedCall:
    """One call a task of a hook's body nested under the body's section:
    the task it runs in, whether it is still before its commit, and the
    section's deadline."""

    __slots__ = ("_before_commit", "_cut", "deadline", "over", "task")

    def __init__(self, before_commit: Callable[[], bool], deadline: float | None) -> None:
        self.task = asyncio.current_task()
        self.over: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.deadline = deadline
        self._before_commit = before_commit
        self._cut = False

    def cut(self) -> None:
        """End the call with its section, unless it reached its commit: a
        committed call goes on to its end."""
        if self.over.done() or self.task is None or not self._before_commit():
            return
        self._cut = True
        self.task.cancel()

    def timed_out(self) -> bool:
        """Read in the call on its cancellation: whether it is the
        section's cut once the section's deadline passed, taken back, so that
        the call is refused ``process_timeout``. Any other cancellation, the
        cut of a section its caller left included, stays raised."""
        if not self._cut or self.task is None or self.deadline is None:
            return False
        if asyncio.get_running_loop().time() < self.deadline:
            return False
        self._cut = False
        return self.task.uncancel() == 0


@contextmanager
def nested_in(
    call: _Call | None, before_commit: Callable[[], bool] | None, deadline: float | None
) -> Generator[NestedCall | None, None, None]:
    """Run a call into the room as one nested under *call*'s section, a
    hook's body a task of which makes it (``Entry.follows``), noted there
    until it is over; *before_commit* tells whether it is still before its
    commit, *deadline* is the section's. ``None`` for a call no section
    waits for."""
    if call is None or before_commit is None:
        yield None
        return
    nested = NestedCall(before_commit, deadline)
    call.nested.add(nested)
    try:
        yield nested
    finally:
        call.nested.discard(nested)
        nested.over.set_result(None)


async def nested_calls_over(call: _Call | None) -> None:
    """Wait for the calls the tasks of *call*, a hook's body, nested under
    its section; cancelled meanwhile, cut those not committed yet, then
    raise the cancellation."""
    if call is None:
        return
    try:
        while call.nested:
            await asyncio.wait([nested.over for nested in call.nested])
    except asyncio.CancelledError:
        cut_nested_calls(call)
        raise


def cut_nested_calls(call: _Call | None) -> None:
    """Cut the calls the tasks of *call* nested under its section that are
    not committed yet: the section ends without them."""
    if call is None:
        return
    for nested in list(call.nested):
        nested.cut()
