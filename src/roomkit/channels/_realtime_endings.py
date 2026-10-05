"""What an ending does to the realtime tool calls it reaches, on every host
(RFC §12.4): cut them but the one whose handler caused it, and settle the
calls it spared when their channel closes. The provider abandoning calls is
one such ending."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.channels._realtime_context import ending_cause, held_by, spare_own_orphaned_call
from roomkit.channels._realtime_tool_executor import report_interrupted_calls
from roomkit.core.task_utils import CLOSE_WAIT_S

if TYPE_CHECKING:
    from roomkit.channels._realtime_tool_calls import RealtimeToolCall, ToolCallBook
    from roomkit.channels._realtime_tool_executor import ToolCallHost

logger = logging.getLogger("roomkit.channels.realtime_tools")


def abandon_calls(
    book: ToolCallBook, session_id: str, call_ids: Iterable[str]
) -> list[RealtimeToolCall]:
    """Interrupt the calls the provider abandoned, on every host: each id is
    freed as the provider freed it, so nothing is sent for the call, and its
    handler is cancelled (RFC §12.4). The calls to report cancelled.

    A call whose outcome ON_TOOL_CALL already has is left to finish, sending
    nothing: a second report would put two outcomes on one call. A call whose
    own handler caused the reconnect that orphaned it is not abandoned: it
    runs on, its result kept off the wire (RFC §9.3).
    """
    abandoned: list[RealtimeToolCall] = []
    for call_id in call_ids:
        call = book.get(session_id, call_id)
        if call is None:
            logger.debug(
                "Cancelled tool call %s is not in flight for session %s", call_id, session_id
            )
            continue
        if _spared_by_own_reconnect(call):
            continue
        if book.release(session_id, call_id) is None or not call.interruptible:
            logger.debug(
                "Cancelled tool call %s already gave its outcome for session %s",
                call_id,
                session_id,
            )
            continue
        assert call.task is not None  # interruptible  # noqa: S101
        call.task.cancel()
        logger.info(
            "Tool call %s(%s) abandoned by the provider for session %s",
            call.name,
            call_id,
            session_id,
        )
        abandoned.append(call)
    return abandoned


def _spared_by_own_reconnect(call: RealtimeToolCall) -> bool:
    """Whether the call's own handler caused the reconnect that orphaned it.

    A call issued under its id after its result went out is another call,
    which the reconnect abandons as any other.
    """
    if not spare_own_orphaned_call(call):
        return False
    logger.info(
        "Tool call %s(%s) lost its id to the reconnect its own handler caused; "
        "the handler runs on and its result stays off the wire (session %s)",
        call.name,
        call.call_id,
        call.session.id,
    )
    return True


def interrupt_for_ending(
    calls: Iterable[RealtimeToolCall], tasks: Iterable[asyncio.Task[Any]]
) -> tuple[list[RealtimeToolCall], list[asyncio.Task[Any]]]:
    """Cut what an ending reaches, on every host: *calls* and *tasks* but the
    call whose handler caused the ending (:func:`ending_cause`, a task it
    started included) and its tasks, which run on to report its own outcome
    (RFC §12.4). Each interrupted call's own task is cut with them. The calls
    to report interrupted, and the tasks cancelled."""
    current = asyncio.current_task()
    cause = ending_cause()
    if cause is not None:
        cause.caused_ending = True
    interrupted = [call for call in calls if call is not cause and call.task is not current]
    own = [call.task for call in interrupted if call.task is not None]
    return interrupted, cut_tasks([*tasks, *own])


def cut_tasks(tasks: Iterable[asyncio.Task[Any]]) -> list[asyncio.Task[Any]]:
    """Cancel each of *tasks* an ending reaches but the current one and those
    holding the call that caused the ending; the tasks cancelled."""
    spared = {asyncio.current_task(), *held_by(ending_cause())}
    cancelled = [task for task in dict.fromkeys(tasks) if task not in spared]
    for task in cancelled:
        task.cancel()
    return cancelled


class SparedCalls:
    """The calls an ending spared that may still run, each held until its task
    ends: waited for, or cut and reported, when their channel closes (RFC
    §12.4)."""

    def __init__(self) -> None:
        self._calls: set[RealtimeToolCall] = set()

    def keep(self, calls: Iterable[RealtimeToolCall]) -> None:
        """Hold each of *calls* still running until its task ends."""
        for call in calls:
            task = call.task
            if task is None or task.done():
                continue
            self._calls.add(call)
            task.add_done_callback(partial(self._end, call))

    def _end(self, call: RealtimeToolCall, _task: asyncio.Task[Any]) -> None:
        self._calls.discard(call)

    def __len__(self) -> int:
        return len(self._calls)

    async def settle(self, host: ToolCallHost, why: str) -> None:
        """At the channel's close: :func:`settle_spared_calls` on those held."""
        calls, self._calls = self._calls, set()
        await settle_spared_calls(host, calls, why)


async def settle_spared_calls(
    host: ToolCallHost, calls: Iterable[RealtimeToolCall], why: str
) -> None:
    """At a close, wait within its bound for the calls an ending spared that
    still run, then interrupt the rest and report each once, as cancelled
    (RFC §12.4); never the call whose handler is closing, which runs on."""
    cause = ending_cause()
    current = asyncio.current_task()
    running = {
        call.task: call
        for call in calls
        if call is not cause and call.task is not None and call.task is not current
        if not call.task.done()
    }
    if not running:
        return
    _, pending = await asyncio.wait(running, timeout=CLOSE_WAIT_S)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=CLOSE_WAIT_S)
    await report_interrupted_calls(host, [running[task] for task in pending], why)
