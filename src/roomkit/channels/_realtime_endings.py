"""What an ending does to the realtime tool calls it reaches, on every host
(RFC §12.4): cut them but the one whose handler caused it, and settle the
calls it spared when their channel closes."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from roomkit.channels._realtime_context import ending_cause, held_by
from roomkit.channels._realtime_tool_executor import report_interrupted_calls
from roomkit.core.task_utils import CLOSE_WAIT_S

if TYPE_CHECKING:
    from roomkit.channels._realtime_tool_calls import RealtimeToolCall
    from roomkit.channels._realtime_tool_executor import ToolCallHost


def interrupt_for_ending(
    calls: Iterable[RealtimeToolCall], tasks: Iterable[asyncio.Task[Any]]
) -> tuple[list[RealtimeToolCall], list[asyncio.Task[Any]]]:
    """Cancel what an ending reaches, on every host: each of *tasks* and of
    *calls* but the call whose handler caused the ending (:func:`ending_cause`,
    a task it started included), its tasks and the current one; it runs on
    to report its own outcome (RFC §12.4). The calls to report interrupted,
    and the tasks cancelled."""
    current = asyncio.current_task()
    cause = ending_cause()
    spared = {current, *held_by(cause)}
    cancelled = [task for task in tasks if task not in spared]
    for task in cancelled:
        task.cancel()
    return [call for call in calls if call is not cause and call.task is not current], cancelled


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
