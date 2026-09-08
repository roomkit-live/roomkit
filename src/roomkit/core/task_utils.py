"""Shared asyncio task utilities."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger("roomkit.tasks")


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
