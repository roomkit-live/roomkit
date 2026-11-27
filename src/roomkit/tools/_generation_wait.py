"""A turn's wait on its provider's next event, which a Cancel cuts (RFC §21.3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any


class _Cut(Enum):
    CUT = "cut"


CUT = _Cut.CUT
"""What :meth:`GenerationWait.next` answers once a Cancel cut the generation."""


class GenerationWait:
    """The task waiting on the provider's next event, while it waits.

    A steering ``Cancel`` aborts the current generation at once (RFC §21.3):
    a provider that has not handed its next event yet (a model still
    reasoning before its first token, a slow server) no longer holds the
    turn until it does. The cut cancels the waiting task at that wait alone,
    and the wait absorbs it; any other cancellation of the task still
    reaches its caller. A cut that comes while no task waits holds for the
    next wait, which then does not start; one the waiting task makes itself
    (its provider steering its own turn) takes effect at the provider's next
    event.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[Any] | None = None
        self._requested = False
        # Whether the waiting task was cancelled for the cut, and the
        # cancellation not withdrawn yet.
        self._cancelled = False

    def cut(self) -> None:
        """Cut the generation: its current wait, when a task other than the
        current one waits, and every wait after it."""
        self._requested = True
        task = self._task
        if task is None or self._cancelled or task is asyncio.current_task():
            return
        self._cancelled = True
        task.cancel()

    async def next[T](self, events: AsyncIterator[T]) -> T | _Cut:
        """The next of *events*, or :data:`CUT` once :meth:`cut` cut the
        generation.

        Raises:
            StopAsyncIteration: at the end of *events*, as ``anext()`` does.
        """
        task = asyncio.current_task()
        if self._requested:
            return CUT
        if task is None:
            return await anext(events)
        self._task = task
        try:
            return await anext(events)
        except asyncio.CancelledError:
            if self._withdraw(task):
                return CUT
            raise
        finally:
            self._task = None
            # A provider that swallowed the cut: its request is withdrawn.
            self._withdraw(task)

    def _withdraw(self, task: asyncio.Task[Any]) -> bool:
        """Withdraw the cut's cancellation of *task*: whether it was the
        task's only one."""
        if not self._cancelled:
            return False
        self._cancelled = False
        return task.uncancel() == 0
