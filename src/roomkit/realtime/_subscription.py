"""Per-subscription queue and drain task shared by realtime backends."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict

from roomkit.realtime.base import EphemeralCallback, EphemeralEvent

logger = logging.getLogger("roomkit.realtime")


class _Subscription:
    """Internal subscription handler with queue and background task."""

    def __init__(
        self,
        sub_id: str,
        channel: str,
        callback: EphemeralCallback,
        max_queue_size: int,
    ) -> None:
        self.sub_id = sub_id
        self.channel = channel
        self.callback = callback
        self._queue: OrderedDict[str, EphemeralEvent] = OrderedDict()
        self._max_queue_size = max_queue_size
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    async def enqueue(self, event: EphemeralEvent) -> None:
        """Add an event to the queue, dropping oldest if full."""
        if self._stopped:
            return

        # Drop oldest if at capacity (LRU-style)
        while len(self._queue) >= self._max_queue_size:
            self._queue.popitem(last=False)

        self._queue[event.id] = event
        self._event.set()

    def start(self) -> None:
        """Start the background task that drains the queue."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background task."""
        self._stopped = True
        self._event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        """Background task that drains the queue and invokes callbacks."""
        while not self._stopped:
            await self._event.wait()
            # Drain all queued events before clearing the wakeup flag.
            # Clearing after drain (with re-check) prevents lost wakeups
            # when a publish() calls event.set() between the last popitem()
            # and the clear().
            while self._queue and not self._stopped:
                self._event.clear()
                _, event = self._queue.popitem(last=False)
                try:
                    await self.callback(event)
                except Exception:
                    logger.exception("Error in realtime callback for subscription %s", self.sub_id)
            if not self._queue:
                self._event.clear()
