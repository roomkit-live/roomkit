"""In-memory realtime backend using asyncio queues."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict
from uuid import uuid4

from roomkit.realtime.base import EphemeralCallback, EphemeralEvent, RealtimeBackend

logger = logging.getLogger("roomkit.realtime")


class InMemoryRealtime(RealtimeBackend):
    """In-process realtime backend using asyncio queues.

    Suitable for single-process deployments. For multi-process or
    distributed setups, provide a custom ``RealtimeBackend`` backed by
    Redis pub/sub, NATS, or similar.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        """Initialize the in-memory realtime backend.

        Args:
            max_queue_size: Maximum number of events to queue per subscription.
                Older events are dropped when the queue is full (LRU-style).
        """
        self._max_queue_size = max_queue_size
        self._subscriptions: dict[str, _Subscription] = {}
        self._channels: dict[str, set[str]] = {}  # channel -> subscription_ids
        self._closed = False

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        """Publish an event to all subscribers on a channel."""
        if self._closed:
            return

        sub_ids = self._channels.get(channel, set())
        for sub_id in sub_ids:
            sub = self._subscriptions.get(sub_id)
            if sub is not None:
                await sub.enqueue(event)

    async def subscribe(self, channel: str, callback: EphemeralCallback) -> str:
        """Subscribe to a channel with a callback.

        Returns:
            A subscription ID that can be used to unsubscribe.
        """
        sub_id = uuid4().hex
        sub = _Subscription(
            sub_id=sub_id,
            channel=channel,
            callback=callback,
            max_queue_size=self._max_queue_size,
        )
        self._subscriptions[sub_id] = sub

        if channel not in self._channels:
            self._channels[channel] = set()
        self._channels[channel].add(sub_id)

        sub.start()
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe and stop the subscription task.

        Returns:
            True if the subscription existed and was removed.
        """
        sub = self._subscriptions.pop(subscription_id, None)
        if sub is None:
            return False

        channel_subs = self._channels.get(sub.channel)
        if channel_subs:
            channel_subs.discard(subscription_id)
            if not channel_subs:
                del self._channels[sub.channel]

        await sub.stop()
        return True

    async def close(self) -> None:
        """Stop all subscriptions and clean up."""
        self._closed = True
        for sub in list(self._subscriptions.values()):
            await sub.stop()
        self._subscriptions.clear()
        self._channels.clear()

    @property
    def subscription_count(self) -> int:
        """Return the number of active subscriptions."""
        return len(self._subscriptions)


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
            self._event.clear()

            while self._queue and not self._stopped:
                _, event = self._queue.popitem(last=False)
                try:
                    await self.callback(event)
                except Exception:
                    logger.exception("Error in realtime callback for subscription %s", self.sub_id)
