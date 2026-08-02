"""In-memory realtime backend using asyncio queues."""

from __future__ import annotations

from uuid import uuid4

from roomkit.realtime._subscription import _Subscription
from roomkit.realtime.base import EphemeralCallback, EphemeralEvent, RealtimeBackend


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

        sub_ids = set(self._channels.get(channel, set()))
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
