"""Redis pub/sub realtime backend.

Distributes ephemeral events (typing, presence, reactions, thinking
deltas, ...) across processes via Redis pub/sub.

Requires ``redis>=5.0.1``::

    pip install roomkit[redis]

Usage::

    from roomkit.realtime import RedisRealtimeBackend

    kit = RoomKit(
        realtime=RedisRealtimeBackend("redis://localhost:6379"),
    )

Semantics
---------

Redis pub/sub is fire-and-forget: events published while a process is
disconnected are lost, and there is no replay. That matches the
ephemeral contract (RFC section 8.4) — typing indicators and presence
are safe to drop. Local subscribers receive events through the Redis
round-trip, so delivery is asynchronous even within the publishing
process (unlike ``InMemoryRealtime``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any
from uuid import uuid4

from roomkit.realtime._subscription import _Subscription
from roomkit.realtime.base import EphemeralCallback, EphemeralEvent, RealtimeBackend

logger = logging.getLogger("roomkit.realtime.redis")

_SUBSCRIBE_CONFIRM_TIMEOUT = 5.0
_READ_RETRY_DELAY = 1.0


def _as_str(value: Any) -> str:
    """Decode a Redis payload that may be ``bytes`` or ``str``."""
    return value if isinstance(value, str) else value.decode()


class RedisRealtimeBackend(RealtimeBackend):
    """Realtime backend backed by Redis pub/sub.

    A single shared ``PubSub`` connection and reader task serve all
    subscriptions in the process. Each subscription gets its own bounded
    queue and drain task (same as ``InMemoryRealtime``), so one slow
    callback never stalls the reader or other subscribers.

    On connection loss the reader retries and redis-py re-subscribes all
    channels automatically on reconnect; events published in between are
    lost (see module docstring).

    Args:
        url: Redis connection URL (ignored if *client* is provided).
        client: Inject an existing ``redis.asyncio.Redis`` instance.
        channel_prefix: Namespace prepended to every Redis channel.
        max_queue_size: Maximum events queued per subscription; oldest
            events are dropped when full.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        client: Any = None,
        channel_prefix: str = "roomkit:realtime",
        max_queue_size: int = 100,
    ) -> None:
        try:
            import redis.asyncio as _aioredis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisRealtimeBackend. "
                "Install it with: pip install roomkit[redis]"
            ) from exc

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = _aioredis.from_url(url)
            self._owns_client = True

        self._prefix = channel_prefix
        self._max_queue_size = max_queue_size
        self._pubsub: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._subscriptions: dict[str, _Subscription] = {}
        self._channels: dict[str, set[str]] = {}  # logical channel -> subscription_ids
        self._confirmations: dict[str, asyncio.Event] = {}  # logical channel -> confirmed
        self._closed = False

    # -- ABC implementation -----------------------------------------------

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        """Publish an event to a channel via Redis PUBLISH."""
        if self._closed:
            return
        payload = json.dumps(event.to_dict())
        await self._client.publish(self._redis_channel(channel), payload)

    async def subscribe(self, channel: str, callback: EphemeralCallback) -> str:
        """Subscribe to a channel.

        Waits (bounded) for the server's subscribe confirmation before
        returning, so events published after this call returns are
        guaranteed to reach the subscription — ``PubSub.subscribe()``
        alone only guarantees the command was written to the socket.

        Returns:
            A subscription ID that can be used to unsubscribe.
        """
        if self._closed:
            raise RuntimeError("RedisRealtimeBackend is closed")

        sub_id = uuid4().hex
        sub = _Subscription(
            sub_id=sub_id,
            channel=channel,
            callback=callback,
            max_queue_size=self._max_queue_size,
        )

        async with self._lock:
            if channel not in self._channels:
                if self._pubsub is None:
                    self._pubsub = self._client.pubsub()
                self._confirmations[channel] = asyncio.Event()
                try:
                    await self._pubsub.subscribe(self._redis_channel(channel))
                except BaseException:
                    self._confirmations.pop(channel, None)
                    raise
                # Start the reader only after the pubsub has a connection:
                # get_message() on an unconnected PubSub raises RuntimeError.
                if self._reader_task is None:
                    self._reader_task = asyncio.create_task(
                        self._reader(), name="redis-realtime-reader"
                    )
            self._subscriptions[sub_id] = sub
            self._channels.setdefault(channel, set()).add(sub_id)
            confirmed = self._confirmations.get(channel)

        sub.start()

        if confirmed is not None and not confirmed.is_set():
            try:
                await asyncio.wait_for(confirmed.wait(), timeout=_SUBSCRIBE_CONFIRM_TIMEOUT)
            except TimeoutError:
                logger.warning(
                    "No subscribe confirmation for channel %s after %.1fs; "
                    "events may be missed until the server processes it",
                    channel,
                    _SUBSCRIBE_CONFIRM_TIMEOUT,
                )
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe; drops the Redis channel when the last local
        subscriber leaves.

        Returns:
            True if the subscription existed and was removed.
        """
        async with self._lock:
            sub = self._subscriptions.pop(subscription_id, None)
            if sub is None:
                return False

            channel_subs = self._channels.get(sub.channel)
            if channel_subs is not None:
                channel_subs.discard(subscription_id)
                if not channel_subs:
                    del self._channels[sub.channel]
                    self._confirmations.pop(sub.channel, None)
                    if self._pubsub is not None:
                        try:
                            await self._pubsub.unsubscribe(self._redis_channel(sub.channel))
                        except Exception:
                            logger.warning(
                                "Redis unsubscribe failed for %s", sub.channel, exc_info=True
                            )

        await sub.stop()
        return True

    async def close(self) -> None:
        """Stop the reader, all subscriptions, and release connections."""
        if self._closed:
            return
        self._closed = True

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        for sub in list(self._subscriptions.values()):
            await sub.stop()
        self._subscriptions.clear()
        self._channels.clear()
        self._confirmations.clear()

        if self._pubsub is not None:
            with contextlib.suppress(Exception):
                await self._pubsub.aclose()
            self._pubsub = None

        if self._owns_client:
            await self._client.aclose()

    @property
    def subscription_count(self) -> int:
        """Return the number of active subscriptions."""
        return len(self._subscriptions)

    # -- Internal ---------------------------------------------------------

    def _redis_channel(self, channel: str) -> str:
        return f"{self._prefix}:{channel}"

    async def _reader(self) -> None:
        """Single reader loop dispatching pub/sub messages to subscriptions."""
        while not self._closed:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=False, timeout=1.0
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                if self._closed:
                    return
                logger.warning("Realtime pub/sub read failed; retrying", exc_info=True)
                await asyncio.sleep(_READ_RETRY_DELAY)
                continue

            if message is None:
                continue
            await self._dispatch(message)

    async def _dispatch(self, message: dict[str, Any]) -> None:
        # message["type"] is always str in redis-py; channel/data may be bytes.
        channel = _as_str(message["channel"]).removeprefix(f"{self._prefix}:")
        msg_type = message.get("type")

        if msg_type == "subscribe":
            confirmed = self._confirmations.get(channel)
            if confirmed is not None:
                confirmed.set()
            return
        if msg_type != "message":
            return

        try:
            event = EphemeralEvent.from_dict(json.loads(_as_str(message["data"])))
        except Exception:
            logger.warning("Dropping malformed realtime payload on %s", channel, exc_info=True)
            return

        # Copy: unsubscribe during dispatch must not mutate the live set.
        for sub_id in list(self._channels.get(channel, ())):
            sub = self._subscriptions.get(sub_id)
            if sub is not None:
                await sub.enqueue(event)
