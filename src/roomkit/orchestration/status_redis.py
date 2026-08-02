"""Redis status bus backend.

Distributes :class:`~roomkit.orchestration.status_bus.StatusEntry`
updates across processes: entries are stored in a capped Redis list
(shared history for ``recent()``) and fanned out to subscribers via
Redis pub/sub.

Requires ``redis>=5.0.1``::

    pip install roomkit[redis]

Usage::

    from roomkit.orchestration.status_bus import StatusBus
    from roomkit.orchestration.status_redis import RedisStatusBackend

    bus = StatusBus(backend=RedisStatusBackend("redis://localhost:6379"))

Semantics
---------

Unlike ``InMemoryStatusBackend``, ``publish()`` returning does **not**
mean subscribers ran: local subscribers are notified through the Redis
round-trip, asynchronously. In a multi-process deployment every
process's subscribers observe every entry from every process — with
RoomKit this means each worker re-emits all entries as ``status_posted``
framework events, which is the point (distributed observability).
Pub/sub is fire-and-forget: notifications published while a process is
disconnected are lost, but the shared history remains available through
``recent()``. If loss-free consumption is ever required, Redis Streams
(the ``RedisDeliveryBackend`` approach) is the upgrade path.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from typing import Any

from roomkit.orchestration.status_bus import StatusBackend, StatusCallback, StatusEntry

logger = logging.getLogger("roomkit.orchestration.status_redis")

_SUBSCRIBE_CONFIRM_TIMEOUT = 5.0
_READ_RETRY_DELAY = 1.0


class RedisStatusBackend(StatusBackend):
    """Status bus backend backed by a capped Redis list + pub/sub.

    Args:
        url: Redis connection URL (ignored if *client* is provided).
        client: Inject an existing ``redis.asyncio.Redis`` instance.
        key_prefix: Namespace for the list key and pub/sub channel.
        max_entries: Maximum entries retained in the shared history.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        client: Any = None,
        key_prefix: str = "roomkit:status",
        max_entries: int = 500,
    ) -> None:
        try:
            import redis.asyncio as _aioredis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisStatusBackend. "
                "Install it with: pip install roomkit[redis]"
            ) from exc

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = _aioredis.from_url(url)
            self._owns_client = True

        self._entries_key = f"{key_prefix}:entries"
        self._events_channel = f"{key_prefix}:events"
        self._max_entries = max_entries
        self._subscribers: list[StatusCallback] = []
        self._pubsub: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._confirmed = asyncio.Event()
        self._closed = False

    # -- ABC implementation -----------------------------------------------

    async def publish(self, entry: StatusEntry) -> None:
        """Store the entry in the shared history and notify subscribers."""
        if self._closed:
            return
        payload = entry.model_dump_json()
        pipe = self._client.pipeline(transaction=False)
        pipe.lpush(self._entries_key, payload)
        pipe.ltrim(self._entries_key, 0, self._max_entries - 1)
        pipe.publish(self._events_channel, payload)
        await pipe.execute()

    async def recent(
        self, n: int, *, agent_id: str | None = None, status: str | None = None
    ) -> list[StatusEntry]:
        """Retrieve recent entries from the shared history."""
        raw_entries = await self._client.lrange(self._entries_key, 0, -1)
        entries: list[StatusEntry] = []
        # LPUSH stores newest first; reverse to chronological order.
        for raw in reversed(raw_entries):
            data = raw if isinstance(raw, str) else raw.decode()
            try:
                entries.append(StatusEntry.model_validate_json(data))
            except Exception:
                logger.warning("Skipping malformed status entry", exc_info=True)
        if agent_id is not None:
            entries = [e for e in entries if e.agent_id == agent_id]
        if status is not None:
            entries = [e for e in entries if e.status == status]
        return entries[-n:]

    async def subscribe(self, callback: StatusCallback) -> None:
        """Subscribe to new entries (from this and every other process).

        The first subscriber starts the pub/sub reader; the call waits
        (bounded) for the server's subscribe confirmation so entries
        published after it returns are guaranteed to be observed.
        """
        if self._closed:
            raise RuntimeError("RedisStatusBackend is closed")

        async with self._lock:
            if self._reader_task is None:
                if self._pubsub is None:
                    self._pubsub = self._client.pubsub()
                await self._pubsub.subscribe(self._events_channel)
                # Start the reader only after the pubsub has a connection:
                # get_message() on an unconnected PubSub raises RuntimeError.
                self._reader_task = asyncio.create_task(self._reader(), name="redis-status-reader")
            self._subscribers.append(callback)

        if not self._confirmed.is_set():
            try:
                await asyncio.wait_for(self._confirmed.wait(), timeout=_SUBSCRIBE_CONFIRM_TIMEOUT)
            except TimeoutError:
                logger.warning(
                    "No subscribe confirmation for %s after %.1fs; "
                    "entries may be missed until the server processes it",
                    self._events_channel,
                    _SUBSCRIBE_CONFIRM_TIMEOUT,
                )

    async def unsubscribe(self, callback: StatusCallback) -> None:
        """Remove a subscriber.

        The reader stays alive until :meth:`close` — the framework never
        unsubscribes in practice, and stopping/restarting it would
        reintroduce lazy-init races for one idle connection of savings.
        """
        async with self._lock:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(callback)

    async def close(self) -> None:
        """Stop the reader and release connections."""
        if self._closed:
            return
        self._closed = True

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None

        self._subscribers.clear()

        if self._pubsub is not None:
            with contextlib.suppress(Exception):
                await self._pubsub.aclose()
            self._pubsub = None

        if self._owns_client:
            await self._client.aclose()

    # -- Internal ---------------------------------------------------------

    async def _reader(self) -> None:
        """Reader loop dispatching pub/sub notifications to local callbacks."""
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
                logger.warning("Status pub/sub read failed; retrying", exc_info=True)
                await asyncio.sleep(_READ_RETRY_DELAY)
                continue

            if message is None:
                continue

            msg_type = message.get("type")
            if msg_type == "subscribe":
                self._confirmed.set()
                continue
            if msg_type != "message":
                continue

            raw = message["data"]
            data = raw if isinstance(raw, str) else raw.decode()
            try:
                entry = StatusEntry.model_validate_json(data)
            except Exception:
                logger.warning("Dropping malformed status payload", exc_info=True)
                continue

            # Copy: unsubscribe during dispatch must not mutate the live list.
            for cb in list(self._subscribers):
                try:
                    result = cb(entry)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.exception("StatusBus subscriber failed")
