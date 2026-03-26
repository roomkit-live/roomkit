"""Redis Streams delivery backend.

Uses Redis Streams with consumer groups for persistent, distributed
delivery across multiple worker processes.

Requires ``redis>=5.0``::

    pip install roomkit[redis]

Usage::

    from roomkit.delivery import RedisDeliveryBackend

    kit = RoomKit(
        delivery_backend=RedisDeliveryBackend("redis://localhost:6379"),
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.delivery.base import DeliveryBackend, DeliveryItem, DeliveryItemStatus
from roomkit.delivery.worker import run_worker_loop

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.delivery.redis")


async def _xdel_safe(client: Any, key: str, entry_id: str) -> None:
    """Delete a stream entry.  Non-fatal — the entry is already acked."""
    try:
        await client.xdel(key, entry_id)
    except Exception:
        logger.warning("XDEL %s %s failed (non-fatal)", key, entry_id, exc_info=True)


class RedisDeliveryBackend(DeliveryBackend):
    """Redis Streams delivery backend with consumer groups.

    Items are serialized as JSON and stored in a Redis Stream.
    Consumer groups distribute items across workers automatically.

    Args:
        url: Redis connection URL (ignored if *client* is provided).
        client: Inject an existing ``redis.asyncio.Redis`` instance.
        stream_prefix: Namespace for stream keys.
        group_name: Consumer group name — all workers in the same
            group share the workload.
        max_dead_letter_size: Approximate cap on the dead-letter stream
            (uses ``MAXLEN ~``).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        client: Any = None,
        stream_prefix: str = "roomkit:delivery",
        group_name: str = "roomkit-workers",
        max_dead_letter_size: int = 10_000,
    ) -> None:
        try:
            import redis.asyncio as _aioredis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisDeliveryBackend. "
                "Install it with: pip install roomkit[redis]"
            ) from exc

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = _aioredis.from_url(url)
            self._owns_client = True

        self._pending_key = f"{stream_prefix}:pending"
        self._dl_key = f"{stream_prefix}:dead_letter"
        self._group = group_name
        self._max_dl = max_dead_letter_size
        self._worker_id = uuid4().hex[:12]
        self._worker_task: asyncio.Task[None] | None = None

        # Maps DeliveryItem.id → Redis stream entry ID
        self._entry_ids: dict[str, str] = {}
        # Maps DeliveryItem.id → DeliveryItem (for nack/dead_letter)
        self._items: dict[str, DeliveryItem] = {}

    # -- ABC implementation -----------------------------------------------

    async def enqueue(self, item: DeliveryItem) -> None:
        item.status = DeliveryItemStatus.PENDING
        await self._client.xadd(self._pending_key, {"data": item.model_dump_json()})
        logger.debug("Enqueued %s for room %s", item.id, item.room_id)

    async def dequeue(
        self,
        worker_id: str,
        batch_size: int = 1,
        timeout: float = 5.0,
    ) -> list[DeliveryItem]:
        block_ms = int(timeout * 1000)
        resp = await self._client.xreadgroup(
            self._group,
            worker_id,
            {self._pending_key: ">"},
            count=batch_size,
            block=block_ms,
        )

        if not resp:
            return []

        items: list[DeliveryItem] = []
        # resp format: [[stream_name, [(entry_id, {field: value}), ...]]]
        for _stream_name, entries in resp:
            for entry_id, fields in entries:
                eid = entry_id if isinstance(entry_id, str) else entry_id.decode()
                raw = fields.get(b"data") or fields.get("data")
                if raw is None:
                    logger.warning("Entry %s has no data field, skipping", eid)
                    await self._client.xack(self._pending_key, self._group, eid)
                    continue

                data = raw if isinstance(raw, str) else raw.decode()
                item = DeliveryItem.model_validate_json(data)
                item.status = DeliveryItemStatus.IN_PROGRESS
                item.worker_id = worker_id
                self._entry_ids[item.id] = eid
                self._items[item.id] = item
                items.append(item)

        return items

    async def ack(self, item_id: str) -> None:
        entry_id = self._entry_ids.pop(item_id, None)
        self._items.pop(item_id, None)
        if entry_id is None:
            return
        await self._client.xack(self._pending_key, self._group, entry_id)
        await _xdel_safe(self._client, self._pending_key, entry_id)
        logger.debug("Acked %s (entry %s)", item_id, entry_id)

    async def nack(self, item_id: str, error: str | None = None) -> None:
        entry_id = self._entry_ids.pop(item_id, None)
        item = self._items.pop(item_id, None)
        if item is None or entry_id is None:
            return

        item.retry_count += 1
        item.error = error

        if item.retry_count >= item.max_retries:
            # Dead-letter: add to DL stream, ack + delete from pending
            item.status = DeliveryItemStatus.DEAD_LETTER
            item.error = error or "max retries exceeded"
            await self._client.xadd(
                self._dl_key,
                {"data": item.model_dump_json()},
                maxlen=self._max_dl,
                approximate=True,
            )
            await self._client.xack(self._pending_key, self._group, entry_id)
            await _xdel_safe(self._client, self._pending_key, entry_id)
            logger.warning(
                "Dead-lettered %s after %d retries: %s",
                item_id,
                item.retry_count,
                error,
            )
        else:
            # Re-enqueue: add new entry, ack + delete old one
            item.status = DeliveryItemStatus.PENDING
            await self._client.xadd(self._pending_key, {"data": item.model_dump_json()})
            await self._client.xack(self._pending_key, self._group, entry_id)
            await _xdel_safe(self._client, self._pending_key, entry_id)
            logger.debug(
                "Re-enqueued %s (attempt %d/%d)",
                item_id,
                item.retry_count,
                item.max_retries,
            )

    async def dead_letter(self, item_id: str, error: str) -> None:
        entry_id = self._entry_ids.pop(item_id, None)
        item = self._items.pop(item_id, None)
        if item is None:
            return

        item.status = DeliveryItemStatus.DEAD_LETTER
        item.error = error
        await self._client.xadd(
            self._dl_key,
            {"data": item.model_dump_json()},
            maxlen=self._max_dl,
            approximate=True,
        )
        if entry_id is not None:
            await self._client.xack(self._pending_key, self._group, entry_id)
            await _xdel_safe(self._client, self._pending_key, entry_id)

    async def get_queue_depth(self) -> int:
        result = await self._client.xlen(self._pending_key)
        return int(result)

    async def get_dead_letter_items(self, limit: int = 50) -> list[DeliveryItem]:
        entries = await self._client.xrevrange(self._dl_key, "+", "-", count=limit)
        items: list[DeliveryItem] = []
        for _entry_id, fields in entries:
            raw = fields.get(b"data") or fields.get("data")
            if raw is None:
                continue
            data = raw if isinstance(raw, str) else raw.decode()
            items.append(DeliveryItem.model_validate_json(data))
        return items

    # -- Lifecycle --------------------------------------------------------

    async def start(self, kit: RoomKit) -> None:
        """Create consumer group and start the worker loop."""
        if self._worker_task is not None:
            return
        try:
            # id="0" reads from the beginning — required so items enqueued
            # before start() are not silently dropped.
            await self._client.xgroup_create(self._pending_key, self._group, id="0", mkstream=True)
            logger.info("Created consumer group %s", self._group)
        except Exception:
            # Group already exists — that's fine
            logger.debug("Consumer group %s already exists", self._group)

        self._worker_task = asyncio.create_task(
            run_worker_loop(
                self,
                kit,
                self._worker_id,
                batch_size=5,
                poll_timeout=2.0,
            ),
            name="redis-delivery-worker",
        )
        logger.info(
            "Redis delivery worker started (id=%s, group=%s)",
            self._worker_id,
            self._group,
        )

    async def close(self) -> None:
        """Stop the worker loop and close the connection if we own it.

        In-flight items remain in the Redis PEL (Pending Entries List)
        and will be reclaimed by another consumer or on restart.
        """
        await self._cancel_worker_task()

        # Clear in-process tracking (items remain in Redis PEL for recovery)
        self._entry_ids.clear()
        self._items.clear()

        if self._owns_client:
            await self._client.aclose()

        logger.info("Redis delivery worker stopped")
