"""Redis Streams delivery backend.

Uses Redis Streams with consumer groups for persistent, distributed
delivery across multiple worker processes.

Requires Redis server 6.2+ and ``redis>=5.0``::

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
import math
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.delivery._redis_scripts import RENEW, TRANSITION
from roomkit.delivery.base import DeliveryBackend, DeliveryItem, DeliveryItemStatus
from roomkit.delivery.worker import run_worker_loop

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.delivery.redis")


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
        claim_idle_seconds: Reclaim abandoned work after this idle interval.
            Live workers renew their pending entries every third of this interval.
            Delivery is at-least-once; destinations should handle duplicate IDs.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        client: Any = None,
        stream_prefix: str = "roomkit:delivery",
        group_name: str = "roomkit-workers",
        max_dead_letter_size: int = 10_000,
        claim_idle_seconds: float = 60.0,
    ) -> None:
        if not math.isfinite(claim_idle_seconds) or claim_idle_seconds <= 0:
            raise ValueError("claim_idle_seconds must be finite and positive")
        try:
            import redis.asyncio as _aioredis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisDeliveryBackend. "
                "Install it with: pip install roomkit[redis]"
            ) from exc

        self._client: Any
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
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._claim_idle_ms = max(1, math.ceil(claim_idle_seconds * 1000))
        self._claim_cursor: str | bytes = "0-0"

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
        # Scan the PEL before reading new work, advancing the cursor even
        # when this scan finds no sufficiently idle entry.
        claimed = await self._client.xautoclaim(
            self._pending_key,
            self._group,
            worker_id,
            self._claim_idle_ms,
            start_id=self._claim_cursor,
            count=batch_size,
        )
        self._claim_cursor = claimed[0]
        if claimed[1]:
            resp = [(self._pending_key, claimed[1])]
        else:
            resp = await self._client.xreadgroup(
                self._group,
                worker_id,
                {self._pending_key: ">"},
                count=batch_size,
                block=max(1, int(timeout * 1000)) if timeout > 0 else None,
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

        if items and (self._heartbeat_task is None or self._heartbeat_task.done()):
            self._heartbeat_task = asyncio.create_task(
                self._renew_pending(), name="redis-delivery-heartbeat"
            )
        return items

    async def _renew_pending(self) -> None:
        while True:
            await asyncio.sleep(self._claim_idle_ms / 3000)
            for item_id, entry_id in list(self._entry_ids.items()):
                item = self._items.get(item_id)
                if item is None:
                    continue
                try:
                    await self._client.eval(
                        RENEW, 1, self._pending_key, self._group, entry_id, item.worker_id
                    )
                except Exception:
                    logger.warning("Failed to renew delivery %s", item_id, exc_info=True)

    async def _transition(self, item_id: str, replacement: DeliveryItem | None = None) -> None:
        entry_id = self._entry_ids.get(item_id)
        item = self._items.get(item_id)
        if entry_id is None or item is None:
            return
        dead = replacement is not None and replacement.status == DeliveryItemStatus.DEAD_LETTER
        await self._client.eval(
            TRANSITION,
            2,
            self._pending_key,
            self._dl_key if dead else self._pending_key,
            self._group,
            entry_id,
            item.worker_id,
            replacement.model_dump_json() if replacement is not None else "",
            self._max_dl if dead else 0,
        )
        # A network failure leaves these intact so the transition can be retried.
        # A zero result means it was already completed or another worker owns it.
        if self._entry_ids.get(item_id) == entry_id:
            self._entry_ids.pop(item_id, None)
            self._items.pop(item_id, None)

    async def ack(self, item_id: str) -> None:
        await self._transition(item_id)

    async def nack(self, item_id: str, error: str | None = None) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        retry_count = item.retry_count + 1
        dead = retry_count >= item.max_retries
        replacement = item.model_copy(
            update={
                "retry_count": retry_count,
                "error": (error or "max retries exceeded") if dead else error,
                "status": DeliveryItemStatus.DEAD_LETTER if dead else DeliveryItemStatus.PENDING,
                "worker_id": None,
            }
        )
        await self._transition(item_id, replacement)

    async def dead_letter(self, item_id: str, error: str) -> None:
        item = self._items.get(item_id)
        if item is not None:
            await self._transition(
                item_id,
                item.model_copy(
                    update={
                        "status": DeliveryItemStatus.DEAD_LETTER,
                        "error": error,
                        "worker_id": None,
                    }
                ),
            )

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
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
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
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        # Clear in-process tracking (items remain in Redis PEL for recovery)
        self._entry_ids.clear()
        self._items.clear()

        if self._owns_client:
            await self._client.aclose()

        logger.info("Redis delivery worker stopped")
