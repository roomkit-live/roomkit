"""In-memory delivery backend.

Default backend using ``asyncio.Queue``.  Suitable for single-process
deployments — items are not persisted and are lost on crash.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING
from uuid import uuid4

from roomkit.delivery.base import DeliveryBackend, DeliveryItem, DeliveryItemStatus
from roomkit.delivery.worker import run_worker_loop

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.delivery.memory")


class InMemoryDeliveryBackend(DeliveryBackend):
    """Asyncio-queue-based delivery backend (single process, no persistence).

    Items flow through ``_queue`` → ``_in_flight`` → acked/dead-lettered.
    A background worker task drains the queue and executes deliveries.
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        max_dead_letter_size: int = 1000,
    ) -> None:
        self._queue: asyncio.Queue[DeliveryItem] = asyncio.Queue(
            maxsize=max_queue_size,
        )
        self._in_flight: dict[str, DeliveryItem] = {}
        self._dead_letter: deque[DeliveryItem] = deque(
            maxlen=max_dead_letter_size,
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._kit: RoomKit | None = None
        self._worker_id = uuid4().hex[:12]

    # -- ABC implementation -----------------------------------------------

    async def enqueue(self, item: DeliveryItem) -> None:
        item.status = DeliveryItemStatus.PENDING
        await self._queue.put(item)
        logger.debug(
            "Enqueued %s for room %s (depth=%d)",
            item.id,
            item.room_id,
            self._queue.qsize(),
        )

    async def dequeue(
        self,
        worker_id: str,
        batch_size: int = 1,
        timeout: float = 5.0,
    ) -> list[DeliveryItem]:
        items: list[DeliveryItem] = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            first.status = DeliveryItemStatus.IN_PROGRESS
            first.worker_id = worker_id
            self._in_flight[first.id] = first
            items.append(first)
        except TimeoutError:
            return items

        # Drain up to batch_size without blocking
        while len(items) < batch_size:
            try:
                item = self._queue.get_nowait()
                item.status = DeliveryItemStatus.IN_PROGRESS
                item.worker_id = worker_id
                self._in_flight[item.id] = item
                items.append(item)
            except asyncio.QueueEmpty:
                break

        return items

    async def ack(self, item_id: str) -> None:
        item = self._in_flight.pop(item_id, None)
        if item is not None:
            item.status = DeliveryItemStatus.DELIVERED
            logger.debug("Acked %s", item_id)

    async def nack(self, item_id: str, error: str | None = None) -> None:
        item = self._in_flight.pop(item_id, None)
        if item is None:
            return

        item.retry_count += 1
        item.error = error

        if item.retry_count >= item.max_retries:
            item.status = DeliveryItemStatus.DEAD_LETTER
            item.error = error or "max retries exceeded"
            self._dead_letter.append(item)
            logger.warning(
                "Dead-lettered %s after %d retries: %s",
                item_id,
                item.retry_count,
                error,
            )
        else:
            item.status = DeliveryItemStatus.PENDING
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                item.status = DeliveryItemStatus.DEAD_LETTER
                item.error = "queue full on retry"
                self._dead_letter.append(item)
                logger.warning("Dead-lettered %s — queue full on retry", item_id)
            else:
                logger.debug(
                    "Re-enqueued %s (attempt %d/%d)",
                    item_id,
                    item.retry_count,
                    item.max_retries,
                )

    async def dead_letter(self, item_id: str, error: str) -> None:
        item = self._in_flight.pop(item_id, None)
        if item is None:
            return
        item.status = DeliveryItemStatus.DEAD_LETTER
        item.error = error
        self._dead_letter.append(item)

    async def get_queue_depth(self) -> int:
        return self._queue.qsize()

    async def get_dead_letter_items(self, limit: int = 50) -> list[DeliveryItem]:
        return list(self._dead_letter)[:limit]

    # -- Lifecycle --------------------------------------------------------

    async def start(self, kit: RoomKit) -> None:
        """Start the background worker loop."""
        if self._worker_task is not None:
            return
        self._kit = kit
        self._worker_task = asyncio.create_task(
            run_worker_loop(
                self,
                kit,
                self._worker_id,
                batch_size=5,
                poll_timeout=2.0,
            ),
            name="delivery-worker",
        )
        logger.info("In-memory delivery worker started (id=%s)", self._worker_id)

    async def close(self) -> None:
        """Stop the worker loop.  In-flight items are re-enqueued."""
        await self._cancel_worker_task()
        # Re-enqueue any in-flight items so they aren't silently lost
        for item in self._in_flight.values():
            item.status = DeliveryItemStatus.PENDING
            item.worker_id = None
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                item.status = DeliveryItemStatus.DEAD_LETTER
                item.error = "queue full at shutdown"
                self._dead_letter.append(item)
                logger.warning("Item %s dead-lettered at shutdown — queue full", item.id)
        self._in_flight.clear()
        logger.info("In-memory delivery worker stopped")
