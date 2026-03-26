"""Delivery worker — shared execution logic.

Used by both ``InMemoryDeliveryBackend`` and ``RedisDeliveryBackend``
to dequeue items and execute the actual delivery.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.core.delivery import DeliveryContext
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.serialization import deserialize_strategy

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.delivery.base import DeliveryBackend

logger = logging.getLogger("roomkit.delivery.worker")


async def execute_delivery(kit: RoomKit, item: DeliveryItem) -> None:
    """Execute a single delivery item against *kit*.

    Deserializes the strategy from the item, builds a
    ``DeliveryContext``, and calls ``strategy.deliver(ctx)``.
    """
    strategy = deserialize_strategy(item.strategy)
    ctx = DeliveryContext(
        kit=kit,
        room_id=item.room_id,
        content=item.content,
        channel_id=item.channel_id,
        metadata=item.metadata,
    )
    await strategy.deliver(ctx)


async def run_worker_loop(
    backend: DeliveryBackend,
    kit: RoomKit,
    worker_id: str,
    *,
    batch_size: int = 1,
    poll_timeout: float = 5.0,
) -> None:
    """Continuously dequeue and execute deliveries until cancelled.

    On success the item is acked.  On failure the item is nacked
    (which either re-enqueues or dead-letters based on retry count).
    """
    while True:
        try:
            items = await backend.dequeue(worker_id, batch_size=batch_size, timeout=poll_timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Dequeue failed, retrying after 1s")
            await asyncio.sleep(1)
            continue

        for item in items:
            try:
                await execute_delivery(kit, item)
                await backend.ack(item.id)
                logger.debug("Delivered %s to room %s", item.id, item.room_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Delivery %s failed (attempt %d/%d): %s",
                    item.id,
                    item.retry_count + 1,
                    item.max_retries,
                    exc,
                )
                await backend.nack(item.id, error=str(exc))
