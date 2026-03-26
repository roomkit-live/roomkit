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
from roomkit.models.enums import EventStatus, EventType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.delivery.base import DeliveryBackend

logger = logging.getLogger("roomkit.delivery.worker")


async def _fire_delivery_hooks(
    kit: RoomKit,
    item: DeliveryItem,
    trigger: HookTrigger,
    *,
    error: str | None = None,
) -> None:
    """Fire BEFORE_DELIVER or AFTER_DELIVER hooks for a delivery item."""
    meta = {
        "channel_id": item.channel_id,
        "strategy": item.strategy.get("type", "immediate"),
        "delivery_item_id": item.id,
        **item.metadata,
    }
    if error is not None:
        meta["error"] = error

    status = EventStatus.PENDING
    if trigger == HookTrigger.AFTER_DELIVER:
        status = EventStatus.FAILED if error else EventStatus.DELIVERED

    hook_event = RoomEvent(
        room_id=item.room_id,
        source=EventSource(channel_id="system", channel_type="system"),  # type: ignore[arg-type]
        content=TextContent(body=item.content),
        type=EventType.MESSAGE,
        status=status,
        visibility="internal",
        metadata=meta,
    )

    try:
        room_context = await kit._build_context(item.room_id)  # noqa: SLF001
        await kit.hook_engine.run_async_hooks(item.room_id, trigger, hook_event, room_context)
    except Exception:
        logger.warning("%s hook failed for item %s", trigger.value, item.id, exc_info=True)


async def execute_delivery(kit: RoomKit, item: DeliveryItem) -> None:
    """Execute a single delivery item against *kit*.

    Deserializes the strategy, fires ``BEFORE_DELIVER`` / ``AFTER_DELIVER``
    hooks, and calls ``strategy.deliver(ctx)``.
    """
    strategy = deserialize_strategy(item.strategy)
    ctx = DeliveryContext(
        kit=kit,
        room_id=item.room_id,
        content=item.content,
        channel_id=item.channel_id,
        metadata=item.metadata,
    )

    await _fire_delivery_hooks(kit, item, HookTrigger.BEFORE_DELIVER)

    error: str | None = None
    try:
        await strategy.deliver(ctx)
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        await _fire_delivery_hooks(kit, item, HookTrigger.AFTER_DELIVER, error=error)


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
