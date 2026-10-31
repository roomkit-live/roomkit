"""Shared proactive delivery execution and queue worker."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.core._publication_keys import application_key
from roomkit.core.delivery import DeliveryContext, DeliveryStrategy
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.serialization import deserialize_strategy
from roomkit.models.delivery import DeliveryError, DeliveryOutcome
from roomkit.models.enums import EventStatus, EventType, HookTrigger, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.core.hooks import HookEngine
    from roomkit.delivery.base import DeliveryBackend

logger = logging.getLogger("roomkit.delivery.worker")
_OUTCOME_STATUS = {
    "queued": EventStatus.PENDING,
    "sent": EventStatus.DELIVERED,
    "blocked": EventStatus.BLOCKED,
    "unavailable": EventStatus.FAILED,
    "failed": EventStatus.FAILED,
    "unknown": EventStatus.PENDING,
}


def build_delivery_hook_event(
    room_id: str,
    content: str,
    *,
    channel_id: str | None = None,
    strategy_name: str = "immediate",
    status: EventStatus = EventStatus.PENDING,
    extra_meta: dict[str, object] | None = None,
    addressed_to: list[str] | None = None,
    idempotency_key: str | None = None,
    instruction: bool = False,
) -> RoomEvent:
    """Build the observation shared by direct execution and queue workers."""
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="system", channel_type="system"),
        content=TextContent(body=content),
        type=EventType.INSTRUCTION if instruction else EventType.MESSAGE,
        status=status,
        visibility=Visibility.INTERNAL,
        addressed_to=addressed_to,
        idempotency_key=idempotency_key,
        metadata={**(extra_meta or {}), "channel_id": channel_id, "strategy": strategy_name},
    )


async def fire_delivery_hooks(
    kit: RoomKit,
    item: DeliveryItem,
    trigger: HookTrigger,
    *,
    error: str | None = None,
    hook_engine: HookEngine | None = None,
) -> str | None:
    """Return effective content, or None for an explicit BEFORE_DELIVER refusal."""
    hooks = hook_engine if hook_engine is not None else kit.hook_engine
    if not hooks.has_hooks(trigger):
        return item.content
    outcome = item.outcome
    published = (
        outcome.inbound.event
        if trigger == HookTrigger.AFTER_DELIVER and outcome and outcome.inbound
        else None
    )
    extra: dict[str, object] = {
        **item.metadata,
        "delivery_item_id": item.id,
        "session_id": item.session_id,
    }
    status = EventStatus.PENDING
    if trigger == HookTrigger.AFTER_DELIVER:
        if outcome is not None:
            status = _OUTCOME_STATUS[outcome.status]
            extra["delivery_outcome"] = outcome.model_dump(mode="json")
            error = None
            if outcome.status in ("blocked", "unavailable", "failed", "unknown"):
                error = outcome.error.message if outcome.error else outcome.reason
        else:
            status = EventStatus.FAILED if error else EventStatus.DELIVERED
        extra["error"] = error
    event = build_delivery_hook_event(
        item.room_id,
        published.content.body
        if published is not None and isinstance(published.content, TextContent)
        else item.content,
        channel_id=item.channel_id,
        strategy_name=item.strategy.get("type", "immediate"),
        status=status,
        extra_meta=extra,
        addressed_to=published.addressed_to if published is not None else item.addressed_to,
        idempotency_key=application_key(published.idempotency_key)
        if published is not None
        else item.idempotency_key,
        instruction=item.instruction,
    )
    if trigger == HookTrigger.AFTER_DELIVER:
        try:
            context = await kit._build_context(item.room_id)
            hooks._start_async_hooks(item.room_id, trigger, event, context)
        except Exception:
            logger.warning("AFTER_DELIVER failed for item %s", item.id, exc_info=True)
        return item.content
    # This trigger is fail-open (§9.3). Only an explicit refusal blocks.
    try:
        context = await kit._build_context(item.room_id)
        result = await hooks.run_sync_hooks(item.room_id, trigger, event, context)
    except Exception:
        logger.warning("BEFORE_DELIVER could not run for item %s", item.id, exc_info=True)
        return item.content
    if not result.allowed:
        item.outcome = DeliveryOutcome(
            status="blocked",
            reason=result.reason or result.blocked_by or "before_deliver_blocked",
            delivery_item_id=item.id,
        )
        return None
    if isinstance(result.event, RoomEvent) and isinstance(result.event.content, TextContent):
        return result.event.content.body
    return item.content


async def reject_invalid_delivery(
    kit: RoomKit,
    item: DeliveryItem,
    *,
    hook_engine: HookEngine | None = None,
) -> DeliveryOutcome | None:
    """Report invalid requests consistently before enqueue or execution."""
    reason = None
    if item.session_id is not None and (item.channel_id is None or item.addressed_to is not None):
        reason = "invalid_session_target"
    elif item.idempotency_key is not None and not item.idempotency_key.strip():
        # A blank key names no publication (RFC §13.4): a request asking for
        # idempotency with one is refused, not delivered unkeyed.
        reason = "empty_idempotency_key"
    if reason is None:
        return None
    item.outcome = DeliveryOutcome(status="blocked", reason=reason, delivery_item_id=item.id)
    await fire_delivery_hooks(kit, item, HookTrigger.AFTER_DELIVER, hook_engine=hook_engine)
    return item.outcome


async def execute_delivery(
    kit: RoomKit,
    item: DeliveryItem,
    *,
    strategy: DeliveryStrategy | None = None,
    hook_engine: HookEngine | None = None,
) -> DeliveryOutcome:
    """Execute once and report its actual outcome, including hook refusals."""
    item.outcome = None
    refusal = await reject_invalid_delivery(kit, item, hook_engine=hook_engine)
    if refusal is not None:
        return refusal
    content = await fire_delivery_hooks(
        kit, item, HookTrigger.BEFORE_DELIVER, hook_engine=hook_engine
    )
    effective = item.model_copy(update={"content": content}) if content is not None else item
    if content is not None:
        try:
            resolved = strategy if strategy is not None else deserialize_strategy(item.strategy)
            outcome = await resolved.deliver(
                DeliveryContext(
                    kit=kit,
                    room_id=item.room_id,
                    content=content,
                    channel_id=item.channel_id,
                    metadata=item.metadata,
                    addressed_to=item.addressed_to,
                    idempotency_key=item.idempotency_key,
                    session_id=item.session_id,
                    chain_depth=item.chain_depth,
                    instruction=item.instruction,
                )
            )
            if not isinstance(outcome, DeliveryOutcome):
                outcome = DeliveryOutcome(status="unknown", reason="strategy_outcome_unknown")
        except Exception as exc:
            outcome = DeliveryOutcome(
                status="failed",
                reason="strategy_failed",
                error=DeliveryError(code=type(exc).__name__, message=str(exc)),
            )
        item.outcome = outcome.model_copy(update={"delivery_item_id": item.id})
    assert item.outcome is not None
    effective.outcome = item.outcome
    await fire_delivery_hooks(kit, effective, HookTrigger.AFTER_DELIVER, hook_engine=hook_engine)
    return item.outcome


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
                outcome = await execute_delivery(kit, item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                outcome = DeliveryOutcome(
                    status="failed",
                    reason="execution_failed",
                    delivery_item_id=item.id,
                    error=DeliveryError(code=type(exc).__name__, message=str(exc)),
                )
                item.outcome = outcome
                logger.exception("Execution failed for item %s", item.id)
            await _settle_item(backend, item, outcome)
            logger.debug("Delivery %s: %s (%s)", item.id, outcome.status, outcome.reason)


async def _settle_item(
    backend: DeliveryBackend,
    item: DeliveryItem,
    outcome: DeliveryOutcome,
) -> None:
    """Retry the same queue transition without repeating delivery execution."""
    while True:
        try:
            failure = outcome.status in ("unavailable", "failed") or (
                outcome.status == "unknown" and outcome.error is not None
            )
            if not failure:
                await backend.ack(item.id)
            elif outcome.error is not None and not outcome.error.retryable:
                await backend.dead_letter(item.id, error=outcome.error.message)
            else:
                reason = (
                    outcome.error.message if outcome.error else outcome.reason or outcome.status
                )
                await backend.nack(item.id, error=reason)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Queue transition failed for %s; retrying after 1s", item.id)
            await asyncio.sleep(1)
