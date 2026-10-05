"""Destination validation and outcome reporting for proactive delivery."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from roomkit.channels.base import hosts_realtime_model
from roomkit.core._voice_delivery import active_sessions as _active_sessions
from roomkit.core._voice_delivery import deliver_to_realtime_voice, replay_explicit_session
from roomkit.core.lanes import deferred_caller_waits
from roomkit.models.delivery import DeliveryError, DeliveryOutcome, InboundMessage, InboundResult
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    EventStatus,
    EventType,
    RoomStatus,
)
from roomkit.models.event import TextContent

if TYPE_CHECKING:
    from roomkit.core.delivery import DeliveryContext


def unavailable(reason: str, targets: list[str] | None = None) -> DeliveryOutcome:
    """A destination may become available on a later attempt."""
    return DeliveryOutcome(
        status="unavailable",
        reason=reason,
        unavailable_targets=targets or [],
        error=DeliveryError(code=reason, message=reason),
    )


def _rides_the_model(ctx: DeliveryContext, channel: object) -> bool:
    """Whether the delivery is injected into the channel's realtime model.

    Once sessions are pinned it stays so even if the model was unplugged
    since: the pinned sessions then refuse it, rather than the text being
    published to the room as the channel's own words.
    """
    if ctx.addressed_to is not None:
        return False
    return ctx._voice_sessions is not None or hosts_realtime_model(channel)


async def prepare_delivery(
    ctx: DeliveryContext,
    channel_id: str | None = None,
) -> tuple[str | None, DeliveryOutcome | None]:
    """Resolve and validate the channel; pin realtime sessions before waiting."""
    previous = await replay_explicit_session(ctx)
    if previous is not None:
        return None, previous
    channel_id = channel_id or await ctx.resolve_channel_id()
    if channel_id is None:
        return None, unavailable("no_transport")
    channel = ctx.kit.get_channel(channel_id)
    bindings = await ctx.kit.store.list_bindings(ctx.room_id)
    binding = next((b for b in bindings if b.channel_id == channel_id), None)
    if channel is None or binding is None:
        return None, unavailable("channel_unavailable", [channel_id])
    if channel.category == ChannelCategory.INTELLIGENCE:
        # The room's transport carries it; never an intelligence channel, so
        # the instruction cannot re-enter the channel it is for.
        transport_id = await ctx.find_transport_channel_id()
        if transport_id is None:
            return None, unavailable("no_transport")
        return await prepare_delivery(ctx, transport_id)
    if not _rides_the_model(ctx, channel):
        if ctx.session_id is not None:
            return None, DeliveryOutcome(status="blocked", reason="session_requires_realtime")
        return channel_id, None
    room = await ctx.kit.store.get_room(ctx.room_id)
    if room is None:
        return None, unavailable("room_unavailable")
    if room.status in (RoomStatus.CLOSED, RoomStatus.ARCHIVED):
        return None, DeliveryOutcome(status="blocked", reason="room_closed")
    if binding.access in (Access.WRITE_ONLY, Access.NONE):
        return None, DeliveryOutcome(status="blocked", reason="channel_cannot_read")
    if ctx._voice_channel is not None and ctx._voice_channel is not channel:
        return None, unavailable("voice_channel_replaced", [channel_id])
    if ctx._voice_sessions is None:
        sessions = _active_sessions(channel, ctx.room_id)
        if ctx.session_id is not None:
            sessions = [s for s in sessions if s.id == ctx.session_id]
        elif ctx.channel_id is not None and len(sessions) > 1:
            return None, unavailable("ambiguous_voice_session", [channel_id])
        if not sessions:
            return None, unavailable("voice_session_unavailable", [ctx.session_id or channel_id])
        ctx._voice_channel = channel
        ctx._voice_sessions = sessions
    return channel_id, None


async def deliver_to_channel(ctx: DeliveryContext, channel_id: str) -> DeliveryOutcome:
    """Use the existing inbound pipeline or the selected realtime provider."""
    resolved_id, refusal = await prepare_delivery(ctx, channel_id)
    if refusal is not None:
        return refusal
    assert resolved_id is not None
    channel_id = resolved_id
    channel = ctx.kit.get_channel(channel_id)
    if channel is None:
        return unavailable("channel_unavailable", [channel_id])
    if _rides_the_model(ctx, channel):
        return await deliver_to_realtime_voice(channel, ctx)
    message = InboundMessage(
        channel_id=channel_id,
        sender_id="system",
        event_type=EventType.INSTRUCTION if ctx.instruction else EventType.MESSAGE,
        content=TextContent(body=ctx.content),
        metadata=ctx.metadata or {},
        addressed_to=ctx.addressed_to,
        idempotency_key=ctx.idempotency_key,
        chain_depth=ctx.chain_depth,
    )
    # A delivery that waits for its turn hands the turn's failure to its
    # caller, who logs it (a hand-back logs it not delivered): once.
    with deferred_caller_waits(ctx._wait_for_turn):
        result = await ctx.kit.process_inbound(message, room_id=ctx.room_id, defer_delivery=True)
    if not isinstance(result, InboundResult):
        return DeliveryOutcome(status="unknown", reason="inbound_outcome_unknown")
    if result.delivery is not None and ctx._wait_for_turn:
        try:
            await result.delivery.wait()
        except asyncio.CancelledError:
            await result.delivery.cancel()
            raise
    return _text_outcome(result)


def _text_outcome(result: InboundResult) -> DeliveryOutcome:
    event = result.event
    outcome = DeliveryOutcome(
        status="sent",
        inbound=result,
        event_id=event.id if event is not None else None,
        duplicate=result.duplicate,
        turn_complete=bool(result.delivery and result.delivery.done and not result.duplicate),
    )
    if result.blocked or (event is not None and event.status == EventStatus.BLOCKED):
        return outcome.model_copy(
            update={"status": "blocked", "reason": result.reason or "inbound_blocked"}
        )
    if result.error is not None or result.cancellation_reason is not None:
        error = result.error
        return outcome.model_copy(
            update={
                "status": "failed",
                "reason": result.cancellation_reason or "inbound_failed",
                "error": DeliveryError(
                    code=type(error).__name__ if error else "turn_cancelled",
                    message=str(error) if error else str(result.cancellation_reason),
                    retryable=event is None,
                ),
            }
        )
    failed = next((r for r in result.delivery_results.values() if r.status == "failed"), None)
    if failed is not None:
        error = failed.error or DeliveryError(
            code="delivery_failed", message="Channel delivery failed"
        )
        return outcome.model_copy(
            update={
                "status": "failed",
                "reason": "channel_delivery_failed",
                "error": error.model_copy(update={"retryable": event is None}),
            }
        )
    if event is None:
        return outcome.model_copy(update={"status": "unknown", "reason": "no_publication_result"})
    # Replayed calls identify the original publication, not a new attempt at
    # the address supplied on this call. Its turn's completion is unknown.
    if result.duplicate:
        outcome.reason = "duplicate_publication"
    if result.unavailable_targets:
        return outcome.model_copy(
            update={
                "status": "unavailable",
                "reason": "addressed_targets_unavailable",
                "unavailable_targets": list(result.unavailable_targets),
                "error": DeliveryError(
                    code="addressed_targets_unavailable",
                    message="The event was published but an addressed target was unavailable",
                    retryable=False,
                ),
            }
        )
    return outcome
