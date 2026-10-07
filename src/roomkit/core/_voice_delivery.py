"""Reserve proactive voice injections and report their submission boundary."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.delivery import DeliveryError, DeliveryOutcome
from roomkit.models.enums import Access, RoomStatus
from roomkit.models.voice_delivery import VoiceDeliveryRecord
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.realtime.injection import VoiceInjectionResult

if TYPE_CHECKING:
    from roomkit.core.delivery import DeliveryContext

logger = logging.getLogger("roomkit.delivery.voice")
_SEVERITY = {"sent": 0, "blocked": 1, "unavailable": 2, "failed": 3, "unknown": 4}


def active_sessions(channel: Any, room_id: str) -> list[Any]:
    return [s for s in channel.get_room_sessions(room_id) if s.state != VoiceSessionState.ENDED]


def injection_silent(binding: ChannelBinding) -> bool:
    """Whether text injected into a realtime host's model under *binding* asks
    no answer: a muted, output-muted or unwritable binding withholds the
    model's response, its context still taking the text (RFC §7.2, §7.5)."""
    return binding.muted or binding.output_muted or not binding.can_write


def _record(ctx: DeliveryContext, channel_id: str, session_id: str) -> VoiceDeliveryRecord:
    assert ctx.idempotency_key is not None
    return VoiceDeliveryRecord(
        room_id=ctx.room_id,
        channel_id=channel_id,
        session_id=session_id,
        idempotency_key=ctx.idempotency_key,
        content_hash=hashlib.sha256(ctx.content.encode()).hexdigest(),
    )


def _lock_key(record: VoiceDeliveryRecord) -> str:
    digest = hashlib.sha256(f"{record.room_id}:{record.key_hash}".encode()).hexdigest()
    return f"voice-delivery:{digest}"


def _unknown(reason: str, error: Exception | None = None) -> DeliveryOutcome:
    return DeliveryOutcome(
        status="unknown",
        reason=reason,
        error=DeliveryError(code=reason, message=str(error) if error else reason, retryable=False),
    )


def _not_sent(
    reason: str, error: Exception | None = None, *, retryable: bool = True
) -> DeliveryOutcome:
    return DeliveryOutcome(
        status="failed",
        reason=reason,
        error=DeliveryError(
            code=reason, message=str(error) if error else reason, retryable=retryable
        ),
    )


def _replay(stored: VoiceDeliveryRecord, request: VoiceDeliveryRecord) -> DeliveryOutcome | None:
    if stored.content_hash != request.content_hash:
        return DeliveryOutcome(status="blocked", reason="voice_idempotency_conflict")
    if stored.retryable:
        return None
    outcome = stored.outcome or _unknown("voice_delivery_unresolved")
    return outcome.model_copy(deep=True, update={"duplicate": True})


def _aggregate(outcomes: dict[str, DeliveryOutcome]) -> DeliveryOutcome:
    worst = max(outcomes.values(), key=lambda item: _SEVERITY[item.status])
    return worst.model_copy(
        deep=True,
        update={
            "session_outcomes": outcomes,
            "session_ids": [sid for sid, item in outcomes.items() if item.status == "sent"],
            "duplicate": all(item.duplicate for item in outcomes.values()),
        },
    )


async def replay_explicit_session(ctx: DeliveryContext) -> DeliveryOutcome | None:
    """Known results do not require a session to remain active."""
    if ctx.idempotency_key is None or ctx.session_id is None or ctx.channel_id is None:
        return None
    record = _record(ctx, ctx.channel_id, ctx.session_id)
    try:
        async with ctx.kit.lock_manager.locked(_lock_key(record)):
            stored = await ctx.kit.store.get_voice_delivery(ctx.room_id, record.key_hash)
            outcome = _replay(stored, record) if stored is not None else None
            return _aggregate({ctx.session_id: outcome}) if outcome is not None else None
    except NotImplementedError:
        return DeliveryOutcome(status="blocked", reason="voice_idempotency_unsupported")
    except Exception as exc:
        return _not_sent("voice_reservation_read_failed", exc)


async def _validate(
    channel: Any, session: Any, ctx: DeliveryContext
) -> tuple[bool, DeliveryOutcome | None]:
    bindings = await ctx.kit.store.list_bindings(ctx.room_id)
    room = await ctx.kit.store.get_room(ctx.room_id)
    binding = next((b for b in bindings if b.channel_id == channel.channel_id), None)
    if room is None or binding is None or ctx.kit.get_channel(channel.channel_id) is not channel:
        return False, DeliveryOutcome(
            status="unavailable",
            reason="channel_unavailable",
            unavailable_targets=[channel.channel_id],
            error=DeliveryError(code="channel_unavailable", message="Channel unavailable"),
        )
    if room.status in (RoomStatus.CLOSED, RoomStatus.ARCHIVED):
        return False, DeliveryOutcome(status="blocked", reason="room_closed")
    if binding.access in (Access.WRITE_ONLY, Access.NONE):
        return False, DeliveryOutcome(status="blocked", reason="channel_cannot_read")
    if not any(session is active for active in active_sessions(channel, ctx.room_id)):
        return False, DeliveryOutcome(
            status="unavailable",
            reason="voice_session_replaced",
            error=DeliveryError(code="voice_session_replaced", message="Pinned session changed"),
        )
    return injection_silent(binding), None


_INJECTION_OUTCOMES: dict[str, Callable[[VoiceInjectionResult, str], DeliveryOutcome]] = {
    "sent": lambda result, sid: DeliveryOutcome(status="sent", session_ids=[sid]),
    "not_sent": lambda result, sid: _not_sent(
        result.reason or "voice_injection_not_sent", retryable=result.retryable
    ),
    "unknown": lambda result, sid: _unknown(result.reason or "voice_injection_unknown"),
}


def _reported(result: VoiceInjectionResult | None, session_id: str) -> DeliveryOutcome:
    if not isinstance(result, VoiceInjectionResult):
        return _unknown("voice_acceptance_unreported")
    return _INJECTION_OUTCOMES[result.status](result, session_id)


async def _persist(
    ctx: DeliveryContext,
    record: VoiceDeliveryRecord | None,
    outcome: DeliveryOutcome,
) -> DeliveryOutcome:
    if record is None:
        return outcome
    try:
        completed = record.model_copy(update={"outcome": outcome})
        if not await ctx.kit.store.complete_voice_delivery(completed):
            return _unknown("voice_reservation_owner_changed")
    except Exception as exc:
        logger.warning("Could not persist voice delivery outcome", exc_info=True)
        return _unknown("voice_outcome_not_persisted", exc)
    return outcome


async def _submit(
    channel: Any,
    session: Any,
    ctx: DeliveryContext,
    record: VoiceDeliveryRecord | None,
) -> DeliveryOutcome:
    started = False
    try:
        # Storage may have yielded since the initial target selection. This
        # check is immediately before the provider call, with no intervening IO.
        silent, refusal = await _validate(channel, session, ctx)
        if refusal is not None:
            return await _persist(ctx, record, refusal)
        started = True
        # An instruction directs the model; content is what someone said (§12.4).
        role = "system" if ctx.instruction else "user"
        result = (
            await channel.inject_text(session, ctx.content, role=role, silent=True)
            if silent
            else await channel.inject_text(
                session, ctx.content, role=role, chain_depth=ctx.chain_depth
            )
        )
        outcome = _reported(result, session.id)
        if outcome.status == "sent" and not any(
            session is active for active in active_sessions(channel, ctx.room_id)
        ):
            outcome = _unknown("voice_session_changed_during_submission")
    except asyncio.CancelledError:
        outcome = (
            _unknown("voice_submission_cancelled")
            if started
            else _not_sent("voice_cancelled_before_submission")
        )
        await _persist(ctx, record, outcome)
        raise
    except Exception as exc:
        outcome = (
            _unknown("voice_submission_unknown", exc)
            if started
            else _not_sent("voice_preflight_failed", exc)
        )
    return await _persist(ctx, record, outcome)


async def _deliver_session(channel: Any, session: Any, ctx: DeliveryContext) -> DeliveryOutcome:
    record = (
        _record(ctx, channel.channel_id, session.id) if ctx.idempotency_key is not None else None
    )
    lock = ctx.kit.lock_manager.locked(_lock_key(record)) if record is not None else nullcontext()
    async with lock:
        if record is not None:
            try:
                stored = await ctx.kit.store.get_voice_delivery(ctx.room_id, record.key_hash)
                previous = _replay(stored, record) if stored is not None else None
                if previous is not None:
                    return previous
                # Preflight refusal does not reserve a key or call a provider.
                _, refusal = await _validate(channel, session, ctx)
                if refusal is not None:
                    return refusal
                claimed = await ctx.kit.store.claim_voice_delivery(record)
                if claimed.attempt_id != record.attempt_id:
                    return _replay(claimed, record) or _unknown("voice_reservation_contended")
            except NotImplementedError:
                return DeliveryOutcome(status="blocked", reason="voice_idempotency_unsupported")
            except Exception as exc:
                return _not_sent("voice_reservation_failed", exc)
        return await _submit(channel, session, ctx, record)


async def deliver_to_realtime_voice(channel: Any, ctx: DeliveryContext) -> DeliveryOutcome:
    """Inject selected sessions once per key, retaining every target's outcome."""
    sessions = ctx._voice_sessions
    if sessions is None:
        sessions = active_sessions(channel, ctx.room_id)
    if not sessions:
        return DeliveryOutcome(status="unavailable", reason="voice_session_unavailable")
    outcomes = {}
    for session in sessions:
        outcomes[session.id] = await _deliver_session(channel, session, ctx)
    return _aggregate(outcomes)
