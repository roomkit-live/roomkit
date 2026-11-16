"""Proactive delivery strategies: when to publish text or inject voice."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from roomkit.channels.base import hosts_realtime_model
from roomkit.channels.voice import VoiceChannel
from roomkit.core._delivery_targets import (
    deliver_to_channel as _deliver_to_channel,
)
from roomkit.core._delivery_targets import (
    prepare_delivery,
)
from roomkit.core.locks import _holds_a_room_lock
from roomkit.models.delivery import DeliveryError, DeliveryOutcome
from roomkit.models.enums import ChannelCategory, ChannelType

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.delivery")
_VOICE_TYPES = frozenset(
    {ChannelType.VOICE, ChannelType.REALTIME_VOICE, ChannelType.REALTIME_AUDIO_VIDEO}
)


@dataclass
class DeliveryContext:
    """One request, including its destination pinned before an idle wait."""

    kit: RoomKit
    room_id: str
    content: str
    channel_id: str | None = None
    metadata: dict[str, Any] | None = None
    addressed_to: list[str] | None = None
    idempotency_key: str | None = None
    session_id: str | None = None
    chain_depth: int = 0
    instruction: bool = False
    runtime_record: str | None = None
    """The provenance of the runtime's own text (``DeliveryItem.runtime_record``)."""
    _voice_sessions: list[Any] | None = field(default=None, repr=False)
    _voice_channel: Any = field(default=None, repr=False)
    _wait_for_turn: bool = field(default=True, repr=False)

    async def find_transport_channel_id(self) -> str | None:
        """Prefer voice, then the first other transport bound to the room.

        Voice is a voice channel type or any channel whose realtime model is
        plugged in at delivery time (a conference with one,
        :func:`hosts_realtime_model`).
        An intelligence channel attached as a transport is not one: an
        instruction delivered through it would re-enter the channel it is for.
        """
        bindings = await self.kit.store.list_bindings(self.room_id)
        voice_id: str | None = None
        text_id: str | None = None
        for binding in bindings:
            if binding.category != ChannelCategory.TRANSPORT or self._is_intelligence(
                binding.channel_id
            ):
                continue
            if binding.channel_type in _VOICE_TYPES or hosts_realtime_model(
                self.kit.get_channel(binding.channel_id)
            ):
                voice_id = binding.channel_id
            elif text_id is None:
                text_id = binding.channel_id
        return voice_id or text_id

    def _is_intelligence(self, channel_id: str) -> bool:
        channel = self.kit.get_channel(channel_id)
        return channel is not None and channel.category == ChannelCategory.INTELLIGENCE

    async def resolve_channel_id(self) -> str | None:
        """Resolve an explicit destination or auto-detect its transport."""
        if self.channel_id is not None:
            return self.channel_id
        return await self.find_transport_channel_id()


class DeliveryStrategy(ABC):
    """Controls when content is delivered; report its actual outcome."""

    @abstractmethod
    async def deliver(self, ctx: DeliveryContext) -> DeliveryOutcome | None:
        """Deliver the request. A custom strategy may leave its outcome unknown."""


class Immediate(DeliveryStrategy):
    """Send now. On a voice session, it plays once the playback under way has
    been sent whole: a response never cuts the agent off (RFC §12.2)."""

    async def deliver(self, ctx: DeliveryContext) -> DeliveryOutcome:
        channel_id, refusal = await prepare_delivery(ctx)
        if refusal is not None:
            return refusal
        assert channel_id is not None
        return await _deliver_to_channel(ctx, channel_id)


class WaitForIdle(DeliveryStrategy):
    """Wait for voice idle, then send; a timeout falls back to immediate."""

    def __init__(self, buffer: float = 1.0, playback_timeout: float = 15.0) -> None:
        self.buffer = buffer
        self.playback_timeout = playback_timeout

    async def deliver(self, ctx: DeliveryContext) -> DeliveryOutcome:
        channel_id, refusal = await prepare_delivery(ctx)
        if refusal is not None:
            return refusal
        assert channel_id is not None
        channel = ctx.kit.get_channel(channel_id)
        if channel is not None:
            try:
                await _wait_for_voice_idle(
                    channel, ctx.room_id, self.playback_timeout, self.buffer, ctx._voice_sessions
                )
            except TimeoutError:
                logger.warning("Voice idle timeout in room %s; delivering", ctx.room_id)
        return await _deliver_to_channel(ctx, channel_id)


@dataclass
class _QueuedRequest:
    context: DeliveryContext
    channel_id: str
    result: asyncio.Future[DeliveryOutcome]
    turn: asyncio.Future[None] = field(default_factory=lambda: _loop_future())
    """Set when the drain passes to this request's caller, its request at the
    head of the queue: it delivers the batch it heads in its own task, as it
    made the request (RFC §8.3, §22.2)."""

    def compatible(self, other: _QueuedRequest) -> bool:
        """Keyed publications retain individual events and outcomes."""
        left, right = self.context, other.context
        return (
            left.idempotency_key is None
            and right.idempotency_key is None
            and left.kit is right.kit
            and left.room_id == right.room_id
            and self.channel_id == other.channel_id
            and left.channel_id == right.channel_id
            and left.addressed_to == right.addressed_to
            and left.chain_depth == right.chain_depth
            and left.instruction == right.instruction
            and left.runtime_record == right.runtime_record
            and left.session_id == right.session_id
            and left._voice_sessions == right._voice_sessions
            and left.metadata == right.metadata
        )


class Queued(DeliveryStrategy):
    """Batch compatible unkeyed requests at idle; preserve each caller's outcome.

    Keyed requests publish individually, including duplicate keys. One caller
    drains the queue at a time: it delivers the batch its request heads, then
    passes the drain to the caller whose request heads the queue next, which
    delivers that batch in its own task, with its own depth, frames and room
    lock: a request that heads a batch is delivered as its caller made it,
    and a draining caller that holds the room's lock leaves the callers
    behind it free to deliver once it returns. A compatible request batched
    behind another is delivered with it, by that request's caller. A caller
    holding a room's lock while another caller drains delivers at once,
    unbatched, as :class:`Immediate` does: nested in the section that holds
    the lock (RFC §9.5.1), it would otherwise wait behind a batch whose
    delivery needs that lock (for that room, whichever room it delivers
    into: one strategy may queue for several), or be batched into one, that
    is, wait on itself. Followers wait for their own request's outcome or
    for their turn.
    """

    def __init__(
        self, buffer: float = 1.0, playback_timeout: float = 15.0, separator: str = "\n\n"
    ) -> None:
        self.buffer = buffer
        self.playback_timeout = playback_timeout
        self.separator = separator
        self._queue: list[_QueuedRequest] = []
        self._delivering = False

    async def deliver(self, ctx: DeliveryContext) -> DeliveryOutcome:
        channel_id, refusal = await prepare_delivery(ctx)
        if refusal is not None:
            return refusal
        assert channel_id is not None
        if self._delivering and _holds_a_room_lock(ctx.kit.lock_manager):
            return await _deliver_to_channel(ctx, channel_id)
        request = _QueuedRequest(ctx, channel_id, _loop_future())
        self._queue.append(request)
        if self._delivering and not await self._wait_for_turn(request):
            return request.result.result()
        self._delivering = True
        try:
            await self._drain(request)
        except BaseException:
            # A drain cut short (its caller cancelled) withdraws its own
            # request, and the callers behind it get the drain: each still
            # receives its own outcome (RFC §22.2).
            request.result.cancel()
            self._pass_on()
            raise
        return request.result.result()

    async def _wait_for_turn(self, request: _QueuedRequest) -> bool:
        """Wait for *request*'s outcome or for the drain to pass to its
        caller; whether it passed. A caller cancelled meanwhile withdraws its
        request, and hands on a drain it was handed."""
        try:
            await asyncio.wait({request.result, request.turn}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            request.result.cancel()
            if request.turn.done():
                self._pass_on()
            raise
        return not request.result.done()

    async def _drain(self, mine: _QueuedRequest) -> None:
        """Deliver the batch *mine* heads, then hand the drain on."""
        while self._queue and self._queue[0].result.cancelled():
            self._queue.pop(0)
        if self._queue and self._queue[0] is mine:
            await self._deliver_batch(mine)
        self._pass_on()

    def _pass_on(self) -> None:
        """Hand the drain to the caller whose request heads the queue, or let
        it go when none waits."""
        while self._queue and self._queue[0].result.cancelled():
            self._queue.pop(0)
        if self._queue:
            self._queue[0].turn.set_result(None)
        else:
            self._delivering = False

    async def _deliver_batch(self, first: _QueuedRequest) -> None:
        """Deliver *first* at idle, with the compatible requests behind it,
        and give each its outcome."""
        channel = first.context.kit.get_channel(first.channel_id)
        try:
            if channel is not None:
                await _wait_for_voice_idle(
                    channel,
                    first.context.room_id,
                    self.playback_timeout,
                    self.buffer,
                    first.context._voice_sessions,
                )
        except TimeoutError:
            # A wait longer than playback_timeout delivers the batch anyway,
            # as WaitForIdle's does (RFC §22.2).
            logger.warning("Voice idle timeout in room %s; delivering", first.context.room_id)
        except Exception as exc:
            first.result.set_result(_strategy_failure(exc))
            self._queue.pop(0)
            return
        batch = [first]
        remainder = []
        for pending in self._queue[1:]:
            if not pending.result.cancelled() and first.compatible(pending):
                batch.append(pending)
            else:
                remainder.append(pending)
        self._queue = remainder
        ctx = replace(
            first.context,
            content=self.separator.join(item.context.content for item in batch),
            _wait_for_turn=False,
        )
        try:
            outcome = await _deliver_to_channel(ctx, first.channel_id)
        except asyncio.CancelledError:
            # Cut short as it was sent, the batch may or may not have reached
            # the channel: the callers it carried for others learn that,
            # rather than a cancel nothing asked of them (RFC §22.2).
            for pending in batch[1:]:
                if not pending.result.done():
                    pending.result.set_result(
                        DeliveryOutcome(status="unknown", reason="strategy_outcome_unknown")
                    )
            raise
        except Exception as exc:
            outcome = _strategy_failure(exc)
        for pending in batch:
            if not pending.result.done():
                pending.result.set_result(outcome.model_copy())


def _loop_future() -> asyncio.Future[Any]:
    return asyncio.get_running_loop().create_future()


def _strategy_failure(exc: Exception) -> DeliveryOutcome:
    return DeliveryOutcome(
        status="failed",
        reason="strategy_failed",
        error=DeliveryError(code=type(exc).__name__, message=str(exc)),
    )


_STRATEGY_MAP: dict[str, type[DeliveryStrategy]] = {
    "immediate": Immediate,
    "wait_for_idle": WaitForIdle,
    "queued": Queued,
}


def resolve_strategy(strategy: DeliveryStrategy | str | None) -> DeliveryStrategy | None:
    """Resolve a built-in shorthand or a custom strategy instance."""
    if strategy is None or isinstance(strategy, DeliveryStrategy):
        return strategy
    cls = _STRATEGY_MAP.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown delivery strategy: {strategy!r}. Options: {list(_STRATEGY_MAP)}"
        )
    return cls()


async def _wait_for_voice_idle(
    channel: Any,
    room_id: str,
    timeout: float,
    buffer: float,
    sessions: list[Any] | None = None,
) -> None:
    """Wait for voice playback or the realtime model's idle, then a buffer.

    A channel that is neither returns at once, with no buffer.
    """
    if isinstance(channel, VoiceChannel):
        await channel.wait_playback_done(room_id, timeout=timeout)
    elif hosts_realtime_model(channel):
        session_ids = None if sessions is None else [s.id for s in sessions]
        await channel.wait_idle(room_id, timeout=timeout, session_ids=session_ids)
    else:
        return
    if buffer > 0:
        await asyncio.sleep(buffer)
