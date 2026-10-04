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
    _voice_sessions: list[Any] | None = field(default=None, repr=False)
    _voice_channel: Any = field(default=None, repr=False)
    _wait_for_turn: bool = field(default=True, repr=False)

    async def find_transport_channel_id(self) -> str | None:
        """Prefer voice, then the first other transport bound to the room.

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
            if binding.channel_type in _VOICE_TYPES:
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
    """Send now. May interrupt ongoing TTS playback."""

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
            and left.session_id == right.session_id
            and left._voice_sessions == right._voice_sessions
            and left.metadata == right.metadata
        )


class Queued(DeliveryStrategy):
    """Batch compatible unkeyed requests at idle; preserve each caller's outcome.

    Keyed requests publish individually, including duplicate keys. One caller
    owns the drain; followers wait for their own request to be processed.
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
        request = _QueuedRequest(ctx, channel_id, asyncio.get_running_loop().create_future())
        self._queue.append(request)
        if self._delivering:
            return await request.result
        self._delivering = True
        try:
            await self._drain()
        finally:
            self._delivering = False
            # Cancellation of the drain must wake every waiting caller.
            for pending in self._queue:
                pending.result.cancel()
            self._queue.clear()
        return request.result.result()

    async def _drain(self) -> None:
        while self._queue:
            first = self._queue[0]
            if first.result.cancelled():
                self._queue.pop(0)
                continue
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
            except Exception as exc:
                first.result.set_result(_strategy_failure(exc))
                self._queue.pop(0)
                continue

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
                for pending in batch:
                    pending.result.cancel()
                raise
            except Exception as exc:
                outcome = _strategy_failure(exc)
            for pending in batch:
                if not pending.result.done():
                    pending.result.set_result(outcome.model_copy())


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
