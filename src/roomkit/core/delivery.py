"""Framework-level delivery service.

``kit.deliver()`` sends content to a room/channel with awareness of
channel state (voice playing, user speaking, idle).  Strategies control
**when** to deliver; channel adapters handle **how**.

Usage::

    # Simple
    await kit.deliver(room_id, content="Payment confirmed.")

    # With strategy
    await kit.deliver(room_id, content="Done.", strategy=WaitForIdle(buffer=2.0))

    # String shorthand
    await kit.deliver(room_id, content="Done.", strategy="wait_for_idle")
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.delivery")

_VOICE_TYPES = frozenset({ChannelType.VOICE, ChannelType.REALTIME_VOICE})


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class DeliveryContext:
    """Context passed to delivery strategies."""

    kit: RoomKit
    room_id: str
    content: str
    channel_id: str | None = None
    metadata: dict[str, Any] | None = None

    async def find_transport_channel_id(self) -> str | None:
        """Find the best transport channel in the room.

        Prefers voice channels (most latency-sensitive), then falls
        back to any other transport channel.
        """
        bindings = await self.kit.store.list_bindings(self.room_id)

        voice_id: str | None = None
        text_id: str | None = None

        for binding in bindings:
            if binding.category != ChannelCategory.TRANSPORT:
                continue
            if binding.channel_type in _VOICE_TYPES:
                voice_id = binding.channel_id
            elif text_id is None:
                text_id = binding.channel_id

        return voice_id or text_id

    async def resolve_channel_id(self) -> str | None:
        """Resolve the target channel — explicit or auto-detected."""
        if self.channel_id is not None:
            return self.channel_id
        return await self.find_transport_channel_id()


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class DeliveryStrategy(ABC):
    """Controls when and how content is delivered to a channel."""

    @abstractmethod
    async def deliver(self, ctx: DeliveryContext) -> None:
        """Deliver the content according to this strategy."""


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class Immediate(DeliveryStrategy):
    """Send now. May interrupt ongoing TTS playback."""

    async def deliver(self, ctx: DeliveryContext) -> None:
        channel_id = await ctx.resolve_channel_id()
        if channel_id is None:
            logger.warning("Immediate: no transport channel in room %s", ctx.room_id)
            return

        await _deliver_to_channel(ctx, channel_id)


class WaitForIdle(DeliveryStrategy):
    """Wait for TTS/speech to finish, then send.

    Args:
        buffer: Extra seconds to wait after playback ends (default 1.0).
        playback_timeout: Max seconds to wait for playback (default 15.0).
    """

    def __init__(
        self,
        buffer: float = 1.0,
        playback_timeout: float = 15.0,
    ) -> None:
        self.buffer = buffer
        self.playback_timeout = playback_timeout

    async def deliver(self, ctx: DeliveryContext) -> None:
        channel_id = await ctx.resolve_channel_id()
        if channel_id is None:
            logger.warning("WaitForIdle: no transport channel in room %s", ctx.room_id)
            return

        channel = ctx.kit.get_channel(channel_id)
        if channel is not None and channel.channel_type in _VOICE_TYPES:
            # VoiceChannel: wait for playback + buffer
            # RealtimeVoiceChannel: no playback tracking, apply buffer as delay
            waited = await _wait_for_voice_idle(
                channel, ctx.room_id, self.playback_timeout, self.buffer
            )
            if not waited and self.buffer > 0:
                await asyncio.sleep(self.buffer)

        await _deliver_to_channel(ctx, channel_id)


class Queued(DeliveryStrategy):
    """Add to queue, deliver at next idle window.

    Multiple deliveries are batched into a single message.

    Args:
        buffer: Extra seconds to wait after playback ends (default 1.0).
        playback_timeout: Max seconds to wait for playback (default 15.0).
        separator: Text between batched items (default newline).
    """

    def __init__(
        self,
        buffer: float = 1.0,
        playback_timeout: float = 15.0,
        separator: str = "\n\n",
    ) -> None:
        self.buffer = buffer
        self.playback_timeout = playback_timeout
        self.separator = separator
        self._queue: list[str] = []
        self._lock = asyncio.Lock()
        self._delivering = False

    async def deliver(self, ctx: DeliveryContext) -> None:
        async with self._lock:
            self._queue.append(ctx.content)

            # If already delivering, the current delivery will pick up queued items
            if self._delivering:
                return

            self._delivering = True

        try:
            channel_id = await ctx.resolve_channel_id()
            if channel_id is None:
                logger.warning(
                    "Queued: no transport channel in room %s, %d items dropped",
                    ctx.room_id,
                    len(self._queue),
                )
                async with self._lock:
                    self._queue.clear()
                return

            channel = ctx.kit.get_channel(channel_id)
            if channel is not None and channel.channel_type in _VOICE_TYPES:
                await _wait_for_voice_idle(
                    channel, ctx.room_id, self.playback_timeout, self.buffer
                )

            # Drain queue — collect everything accumulated while waiting
            async with self._lock:
                batched = self.separator.join(self._queue)
                self._queue.clear()

            batched_ctx = DeliveryContext(
                kit=ctx.kit,
                room_id=ctx.room_id,
                content=batched,
                channel_id=channel_id,
                metadata=ctx.metadata,
            )
            await _deliver_to_channel(batched_ctx, channel_id)
        finally:
            async with self._lock:
                self._delivering = False


# ---------------------------------------------------------------------------
# String shorthand resolver
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, type[DeliveryStrategy]] = {
    "immediate": Immediate,
    "wait_for_idle": WaitForIdle,
    "queued": Queued,
}


def resolve_strategy(strategy: DeliveryStrategy | str | None) -> DeliveryStrategy | None:
    """Resolve a strategy from string shorthand or instance."""
    if strategy is None or isinstance(strategy, DeliveryStrategy):
        return strategy
    cls = _STRATEGY_MAP.get(strategy)
    if cls is None:
        msg = f"Unknown delivery strategy: {strategy!r}. Options: {list(_STRATEGY_MAP)}"
        raise ValueError(msg)
    return cls()


# ---------------------------------------------------------------------------
# Channel adapters (internal)
# ---------------------------------------------------------------------------


async def _deliver_to_channel(ctx: DeliveryContext, channel_id: str) -> None:
    """Deliver content to a channel via the appropriate mechanism."""
    channel = ctx.kit.get_channel(channel_id)
    if channel is None:
        logger.warning("Channel %s not found, delivery skipped", channel_id)
        return

    # RealtimeVoiceChannel: inject text directly
    if channel.channel_type == ChannelType.REALTIME_VOICE:
        await _deliver_to_realtime_voice(channel, ctx)
        return

    # All other channels: synthetic inbound message
    logger.debug("Delivering to %s in room %s", channel_id, ctx.room_id)
    await ctx.kit.process_inbound(
        InboundMessage(
            channel_id=channel_id,
            sender_id="system",
            content=TextContent(body=ctx.content),
            metadata=ctx.metadata or {},
        ),
        room_id=ctx.room_id,
    )


async def _deliver_to_realtime_voice(channel: Any, ctx: DeliveryContext) -> None:
    """Deliver to a RealtimeVoiceChannel via inject_text."""
    sessions = channel.get_room_sessions(ctx.room_id)
    if not sessions:
        logger.warning("RealtimeVoice: no active sessions in room %s", ctx.room_id)
        return

    for session in sessions:
        logger.debug("RealtimeVoice: injecting into session %s", session.id)
        await channel.inject_text(session, ctx.content)


async def _wait_for_voice_idle(
    channel: Any,
    room_id: str,
    timeout: float,
    buffer: float,
) -> bool:
    """Wait for voice channel to finish playback + buffer.

    Returns True if the channel was a VoiceChannel and we waited,
    False if the channel type has no playback tracking (e.g. RealtimeVoice).
    """
    from roomkit.channels.voice import VoiceChannel as _VoiceChannel

    if not isinstance(channel, _VoiceChannel):
        return False
    logger.debug("Waiting for playback idle in room %s", room_id)
    await channel.wait_playback_done(room_id, timeout=timeout)
    if buffer > 0:
        await asyncio.sleep(buffer)
    return True
