"""Background task delivery strategies.

Controls **when and how** delegated task results are proactively delivered
to the parent conversation.  The framework always injects the result into
the notified agent's system prompt — the strategy decides whether to also
send a synthetic inbound message that triggers the agent to respond
immediately (rather than waiting for the next user turn).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.models.enums import ChannelCategory, ChannelType

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTaskResult

logger = logging.getLogger("roomkit.tasks.delivery")

_VOICE_TYPES = frozenset({ChannelType.VOICE, ChannelType.REALTIME_VOICE})


@dataclass
class TaskDeliveryContext:
    """Context passed to delivery strategies after task completion."""

    kit: RoomKit
    result: DelegatedTaskResult
    notify_channel_id: str

    @property
    def room_id(self) -> str:
        """Parent room that should receive the delivery."""
        return self.result.parent_room_id

    async def find_transport_channel_id(self) -> str | None:
        """Find the best transport channel in the parent room.

        Prefers voice channels (most latency-sensitive), then falls back
        to any other transport channel.

        Returns:
            Channel ID of a transport channel, or ``None`` if none found.
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


class BackgroundTaskDeliveryStrategy(ABC):
    """Controls proactive delivery of background task results.

    After a delegated task completes, the framework:

    1. Injects the result into the notified agent's system prompt (always).
    2. Fires the ``ON_TASK_COMPLETED`` hook (always).
    3. Calls ``strategy.deliver(ctx)`` (this method).

    The strategy decides whether/when to send a synthetic inbound message
    that triggers the agent to respond about the result.
    """

    @abstractmethod
    async def deliver(self, ctx: TaskDeliveryContext) -> None:
        """Deliver the task result proactively.

        Args:
            ctx: Delivery context with access to the framework, result,
                and helper methods.
        """


class ContextOnlyDelivery(BackgroundTaskDeliveryStrategy):
    """No proactive delivery — wait for the user's next message.

    The result is available in the agent's system prompt and will be
    referenced when the user next speaks.  This is the default behavior
    and maintains backward compatibility.
    """

    async def deliver(self, ctx: TaskDeliveryContext) -> None:
        logger.debug(
            "ContextOnly: task %s result injected, waiting for next user turn",
            ctx.result.task_id,
        )


class ImmediateDelivery(BackgroundTaskDeliveryStrategy):
    """Deliver immediately via a synthetic inbound message.

    Sends ``process_inbound()`` right away, which triggers the agent to
    generate a response about the completed task.  For voice this may
    interrupt ongoing TTS playback.
    """

    def __init__(self, prompt: str | None = None) -> None:
        self._prompt = prompt

    def _build_prompt(self, ctx: TaskDeliveryContext) -> str:
        if self._prompt is not None:
            return self._prompt
        return (
            f"[The background task from {ctx.result.agent_id} just completed. "
            f"The result is in your context. Share it with the user now.]"
        )

    async def deliver(self, ctx: TaskDeliveryContext) -> None:
        transport_id = await ctx.find_transport_channel_id()
        if transport_id is None:
            logger.warning(
                "ImmediateDelivery: no transport channel in room %s, skipping",
                ctx.room_id,
            )
            return

        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import TextContent

        logger.debug(
            "ImmediateDelivery: sending prompt via channel %s in room %s",
            transport_id,
            ctx.room_id,
        )
        await ctx.kit.process_inbound(
            InboundMessage(
                channel_id=transport_id,
                sender_id="system",
                content=TextContent(body=self._build_prompt(ctx)),
            ),
            room_id=ctx.room_id,
        )


class WaitForIdleDelivery(BackgroundTaskDeliveryStrategy):
    """Wait for TTS playback to finish, then deliver.

    For voice channels, waits until ``VoiceChannel.wait_playback_done()``
    returns before sending the synthetic inbound message.  This avoids
    interrupting the agent mid-sentence.

    For non-voice channels (text, WebSocket, etc.), delivers immediately.
    """

    def __init__(
        self,
        prompt: str | None = None,
        playback_timeout: float = 15.0,
        interrupt_playback: bool = False,
    ) -> None:
        self._prompt = prompt
        self.playback_timeout = playback_timeout
        self.interrupt_playback = interrupt_playback

    def _build_prompt(self, ctx: TaskDeliveryContext) -> str:
        if self._prompt is not None:
            return self._prompt
        return (
            f"[The background task from {ctx.result.agent_id} just completed. "
            f"The result is in your context. Share it with the user now.]"
        )

    async def deliver(self, ctx: TaskDeliveryContext) -> None:
        transport_id = await ctx.find_transport_channel_id()
        if transport_id is None:
            logger.warning(
                "WaitForIdleDelivery: no transport channel in room %s, skipping",
                ctx.room_id,
            )
            return

        # If the transport is a voice channel, wait for playback to finish
        channel = ctx.kit.get_channel(transport_id)
        if channel is not None and channel.channel_type in _VOICE_TYPES:
            from roomkit.channels.voice import VoiceChannel

            if isinstance(channel, VoiceChannel):
                if self.interrupt_playback:
                    logger.debug(
                        "WaitForIdleDelivery: interrupting playback in room %s",
                        ctx.room_id,
                    )
                    await channel.interrupt_all(ctx.room_id, reason="task_delivery")
                else:
                    logger.debug(
                        "WaitForIdleDelivery: waiting for playback done in room %s",
                        ctx.room_id,
                    )
                    await channel.wait_playback_done(ctx.room_id, timeout=self.playback_timeout)

        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import TextContent

        logger.debug(
            "WaitForIdleDelivery: sending prompt via channel %s in room %s",
            transport_id,
            ctx.room_id,
        )
        await ctx.kit.process_inbound(
            InboundMessage(
                channel_id=transport_id,
                sender_id="system",
                content=TextContent(body=self._build_prompt(ctx)),
            ),
            room_id=ctx.room_id,
        )
