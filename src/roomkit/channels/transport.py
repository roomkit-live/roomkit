"""Unified transport channel implementation."""

from __future__ import annotations

import logging
from typing import Any

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import CompositeContent, EventSource, MediaContent, RoomEvent

logger = logging.getLogger("roomkit.channels.transport")


class TransportChannel(Channel):
    """Generic transport channel driven by configuration rather than subclassing.

    All transport channels (SMS, Email, WhatsApp, Messenger, HTTP) share the
    same inbound/deliver logic.  The only differences are data: which
    ``ChannelType``, which ``ChannelCapabilities``, which metadata key holds the
    recipient address, and which extra kwargs to pass to the provider's
    ``send()`` method.

    Use the factory functions (``SMSChannel``, ``EmailChannel``, …) in
    ``roomkit.channels`` for convenient construction.
    """

    def __init__(
        self,
        channel_id: str,
        channel_type: ChannelType,
        *,
        provider: Any = None,
        capabilities: ChannelCapabilities | None = None,
        recipient_key: str = "recipient_id",
        defaults: dict[str, Any] | None = None,
    ) -> None:
        """Initialise a transport channel.

        Args:
            channel_id: Unique identifier for this channel instance.
            channel_type: The channel type (SMS, email, etc.).
            provider: Provider that handles external delivery (e.g. ElasticEmailProvider).
            capabilities: Media and feature capabilities for this channel.
            recipient_key: Binding metadata key that holds the recipient address.
            defaults: Default kwargs passed to ``provider.send()``.  If a default
                value is ``None``, the actual value is read from the binding metadata
                at delivery time.
        """
        super().__init__(channel_id)
        self.channel_type = channel_type
        self._provider = provider
        self._capabilities = capabilities or ChannelCapabilities()
        self._recipient_key = recipient_key
        self._defaults: dict[str, Any] = defaults or {}

    @property
    def info(self) -> dict[str, Any]:
        """Return non-None default values as channel info metadata."""
        return {k: v for k, v in self._defaults.items() if v is not None}

    def capabilities(self) -> ChannelCapabilities:
        """Return the channel's media and feature capabilities."""
        return self._capabilities

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        """Convert an inbound message into a room event."""
        # Use MMS channel type for SMS with media content
        channel_type = self.channel_type
        if channel_type == ChannelType.SMS and self._has_media(message.content):
            channel_type = ChannelType.MMS

        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=channel_type,
                participant_id=message.sender_id or None,
                external_id=message.external_id,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )

    @staticmethod
    def _has_media(content: Any) -> bool:
        """Check if content contains media."""
        if isinstance(content, MediaContent):
            return True
        if isinstance(content, CompositeContent):
            return any(isinstance(part, MediaContent) for part in content.parts)
        return False

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Deliver an event to the external recipient via the provider.

        The recipient address is read from ``binding.metadata[recipient_key]``.
        Extra kwargs are built from ``defaults``: fixed values are passed as-is,
        ``None`` defaults are resolved from binding metadata at delivery time.
        """
        if self._provider is None:
            logger.debug("No provider configured for %s, skipping delivery", self.channel_id)
            return ChannelOutput.empty()

        to = binding.metadata.get(self._recipient_key, "")
        kwargs: dict[str, Any] = {}
        for key, value in self._defaults.items():
            kwargs[key] = binding.metadata.get(key, value) if value is None else value
        await self._provider.send(event, to=to, **kwargs)
        return ChannelOutput.empty()
