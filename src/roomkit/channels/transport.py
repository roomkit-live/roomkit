"""Unified transport channel implementation."""

from __future__ import annotations

import logging
import time as _time
from typing import Any

from roomkit.channels.addressing import (
    ChannelAddressing,
    addressing_of,
    checked_country_code,
    written_recipient,
)
from roomkit.channels.base import Channel
from roomkit.core.exceptions import NoRecipientError, ProviderDeliveryError
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import (
    ChannelData,
    CompositeContent,
    EventSource,
    MediaContent,
    RoomEvent,
)

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
        addressing: ChannelAddressing | None = None,
        country_code: str | None = None,
        defaults: dict[str, Any] | None = None,
        requires_recipient: bool = True,
    ) -> None:
        """Initialise a transport channel.

        Args:
            channel_id: Unique identifier for this channel instance.
            channel_type: The channel type (SMS, email, etc.).
            provider: Provider that handles external delivery (e.g. ElasticEmailProvider).
            capabilities: Media and feature capabilities for this channel.
            addressing: How the channel's addresses are written, the binding
                metadata key its recipient is written under, the inbound
                metadata key its chats come under, and whether replies go to
                the sender (RFC §5.7). Omitted, the channel takes its type's
                (:func:`~roomkit.channels.addressing.addressing_of`): an SMS
                channel compares numbers in E.164 and writes its recipient
                under ``phone_number``. A recipient that names a conversation
                (the sender's address, or a chat) names its correspondent: the
                router admits no one else through the binding (RFC §10.4).
                On a channel whose replies go to a chat only, a message that
                does not carry its chat has no address: it records no one,
                and a room opened for it names neither correspondent nor
                recipient.
            country_code: On a channel addressed by phone number, the country
                code a number written without one takes (``"1"``,
                ``"+44"``); without it such a number is kept as written.
            defaults: Default kwargs passed to ``provider.send()``.  If a default
                value is ``None``, the actual value is read from the binding metadata
                at delivery time.
            requires_recipient: Whether a delivery needs a recipient in the
                binding's metadata. A channel whose provider delivers to a
                destination of its own (a webhook's configured URL) passes
                ``False``: its provider is called without one (RFC §10.2
                step 3d), and so is a channel whose addressing names no
                recipient key.

        Raises:
            ValueError: *country_code* is not one to three digits, the first
                not ``0``.
        """
        super().__init__(channel_id)
        self.channel_type = channel_type
        self._provider = provider
        self._capabilities = capabilities or ChannelCapabilities()
        self._addressing = addressing or addressing_of(channel_type)
        self._country_code = (
            checked_country_code(country_code) if country_code is not None else None
        )
        self._defaults: dict[str, Any] = defaults or {}
        self._requires_recipient = (
            requires_recipient and self._addressing.recipient_key is not None
        )

    @property
    def addressing(self) -> ChannelAddressing:
        """The channel's addressing declaration (RFC §5.7)."""
        return self._addressing

    def _propagate_telemetry(self) -> None:
        """Propagate telemetry to transport provider."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None and self._provider is not None:
            self._provider._telemetry = telemetry

    @property
    def info(self) -> dict[str, Any]:
        """Return non-None default values as channel info metadata."""
        return {k: v for k, v in self._defaults.items() if v is not None}

    def capabilities(self) -> ChannelCapabilities:
        """Return the channel's media and feature capabilities."""
        return self._capabilities

    def normalize_address(self, address: str) -> str:
        if not address:
            return address
        return self._addressing.normalize(address, self._country_code)

    @property
    def recipient_key(self) -> str | None:
        addressing = self._addressing
        return addressing.recipient_key if addressing.names_conversations else None

    def recipient_address(self, binding: ChannelBinding) -> str | None:
        key = self.recipient_key
        if key is None:
            return None
        value = written_recipient(binding.metadata, key)
        return None if value is None else self.normalize_address(str(value))

    def recipient_metadata(self, address: str) -> dict[str, str]:
        """``{recipient_key: address}`` when the recipient is a conversation's address."""
        key = self.recipient_key
        if key is None or not address:
            return {}
        return {key: address}

    def conversation_address(self, message: InboundMessage) -> str:
        chat = self._chat_of(message)
        if chat:
            return chat
        if self._addressing.chat_key is not None and not self._addressing.replies_to_sender:
            # Replies go to a chat only: a message that does not carry its
            # chat has no address (RFC §5.7). It records no one, and is
            # never routed by its author, whose private chat is another
            # conversation.
            return ""
        return message.sender_id

    def reply_metadata(self, message: InboundMessage) -> dict[str, str]:
        # The same precedence as conversation_address: the chat the message
        # carries, then the sender, so a room replies where it is found.
        chat = self._chat_of(message)
        if chat:
            return self.recipient_metadata(chat)
        if self._addressing.replies_to_sender:
            return self.recipient_metadata(message.sender_id)
        return {}

    def _chat_of(self, message: InboundMessage) -> str:
        """The chat *message* was posted in, when this channel answers chats."""
        key = self._addressing.chat_key
        if key is None:
            return ""
        chat = written_recipient(message.metadata, key)
        return "" if chat is None else str(chat)

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        """Convert an inbound message into a room event."""
        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.context import get_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        span_id = telemetry.start_span(
            SpanKind.INBOUND_PIPELINE,
            "channel.handle_inbound",
            parent_id=get_current_span(),
            room_id=context.room.id,
            channel_id=self.channel_id,
            attributes={
                Attr.CHANNEL_ID: self.channel_id,
                "channel_type": str(self.channel_type),
            },
        )
        try:
            # Use MMS channel type for SMS with media content
            channel_type = self.channel_type
            if channel_type == ChannelType.SMS and self._has_media(message.content):
                channel_type = ChannelType.MMS

            event = RoomEvent(
                room_id=context.room.id,
                type=message.event_type,
                source=EventSource(
                    channel_id=self.channel_id,
                    channel_type=channel_type,
                    participant_id=message.sender_id or None,
                    external_id=message.external_id,
                    provider=self.provider_name,
                    # RFC §5.2 MUST — preserved unmodified. This is the one
                    # place the provider's own payload survives parsing.
                    raw_payload=message.raw_payload,
                    provider_message_id=message.provider_message_id,
                ),
                content=message.content,
                channel_data=ChannelData(
                    provider=self.provider_name,
                    external_id=message.external_id,
                    thread_id=message.thread_id,
                ),
                idempotency_key=message.idempotency_key,
                metadata=message.metadata,
            )
            telemetry.end_span(span_id)
            return event
        except Exception as exc:
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise

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

        The recipient address is read from the binding's metadata under the
        addressing's recipient key.
        Extra kwargs are built from ``defaults``: fixed values are passed as-is,
        ``None`` defaults are resolved from binding metadata at delivery time.
        """
        if self._provider is None:
            logger.debug("No provider configured for %s, skipping delivery", self.channel_id)
            return ChannelOutput.empty()

        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.context import get_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()

        key = self._addressing.recipient_key
        recipient = written_recipient(binding.metadata, key) if key is not None else None
        if recipient is None and self._requires_recipient:
            # Nobody to send to: refused here rather than handed to the
            # provider as "" or as a value no one is reached at (`true`, a
            # blank string), which a lenient one would acknowledge (RFC §5.7,
            # §10.2 step 3d).
            raise NoRecipientError(self.channel_id, key or "")
        to = "" if recipient is None else recipient
        kwargs: dict[str, Any] = {}
        for key, value in self._defaults.items():
            kwargs[key] = binding.metadata.get(key, value) if value is None else value

        provider_name = getattr(self._provider, "name", type(self._provider).__name__)
        span_id = telemetry.start_span(
            SpanKind.DELIVERY,
            "framework.delivery",
            parent_id=get_current_span(),
            room_id=event.room_id,
            channel_id=self.channel_id,
            attributes={
                Attr.DELIVERY_CHANNEL_TYPE: str(self.channel_type),
                Attr.DELIVERY_RECIPIENT: to,
                Attr.PROVIDER: provider_name,
            },
        )
        t0 = _time.monotonic()
        try:
            result = await self._provider.send(event, to=to, **kwargs)
            duration_ms = (_time.monotonic() - t0) * 1000
            attrs: dict[str, Any] = {Attr.DELIVERY_SUCCESS: result.success}
            if result.error:
                attrs[Attr.DELIVERY_ERROR] = result.error
            if result.provider_message_id:
                attrs[Attr.DELIVERY_MESSAGE_ID] = result.provider_message_id
            if result.success:
                telemetry.end_span(span_id, attributes=attrs)
            else:
                telemetry.end_span(
                    span_id,
                    status="error",
                    error_message=result.error or "send_failed",
                    attributes=attrs,
                )
            telemetry.record_metric(
                "roomkit.delivery.duration_ms",
                duration_ms,
                unit="ms",
                attributes={Attr.PROVIDER: provider_name},
            )
        except Exception as exc:
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise
        if not result.success:
            raise ProviderDeliveryError(result)
        return ChannelOutput(provider_result=result)
