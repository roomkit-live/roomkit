"""Unified transport channel implementation."""

from __future__ import annotations

import logging
import time as _time
from collections.abc import Callable
from typing import Any

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
        recipient_key: str = "recipient_id",
        defaults: dict[str, Any] | None = None,
        address_normalizer: Callable[[str], str] | None = None,
        replies_to_sender: bool = False,
        reply_metadata_key: str | None = None,
        requires_recipient: bool = True,
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
            address_normalizer: How this channel writes an address (E.164 for
                the channels addressed by phone number); unchanged if omitted.
            replies_to_sender: Whether a reply goes to the address the
                correspondent writes from (a phone number, an email address),
                rather than to a chat or a URL. Then the framework records an
                inbound sender as the binding's recipient, and a binding's
                recipient names its correspondent: the router admits no one
                else through it (RFC §10.4).
            reply_metadata_key: On a channel that replies to a chat or a
                conversation rather than to the sender, the inbound
                message's metadata key naming it (``chat_id`` on Telegram):
                a room opened for a message replies there, and the chat is
                the message's conversation (:meth:`conversation_address`):
                its room is found by the chat, not by the sender, and a
                binding's recipient names its chat as a phone channel's names
                its correspondent. Exclusive with *replies_to_sender*.
            requires_recipient: Whether a delivery needs a recipient in the
                binding's metadata. A channel whose provider delivers to a
                destination of its own (a webhook's configured URL) passes
                ``False``: its provider is called without one (RFC §10.2
                step 3d).
        """
        if replies_to_sender and reply_metadata_key is not None:
            raise ValueError(
                "A channel replies to the sender's address or to a chat, not both: "
                "pass replies_to_sender or reply_metadata_key"
            )
        super().__init__(channel_id)
        self.channel_type = channel_type
        self._provider = provider
        self._capabilities = capabilities or ChannelCapabilities()
        self._recipient_key = recipient_key
        self._defaults: dict[str, Any] = defaults or {}
        self._address_normalizer = address_normalizer
        self._replies_to_sender = replies_to_sender
        self._reply_metadata_key = reply_metadata_key
        self._requires_recipient = requires_recipient

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
        if self._address_normalizer is None or not address:
            return address
        return self._address_normalizer(address)

    @property
    def _recipient_is_conversation(self) -> bool:
        """Whether the recipient is a conversation's address: the sender's, or a chat."""
        return self._replies_to_sender or self._reply_metadata_key is not None

    def recipient_address(self, binding: ChannelBinding) -> str | None:
        if not self._recipient_is_conversation:
            return None
        value = binding.metadata.get(self._recipient_key)
        return self.normalize_address(str(value)) if value else None

    def recipient_metadata(self, address: str) -> dict[str, str]:
        """``{recipient_key: address}`` when the recipient is a conversation's address."""
        if not self._recipient_is_conversation or not address:
            return {}
        return {self._recipient_key: address}

    def conversation_address(self, message: InboundMessage) -> str:
        if self._reply_metadata_key is not None:
            chat = message.metadata.get(self._reply_metadata_key)
            if chat:
                return str(chat)
        return message.sender_id

    def reply_metadata(self, message: InboundMessage) -> dict[str, str]:
        if self._replies_to_sender:
            return self.recipient_metadata(message.sender_id)
        if self._reply_metadata_key is None:
            return {}
        value = message.metadata.get(self._reply_metadata_key)
        return {self._recipient_key: str(value)} if value else {}

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

        The recipient address is read from ``binding.metadata[recipient_key]``.
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

        to = binding.metadata.get(self._recipient_key, "")
        if not to and self._requires_recipient:
            # Nobody to send to: refused here rather than handed to the
            # provider as "", which a lenient one would acknowledge (RFC §10.2 step 3d).
            raise NoRecipientError(self.channel_id, self._recipient_key)
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
