"""Normalized webhook metadata extraction for SMS providers."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.models.delivery import DeliveryStatus, InboundMessage
    from roomkit.models.event import EventContent


# ---------------------------------------------------------------------------
# Shared content helpers
# ---------------------------------------------------------------------------


def extract_media_urls(content: EventContent) -> list[str]:
    """Extract media URLs from MediaContent or CompositeContent.

    Returns a list of URLs found in the content. Returns an empty list for
    text-only content.
    """
    from roomkit.models.event import CompositeContent, MediaContent

    if isinstance(content, MediaContent):
        return [content.url]
    if isinstance(content, CompositeContent):
        urls: list[str] = []
        for part in content.parts:
            if isinstance(part, MediaContent):
                urls.append(part.url)
        return urls
    return []


def extract_text_body(content: EventContent) -> str:
    """Extract text from any content type.

    - TextContent / RichContent / SystemContent / TemplateContent → body
    - MediaContent → caption or empty string
    - CompositeContent → joined text from all parts
    """
    from roomkit.models.event import CompositeContent, MediaContent

    if isinstance(content, MediaContent):
        return content.caption or ""
    if isinstance(content, CompositeContent):
        parts: list[str] = []
        for part in content.parts:
            text = extract_text_body(part)
            if text:
                parts.append(text)
        return " ".join(parts)
    # TextContent, RichContent, SystemContent, TemplateContent all have .body
    if hasattr(content, "body") and content.body:
        return str(content.body)
    return ""


def build_inbound_content(
    body: str,
    media: list[dict[str, str | None]],
) -> EventContent:
    """Build the appropriate EventContent from text + media list.

    Args:
        body: Message text (may be empty).
        media: List of dicts with ``url`` and optional ``mime_type`` keys.

    Returns:
        TextContent, MediaContent, or CompositeContent as appropriate.
    """
    from roomkit.models.event import (
        CompositeContent,
        MediaContent,
        TextContent,
    )

    media_parts = [
        MediaContent(
            url=str(m["url"]),
            mime_type=m.get("mime_type") or "application/octet-stream",
        )
        for m in media
        if m.get("url")
    ]

    if not media_parts:
        return TextContent(body=body)

    if len(media_parts) == 1 and not body:
        return media_parts[0]

    if len(media_parts) == 1 and body:
        # Single media with text → MediaContent with caption
        mc = media_parts[0]
        return MediaContent(url=mc.url, mime_type=mc.mime_type, caption=body)

    # Multiple media, optionally with text
    parts: list[TextContent | MediaContent] = []
    if body:
        parts.append(TextContent(body=body))
    parts.extend(media_parts)
    return CompositeContent(parts=parts)  # type: ignore[arg-type]


@dataclass
class WebhookMeta:
    """Normalized metadata extracted from any SMS provider webhook.

    Attributes:
        provider: Provider name (e.g., "telnyx", "twilio").
        sender: Phone number that sent the message.
        recipient: Phone number that received the message.
        body: Message text content.
        external_id: Provider's unique message identifier.
        timestamp: When the message was received (if available).
        raw: Original webhook payload for debugging.
        media_urls: List of media attachments with url and mime_type.
        direction: Message direction ("inbound" or "outbound").
        event_type: Webhook event type (e.g., "message.received").
    """

    provider: str
    sender: str
    recipient: str
    body: str
    external_id: str | None
    timestamp: datetime | None
    raw: dict[str, Any]
    media_urls: list[dict[str, str | None]] = field(default_factory=list)
    direction: str | None = None
    event_type: str | None = None

    @property
    def is_inbound(self) -> bool:
        """Check if this webhook represents an inbound message.

        Returns True if direction is "inbound" or event_type indicates
        a received message. Returns True by default if direction/event_type
        are not available (backwards compatibility).
        """
        # Explicit outbound check
        if self.direction == "outbound":
            return False

        # Explicit inbound check
        if self.direction == "inbound":
            return True

        # Event type checks for providers that use event_type
        if self.event_type:
            # Telnyx event types
            if self.event_type in ("message.sent", "message.finalized"):
                return False
            if self.event_type == "message.received":
                return True

        # Default to True for backwards compatibility
        # (older payloads may not have direction/event_type)
        return True

    @property
    def is_status(self) -> bool:
        """Check if this webhook represents a delivery status update."""
        if self.direction != "outbound":
            return False
        status_events = {
            "message.sent",
            "message.delivered",
            "message.failed",
            "message.finalized",
        }
        return self.event_type in status_events

    def to_status(self) -> DeliveryStatus:
        """Convert to DeliveryStatus for delivery tracking."""
        from roomkit.models.delivery import DeliveryStatus

        status = (self.event_type or "unknown").replace("message.", "")
        errors = self.raw.get("data", {}).get("payload", {}).get("errors", [])
        error_code = str(errors[0].get("code", "")) if errors else None
        error_message = str(errors[0].get("detail", "")) if errors else None

        return DeliveryStatus(
            provider=self.provider,
            message_id=self.external_id or "",
            status=status,
            recipient=self.recipient,
            sender=self.sender,
            error_code=error_code,
            error_message=error_message,
            timestamp=self.timestamp,
            raw=self.raw,
        )

    def to_inbound(self, channel_id: str) -> InboundMessage:
        """Convert to InboundMessage for use with RoomKit.process_inbound().

        Args:
            channel_id: The channel ID to associate with the message.

        Returns:
            An InboundMessage ready for process_inbound().

        Raises:
            ValueError: If this is not an inbound message (e.g., outbound status webhook).

        Example:
            meta = extract_sms_meta("twilio", payload)
            sender = normalize_phone(meta.sender)
            inbound = meta.to_inbound(channel_id="sms-channel")
            result = await kit.process_inbound(inbound)
        """
        if not self.is_inbound:
            raise ValueError(
                f"Cannot convert outbound webhook to InboundMessage "
                f"(provider={self.provider}, direction={self.direction}, "
                f"event_type={self.event_type}). "
                f"This is likely a delivery status webhook, not an inbound message."
            )

        from roomkit.models.delivery import InboundMessage

        return InboundMessage(
            channel_id=channel_id,
            sender_id=self.sender,
            content=build_inbound_content(self.body, self.media_urls),
            external_id=self.external_id,
            idempotency_key=self.external_id,
            metadata={
                "provider": self.provider,
                "recipient": self.recipient,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            },
        )


def extract_voicemeup_meta(payload: dict[str, Any]) -> WebhookMeta:
    """Extract normalized metadata from a VoiceMeUp webhook payload."""
    timestamp = None
    if ts := payload.get("datetime_transmission"):
        with contextlib.suppress(ValueError, AttributeError):
            timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    media_urls: list[dict[str, str | None]] = []
    # VoiceMeUp uses "attachment" for URL and "attachment_mime_type" for type
    attachment_url = payload.get("attachment") or payload.get("attachment_url")
    if attachment_url:
        media_urls.append(
            {
                "url": attachment_url,
                "mime_type": payload.get("attachment_mime_type") or payload.get("attachment_type"),
            }
        )

    return WebhookMeta(
        provider="voicemeup",
        sender=payload.get("source_number", ""),
        recipient=payload.get("destination_number", ""),
        body=payload.get("message", ""),
        external_id=payload.get("sms_hash"),
        timestamp=timestamp,
        raw=payload,
        media_urls=media_urls,
    )


def extract_telnyx_meta(payload: dict[str, Any]) -> WebhookMeta:
    """Extract normalized metadata from a Telnyx webhook payload.

    Note: This extracts metadata from ALL Telnyx webhooks, including outbound
    status updates. Use ``meta.is_inbound`` to check if it's an inbound message
    before processing.
    """
    event_data = payload.get("data", {})
    data = event_data.get("payload", {})

    timestamp = None
    if ts := data.get("received_at"):
        with contextlib.suppress(ValueError, AttributeError):
            timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    from_data = data.get("from", {})
    to_list = data.get("to", [])

    media_urls: list[dict[str, str | None]] = []
    for m in data.get("media", []):
        if url := m.get("url"):
            media_urls.append(
                {
                    "url": url,
                    "mime_type": m.get("content_type"),
                }
            )

    return WebhookMeta(
        provider="telnyx",
        sender=from_data.get("phone_number", ""),
        recipient=to_list[0].get("phone_number", "") if to_list else "",
        body=data.get("text", ""),
        external_id=data.get("id"),
        timestamp=timestamp,
        raw=payload,
        media_urls=media_urls,
        direction=data.get("direction"),
        event_type=event_data.get("event_type"),
    )


def extract_twilio_meta(payload: dict[str, Any]) -> WebhookMeta:
    """Extract normalized metadata from a Twilio webhook payload.

    Note: Twilio sends form-encoded data. Convert to dict first:
        payload = dict(await request.form())
    """
    media_urls: list[dict[str, str | None]] = []
    num_media = int(payload.get("NumMedia", "0"))
    for i in range(num_media):
        url = payload.get(f"MediaUrl{i}")
        if url:
            media_urls.append(
                {
                    "url": url,
                    "mime_type": payload.get(f"MediaContentType{i}"),
                }
            )

    return WebhookMeta(
        provider="twilio",
        sender=payload.get("From", ""),
        recipient=payload.get("To", ""),
        body=payload.get("Body", ""),
        external_id=payload.get("MessageSid"),
        timestamp=None,  # Twilio doesn't include timestamp in webhook
        raw=payload,
        media_urls=media_urls,
    )


def extract_sinch_meta(payload: dict[str, Any]) -> WebhookMeta:
    """Extract normalized metadata from a Sinch webhook payload."""
    timestamp = None
    if ts := payload.get("received_at"):
        with contextlib.suppress(ValueError, AttributeError):
            timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    media_urls: list[dict[str, str | None]] = []
    for m in payload.get("media", []):
        if url := m.get("url"):
            media_urls.append(
                {
                    "url": url,
                    "mime_type": m.get("mimeType"),
                }
            )

    return WebhookMeta(
        provider="sinch",
        sender=payload.get("from", ""),
        recipient=payload.get("to", ""),
        body=payload.get("body", ""),
        external_id=payload.get("id"),
        timestamp=timestamp,
        raw=payload,
        media_urls=media_urls,
    )


_ExtractorFn = Callable[[dict[str, Any]], WebhookMeta]

_EXTRACTORS: dict[str, _ExtractorFn] = {
    "voicemeup": extract_voicemeup_meta,
    "telnyx": extract_telnyx_meta,
    "twilio": extract_twilio_meta,
    "sinch": extract_sinch_meta,
}


def extract_sms_meta(provider: str, payload: dict[str, Any]) -> WebhookMeta:
    """Extract normalized metadata from any supported SMS provider webhook.

    Args:
        provider: Provider name (e.g., "voicemeup", "telnyx").
        payload: Raw webhook payload dictionary.

    Returns:
        Normalized WebhookMeta with provider-agnostic fields.

    Raises:
        ValueError: If the provider is not supported.
    """
    extractor = _EXTRACTORS.get(provider.lower())
    if extractor is None:
        supported = ", ".join(sorted(_EXTRACTORS.keys()))
        raise ValueError(f"Unknown SMS provider: {provider}. Supported: {supported}")
    return extractor(payload)
