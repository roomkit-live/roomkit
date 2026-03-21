"""Shared Twilio webhook parsing utilities."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.providers.sms.meta import build_inbound_content


def parse_twilio_payload(
    payload: dict[str, Any],
    channel_id: str,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> InboundMessage:
    """Parse a Twilio webhook POST body into an InboundMessage.

    Shared logic for SMS and RCS webhooks. Both use the same body/media
    extraction and differ only in metadata fields.

    Args:
        payload: Form-encoded webhook body converted to dict.
        channel_id: RoomKit channel ID.
        extra_metadata: Additional metadata fields specific to the channel type.
    """
    body = payload.get("Body", "")

    media: list[dict[str, str | None]] = []
    num_media = min(int(payload.get("NumMedia", "0")), 20)
    for i in range(num_media):
        url = payload.get(f"MediaUrl{i}")
        if url:
            media.append({"url": url, "mime_type": payload.get(f"MediaContentType{i}")})

    metadata: dict[str, Any] = {
        "to": payload.get("To", ""),
        "account_sid": payload.get("AccountSid", ""),
        "num_media": payload.get("NumMedia", "0"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return InboundMessage(
        channel_id=channel_id,
        sender_id=payload.get("From", ""),
        content=build_inbound_content(body, media),
        external_id=payload.get("MessageSid"),
        idempotency_key=payload.get("MessageSid"),
        metadata=metadata,
    )
