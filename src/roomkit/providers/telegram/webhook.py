"""Telegram webhook parsing helpers."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import LocationContent, MediaContent, TextContent


def parse_telegram_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> list[InboundMessage]:
    """Convert a Telegram Update payload into InboundMessages.

    Telegram sends one update at a time (unless using ``getUpdates``).
    Only ``message`` updates are processed; edits, channel posts, and
    callback queries are silently skipped.

    For photo messages the largest available ``file_id`` is stored in
    metadata under ``"file_id"``; callers must resolve it to a download
    URL via the Bot API ``getFile`` endpoint.
    """
    messages: list[InboundMessage] = []

    msg = payload.get("message")
    if msg is None:
        return messages

    sender = msg.get("from", {})
    sender_id = str(sender.get("id", ""))
    message_id = str(msg.get("message_id", ""))

    content: TextContent | MediaContent | LocationContent | None = None
    extra_metadata: dict[str, Any] = {}

    if "text" in msg:
        content = TextContent(body=msg["text"])
    elif "photo" in msg:
        # Telegram sends multiple sizes; take the largest (last).
        photos = msg["photo"]
        file_id = photos[-1]["file_id"] if photos else ""
        caption = msg.get("caption", "")
        content = TextContent(body=caption)
        extra_metadata["file_id"] = file_id
        extra_metadata["media_type"] = "photo"
    elif "location" in msg:
        loc = msg["location"]
        content = LocationContent(
            latitude=loc["latitude"],
            longitude=loc["longitude"],
        )

    if content is None:
        return messages

    metadata = {
        "chat_id": str(msg.get("chat", {}).get("id", "")),
        "date": msg.get("date", 0),
        **extra_metadata,
    }

    messages.append(
        InboundMessage(
            channel_id=channel_id,
            sender_id=sender_id,
            content=content,
            external_id=message_id,
            idempotency_key=message_id,
            metadata=metadata,
        )
    )
    return messages
