"""Facebook Messenger webhook parsing helpers."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent


def parse_messenger_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> list[InboundMessage]:
    """Convert a Facebook Messenger webhook payload into InboundMessages.

    Facebook sends batches of events under ``payload["entry"]``.  Each entry
    contains a ``messaging`` list with individual messages.  Only messages that
    carry a ``message.text`` field are returned; delivery/read receipts and
    postbacks are silently skipped.
    """
    messages: list[InboundMessage] = []
    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):
            msg = event.get("message")
            if msg is None or "text" not in msg:
                continue
            sender_id = event.get("sender", {}).get("id", "")
            messages.append(
                InboundMessage(
                    channel_id=channel_id,
                    sender_id=sender_id,
                    content=TextContent(body=msg["text"]),
                    external_id=msg.get("mid"),
                    idempotency_key=msg.get("mid"),
                    metadata={
                        "recipient_id": event.get("recipient", {}).get("id", ""),
                        "timestamp": event.get("timestamp", 0),
                    },
                )
            )
    return messages
