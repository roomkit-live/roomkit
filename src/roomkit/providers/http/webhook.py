"""Inbound HTTP webhook parser."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent


def parse_http_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> InboundMessage:
    """Convert a simple JSON body into an InboundMessage.

    Expected payload shape::

        {
            "sender_id": "user-123",
            "body": "Hello!",
            "external_id": "msg-456",   // optional
            "metadata": {}              // optional
        }
    """
    return InboundMessage(
        channel_id=channel_id,
        sender_id=payload["sender_id"],
        content=TextContent(body=payload.get("body", "")),
        external_id=payload.get("external_id"),
        idempotency_key=payload.get("external_id"),
        metadata=payload.get("metadata", {}),
    )
