"""HTTP webhook example — send and receive messages via generic HTTP.

Run with:
    WEBHOOK_URL=https://example.com/hook uv run python examples/http_webhook.py
"""

from __future__ import annotations

import asyncio
import os

from roomkit import (
    HTTPProviderConfig,
    RoomKit,
    WebhookHTTPProvider,
    WebSocketChannel,
    parse_http_webhook,
)
from roomkit.channels import HTTPChannel


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    webhook_url = os.environ.get("WEBHOOK_URL", "")
    if not webhook_url:
        print("Set WEBHOOK_URL to run this example.")
        return

    config = HTTPProviderConfig(webhook_url=webhook_url)
    provider = WebhookHTTPProvider(config)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()
    http_channel = HTTPChannel("http-main", provider=provider)
    ws = WebSocketChannel("ws-agent")
    kit.register_channel(http_channel)
    kit.register_channel(ws)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel(
        "demo-room",
        "http-main",
        metadata={"recipient_id": "user-123"},
    )
    await kit.attach_channel("demo-room", "ws-agent")

    # --- Simulate inbound webhook --------------------------------------------
    raw_payload = {
        "sender_id": "user-123",
        "body": "Hello from HTTP!",
        "external_id": "msg-001",
    }

    inbound = parse_http_webhook(raw_payload, channel_id="http-main")
    print(f"Parsed inbound from {inbound.sender_id}: {inbound.content.body}")  # type: ignore[union-attr]
    result = await kit.process_inbound(inbound)
    print(f"  Processed: blocked={result.blocked}")

    # --- Show conversation history -------------------------------------------
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]

    await provider.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
