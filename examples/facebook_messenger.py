"""Facebook Messenger example — send and receive messages via Messenger.

Run with:
    FB_PAGE_ACCESS_TOKEN=... uv run python examples/facebook_messenger.py
"""

from __future__ import annotations

import asyncio
import os

from roomkit import (
    FacebookMessengerProvider,
    MessengerConfig,
    RoomKit,
    WebSocketChannel,
    parse_messenger_webhook,
)
from roomkit.channels import MessengerChannel


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    token = os.environ.get("FB_PAGE_ACCESS_TOKEN", "")
    if not token:
        print("Set FB_PAGE_ACCESS_TOKEN to run this example.")
        return

    config = MessengerConfig(page_access_token=token)
    provider = FacebookMessengerProvider(config)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()
    messenger = MessengerChannel("msg-main", provider=provider)
    ws = WebSocketChannel("ws-agent")
    kit.register_channel(messenger)
    kit.register_channel(ws)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel(
        "demo-room",
        "msg-main",
        metadata={"facebook_user_id": "RECIPIENT_PSID"},
    )
    await kit.attach_channel("demo-room", "ws-agent")

    # --- Simulate inbound webhook --------------------------------------------
    raw_webhook = {
        "object": "page",
        "entry": [
            {
                "id": "PAGE_ID",
                "time": 1700000000000,
                "messaging": [
                    {
                        "sender": {"id": "USER_PSID"},
                        "recipient": {"id": "PAGE_ID"},
                        "timestamp": 1700000000000,
                        "message": {
                            "mid": "mid.example",
                            "text": "Hello from Messenger!",
                        },
                    }
                ],
            }
        ],
    }

    inbound_messages = parse_messenger_webhook(raw_webhook, channel_id="msg-main")
    for inbound in inbound_messages:
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
