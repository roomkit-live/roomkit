"""VoiceMeUp SMS example — send an SMS and parse an inbound webhook.

Run with:
    uv run python examples/voicemeup_sms.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    InboundMessage,
    RoomKit,
    SMSChannel,
    VoiceMeUpConfig,
    VoiceMeUpSMSProvider,
    WebSocketChannel,
    parse_voicemeup_webhook,
)


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    config = VoiceMeUpConfig(
        username="demo_user",
        auth_token="demo_token",
        from_number="+15145551234",
        environment="sandbox",
    )
    provider = VoiceMeUpSMSProvider(config)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()
    sms = SMSChannel("sms-main", provider=provider, from_number=config.from_number)
    ws = WebSocketChannel("ws-agent")
    kit.register_channel(sms)
    kit.register_channel(ws)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "sms-main")
    await kit.attach_channel("demo-room", "ws-agent")

    print(f"Provider ready — base URL: {config.base_url}")
    print(f"From number: {config.from_number}")

    # --- Simulate inbound webhook --------------------------------------------
    raw_webhook = {
        "message": "Hello from the outside!",
        "source_number": "+15145559999",
        "destination_number": "+15145551234",
        "direction": "inbound",
        "sms_hash": "example-hash-001",
        "datetime_transmission": "2026-01-27T14:30:00Z",
    }

    inbound: InboundMessage = parse_voicemeup_webhook(raw_webhook, channel_id="sms-main")
    print(f"\nParsed inbound message from {inbound.sender_id}:")
    print(f"  Body: {inbound.content.body}")  # type: ignore[union-attr]
    print(f"  External ID: {inbound.external_id}")
    print(f"  Metadata: {inbound.metadata}")

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
