"""Multi-channel bridge — the unified conversation demo.

Demonstrates RoomKit's core value proposition: a single room bridging
multiple channels with an AI assistant. Messages from any channel are
automatically broadcast to all others. Shows:
- Multiple transport channels in one room (WebSocket, SMS, Email, HTTP)
- AI intelligence channel responding to all messages
- Content flowing seamlessly between channels
- Each channel's inbox receives all messages from other channels
- Using mock providers for local testing without credentials

Run with:
    uv run python examples/multichannel_bridge.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    InboundMessage,
    MockEmailProvider,
    MockHTTPProvider,
    MockSMSProvider,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels import EmailChannel, HTTPChannel, SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider


async def main() -> None:
    kit = RoomKit()

    # --- Register channels ---
    # WebSocket: real-time web client
    ws = WebSocketChannel("ws-webapp")

    # SMS: mobile user via text message (mock provider)
    sms = SMSChannel("sms-mobile", provider=MockSMSProvider())

    # Email: email participant (mock provider)
    email = EmailChannel("email-user", provider=MockEmailProvider())

    # HTTP: external system integration (mock provider)
    http = HTTPChannel("http-crm", provider=MockHTTPProvider())

    # AI: intelligent assistant
    ai = AIChannel(
        "ai-assistant",
        provider=MockAIProvider(responses=[
            "Thanks for reaching out! I'm looking into your request.",
            "I've found the information you need.",
            "Is there anything else I can help with?",
        ]),
        system_prompt="You are a helpful support assistant.",
    )

    for channel in [ws, sms, email, http, ai]:
        kit.register_channel(channel)

    # --- Track messages per channel ---
    ws_inbox: list[RoomEvent] = []
    sms_inbox: list[RoomEvent] = []  # noqa: F841

    async def ws_recv(_conn: str, event: RoomEvent) -> None:
        ws_inbox.append(event)

    ws.register_connection("webapp-conn", ws_recv)

    # --- Create room and attach all channels ---
    await kit.create_room(
        room_id="bridge-room",
        metadata={"topic": "Customer Support", "priority": "high"},
    )

    await kit.attach_channel("bridge-room", "ws-webapp")
    await kit.attach_channel(
        "bridge-room", "sms-mobile",
        metadata={"recipient_phone": "+15551234567"},
    )
    await kit.attach_channel(
        "bridge-room", "email-user",
        metadata={"recipient_email": "user@example.com"},
    )
    await kit.attach_channel(
        "bridge-room", "http-crm",
        metadata={"webhook_url": "https://crm.example.com/api/messages"},
    )
    await kit.attach_channel(
        "bridge-room", "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
    )

    # --- Simulate messages from different channels ---

    # 1. SMS user sends a message
    print("1. SMS user sends a message...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms-mobile",
            sender_id="+15551234567",
            content=TextContent(body="Hi, I need help with my subscription"),
        )
    )

    # 2. WebSocket user replies
    print("2. WebSocket user replies...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-webapp",
            sender_id="agent-web",
            content=TextContent(body="I see your subscription issue. Let me check."),
        )
    )

    # 3. HTTP webhook from CRM
    print("3. CRM sends an update via HTTP...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="http-crm",
            sender_id="crm-system",
            content=TextContent(body="CRM Note: Customer subscription expires in 3 days"),
        )
    )

    # --- Show what each channel received ---
    print(f"\n--- WebSocket inbox ({len(ws_inbox)} messages) ---")
    for ev in ws_inbox:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] {ev.content.body}")

    # --- Show full conversation timeline ---
    events = await kit.store.list_events("bridge-room")
    msg_events = [e for e in events if e.type.value == "message"]
    print(f"\n--- Full Conversation Timeline ({len(msg_events)} messages) ---")
    for ev in msg_events:
        if isinstance(ev.content, TextContent):
            source = ev.source.channel_id
            direction = ">>>" if ev.source.channel_type.value == "ai" else "-->"
            print(f"  {source:>15} {direction} {ev.content.body}")

    # --- Show channel bindings ---
    bindings = await kit.store.list_bindings("bridge-room")
    print(f"\n--- Channel Bindings ({len(bindings)}) ---")
    for b in bindings:
        meta_parts = (f"{k}={v}" for k, v in b.metadata.items()) if b.metadata else []
        meta_summary = ", ".join(meta_parts) or "none"
        print(
            f"  {b.channel_id:>15}: type={b.channel_type},"
            f" category={b.category}, meta=[{meta_summary}]"
        )

    await kit.close()
    print("\nDone! All channels bridged in a single room.")


if __name__ == "__main__":
    asyncio.run(main())
