"""Rich content with buttons, cards, and quick replies.

Demonstrates how to send rich formatted messages with interactive
elements using RichContent. Shows:
- Markdown/HTML formatted text
- Buttons (action buttons with payloads)
- Cards (rich card layouts)
- Quick replies (suggested responses)

Run with:
    uv run python examples/rich_content_buttons.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    InboundMessage,
    RichContent,
    RoomEvent,
    RoomKit,
    WebSocketChannel,
)


async def main() -> None:
    kit = RoomKit()

    ws_user = WebSocketChannel("ws-user")
    ws_agent = WebSocketChannel("ws-agent")
    kit.register_channel(ws_user)
    kit.register_channel(ws_agent)

    agent_inbox: list[RoomEvent] = []

    async def agent_recv(_conn: str, event: RoomEvent) -> None:
        agent_inbox.append(event)

    ws_user.register_connection("user-conn", lambda _c, _e: asyncio.sleep(0))
    ws_agent.register_connection("agent-conn", agent_recv)

    await kit.create_room(room_id="rich-room")
    await kit.attach_channel("rich-room", "ws-user")
    await kit.attach_channel("rich-room", "ws-agent")

    # --- Example 1: Markdown with quick replies ---
    print("1. Markdown message with quick replies...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=RichContent(
                body="**How can I help you today?**\n\nChoose an option below:",
                format="markdown",
                plain_text="How can I help you today? Choose an option below:",
                quick_replies=["Check balance", "Transfer money", "Talk to agent"],
            ),
        )
    )

    # --- Example 2: HTML with buttons ---
    print("2. HTML message with action buttons...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=RichContent(
                body="<h3>Order Confirmation</h3><p>Your order #12345 has been placed.</p>",
                format="html",
                plain_text="Order Confirmation: Your order #12345 has been placed.",
                buttons=[
                    {
                        "type": "url",
                        "label": "Track Order",
                        "url": "https://example.com/track/12345",
                    },
                    {
                        "type": "postback",
                        "label": "Cancel Order",
                        "payload": "cancel_order_12345",
                    },
                ],
            ),
        )
    )

    # --- Example 3: Cards (product catalog) ---
    print("3. Rich cards (product catalog)...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=RichContent(
                body="Here are our top picks:",
                format="markdown",
                plain_text="Here are our top picks",
                cards=[
                    {
                        "title": "Premium Plan",
                        "subtitle": "$29/month",
                        "image_url": "https://example.com/premium.png",
                        "buttons": [
                            {"label": "Subscribe", "payload": "subscribe_premium"},
                        ],
                    },
                    {
                        "title": "Basic Plan",
                        "subtitle": "$9/month",
                        "image_url": "https://example.com/basic.png",
                        "buttons": [
                            {"label": "Subscribe", "payload": "subscribe_basic"},
                        ],
                    },
                ],
            ),
        )
    )

    # --- Show what the agent received ---
    print(f"\nAgent received {len(agent_inbox)} rich messages:")
    for i, ev in enumerate(agent_inbox, 1):
        if isinstance(ev.content, RichContent):
            print(f"\n  Message {i} ({ev.content.format}):")
            print(f"    Body: {ev.content.plain_text or ev.content.body[:60]}...")
            if ev.content.quick_replies:
                print(f"    Quick replies: {ev.content.quick_replies}")
            if ev.content.buttons:
                labels = [b.get("label", "?") for b in ev.content.buttons]
                print(f"    Buttons: {labels}")
            if ev.content.cards:
                titles = [c.get("title", "?") for c in ev.content.cards]
                print(f"    Cards: {titles}")


if __name__ == "__main__":
    asyncio.run(main())
