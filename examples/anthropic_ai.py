"""Anthropic AI example — AI-powered assistant using Anthropic Claude.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/anthropic_ai.py
"""

from __future__ import annotations

import asyncio
import os

from roomkit import (
    AnthropicAIProvider,
    AnthropicConfig,
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to run this example.")
        return

    config = AnthropicConfig(api_key=api_key)
    provider = AnthropicAIProvider(config)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Keep answers concise.",
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    # Capture what the user receives back.
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_receive)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-user")
    await kit.attach_channel("demo-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    # --- Send a message and get an AI response -------------------------------
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What is RoomKit?"),
        )
    )
    print(f"Sent message -> blocked={result.blocked}")

    # Show the AI response delivered back to the user.
    for ev in inbox:
        print(f"  AI replied: {ev.content.body}")  # type: ignore[union-attr]

    # --- Show conversation history -------------------------------------------
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
