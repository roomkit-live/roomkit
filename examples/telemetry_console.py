"""Telemetry with ConsoleTelemetryProvider — logs span summaries to terminal.

Demonstrates how to enable telemetry in RoomKit to see span timing
for hooks, inbound pipeline, and LLM generation.

Run with:
    uv run python examples/telemetry_console.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    ChannelCategory,
    ConsoleTelemetryProvider,
    HookResult,
    HookTrigger,
    InboundMessage,
    MockAIProvider,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel

# Enable logging to see telemetry output
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")


async def main() -> None:
    # Create RoomKit with console telemetry
    telemetry = ConsoleTelemetryProvider()
    kit = RoomKit(telemetry=telemetry)

    # Channels
    ws = WebSocketChannel("ws-user")
    ai = AIChannel("ai-bot", provider=MockAIProvider(responses=["Hello from AI!"]))

    kit.register_channel(ws)
    kit.register_channel(ai)

    # Hook
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="logger_hook")
    async def log_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    # Create room and attach channels
    await kit.create_room(room_id="demo")
    await kit.attach_channel("demo", "ws-user")
    await kit.attach_channel("demo", "ai-bot", category=ChannelCategory.INTELLIGENCE)

    # Send a message — telemetry spans will be logged
    print("\n--- Sending message (watch for [SPAN] logs) ---\n")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user1",
            content=TextContent(body="What's the weather?"),
        )
    )

    telemetry.close()
    print("\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())
