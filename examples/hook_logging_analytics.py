"""Async hooks for logging and analytics.

Demonstrates AFTER_BROADCAST async hooks that fire after events are
delivered. These hooks cannot block or modify events — they're for
side effects like logging, analytics, and notifications. Shows:
- Async hooks (fire-and-forget, don't block the pipeline)
- Hook filtering by channel_types and directions
- Multiple hooks with priority ordering

Run with:
    uv run python examples/hook_logging_analytics.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelDirection,
    ChannelType,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider

# Simulated analytics log
analytics_log: list[dict[str, str]] = []
audit_log: list[str] = []


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel("ai-bot", provider=MockAIProvider(responses=["Sure, I can help!"]))
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []
    ws.register_connection("conn", lambda _c, ev: inbox.append(ev))  # type: ignore[arg-type,return-value]

    await kit.create_room(room_id="analytics-room")
    await kit.attach_channel("analytics-room", "ws-user")
    await kit.attach_channel(
        "analytics-room",
        "ai-bot",
        category="intelligence",  # type: ignore[arg-type]
    )

    # --- Hook 1: Log all events (no filter) ---
    @kit.hook(
        HookTrigger.AFTER_BROADCAST,
        execution=HookExecution.ASYNC,
        name="analytics_logger",
        priority=0,
    )
    async def analytics_logger(event: RoomEvent, ctx: RoomContext) -> None:
        analytics_log.append(
            {
                "event_id": event.id,
                "channel": event.source.channel_id,
                "type": event.type.value,
                "room": event.room_id,
            }
        )

    # --- Hook 2: Audit only inbound user messages from WebSocket ---
    @kit.hook(
        HookTrigger.AFTER_BROADCAST,
        execution=HookExecution.ASYNC,
        name="ws_audit",
        priority=1,
        channel_types={ChannelType.WEBSOCKET},
        directions={ChannelDirection.INBOUND},
    )
    async def ws_audit(event: RoomEvent, ctx: RoomContext) -> None:
        if isinstance(event.content, TextContent):
            audit_log.append(f"[AUDIT] User said: {event.content.body}")

    # --- Send messages ---
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What is RoomKit?"),
        )
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Can you explain hooks?"),
        )
    )

    # Give async hooks time to fire
    await asyncio.sleep(0.1)

    # --- Show results ---
    print(f"Analytics log ({len(analytics_log)} entries):")
    for entry in analytics_log:
        print(f"  {entry['channel']:>10} | {entry['type']:>8} | {entry['event_id'][:8]}...")

    print(f"\nAudit log ({len(audit_log)} entries):")
    for entry in audit_log:
        print(f"  {entry}")


if __name__ == "__main__":
    asyncio.run(main())
