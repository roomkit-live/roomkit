"""
Activity persistence — interleaved tool call events in the store.

Demonstrates how RoomKit persists AI responses with tool calls as separate
events rather than a single concatenated text blob. Shows:

- Interleaved text + tool call events with shared correlation_id
- EventFilter for querying by type, correlation, and time
- PersistencePolicy for controlling what gets stored
- get_conversation() vs get_timeline() for different use cases

Run with:
    uv run python examples/ai_activity_persistence.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import (
    ChannelCategory,
    EventFilter,
    EventType,
    InboundMessage,
    PersistencePolicy,
    RoomEvent,
    RoomKit,
    TextContent,
    ToolCallContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider


async def main() -> None:
    # --- 1. Set up a kit with a persistence policy ---

    kit = RoomKit(
        persistence_policy=PersistencePolicy(
            exclude_types={EventType.TYPING, EventType.PRESENCE},
        ),
    )

    # --- 2. Create an AI that uses tools ---

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        if name == "search":
            return '{"results": ["RoomKit docs", "API reference"]}'
        if name == "fetch_page":
            return '{"title": "RoomKit", "summary": "Multi-channel framework"}'
        return '{"error": "unknown tool"}'

    # Mock provider: round 1 calls search, round 2 calls fetch, round 3 gives final answer
    responses = [
        AIResponse(
            content="Let me search for that.",
            finish_reason="tool_calls",
            usage={"input_tokens": 50, "output_tokens": 20},
            tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "roomkit"})],
        ),
        AIResponse(
            content="Found some results. Let me get more details.",
            finish_reason="tool_calls",
            usage={"input_tokens": 100, "output_tokens": 30},
            tool_calls=[AIToolCall(id="tc2", name="fetch_page", arguments={"url": "docs"})],
        ),
        AIResponse(
            content="RoomKit is a multi-channel conversation framework with tool support.",
            finish_reason="stop",
            usage={"input_tokens": 150, "output_tokens": 40},
        ),
    ]

    provider = MockAIProvider(ai_responses=responses, streaming=False)
    tools = [
        {"name": "search", "description": "Search the web", "parameters": {}},
        {"name": "fetch_page", "description": "Fetch a page", "parameters": {}},
    ]

    ai = AIChannel("ai-agent", provider=provider, tool_handler=tool_handler)
    ws = WebSocketChannel("ws-user")

    kit.register_channel(ai)
    kit.register_channel(ws)

    # --- 3. Create room and send a message ---

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-user")
    await kit.attach_channel(
        "demo-room",
        "ai-agent",
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": tools},
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Tell me about RoomKit"),
        )
    )

    # --- 4. Query the store ---

    store = kit.store
    print("=" * 60)
    print("FULL TIMELINE (get_timeline)")
    print("=" * 60)

    timeline = await store.get_timeline("demo-room")
    for ev in timeline:
        if ev.type == EventType.MESSAGE:
            body = ev.content.body if isinstance(ev.content, TextContent) else "?"
            src = "user" if ev.source.channel_id == "ws-user" else "ai"
            print(f"  [{ev.index}] {src}: {body[:60]}")
        elif ev.type == EventType.TOOL_CALL_START:
            assert isinstance(ev.content, ToolCallContent)
            print(f"  [{ev.index}] TOOL START: {ev.content.tool_name}({ev.content.arguments})")
        elif ev.type == EventType.TOOL_CALL_END:
            assert isinstance(ev.content, ToolCallContent)
            print(
                f"  [{ev.index}] TOOL END:   {ev.content.tool_name}"
                f" -> {ev.content.status} ({ev.content.duration_ms}ms)"
            )

    # --- 5. Filtered queries ---

    print()
    print("=" * 60)
    print("CONVERSATION ONLY (get_conversation)")
    print("=" * 60)

    conversation = await store.get_conversation("demo-room")
    for ev in conversation:
        body = ev.content.body if isinstance(ev.content, TextContent) else "?"
        src = "user" if ev.source.channel_id == "ws-user" else "ai"
        print(f"  [{ev.index}] {src}: {body[:60]}")

    print()
    print("=" * 60)
    print("TOOL CALLS ONLY (EventFilter)")
    print("=" * 60)

    tool_events = await store.list_events(
        "demo-room",
        event_filter=EventFilter(
            event_types=[EventType.TOOL_CALL_START, EventType.TOOL_CALL_END],
        ),
    )
    for ev in tool_events:
        assert isinstance(ev.content, ToolCallContent)
        print(f"  [{ev.index}] {ev.type}: {ev.content.tool_name} (status={ev.content.status})")

    # --- 6. Correlation ID groups ---

    # All AI response events share a correlation_id
    ai_events = [ev for ev in timeline if ev.source.channel_id == "ai-agent"]
    if ai_events:
        corr_id = ai_events[0].correlation_id
        print()
        print("=" * 60)
        print(f"EVENTS BY CORRELATION ID ({corr_id})")
        print("=" * 60)

        correlated = await store.list_events(
            "demo-room",
            event_filter=EventFilter(correlation_id=corr_id),
        )
        print(f"  {len(correlated)} events in this AI response group")
        for ev in correlated:
            print(f"  [{ev.index}] {ev.type}")

    await kit.close()
    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
