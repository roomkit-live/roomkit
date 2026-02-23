"""Tool call ephemeral events.

Demonstrates how AI channels broadcast TOOL_CALL_START and TOOL_CALL_END
ephemeral events during tool execution. Shows:
- Subscribing to tool call events in a room
- TOOL_CALL_START fires before tool execution with tool names and arguments
- TOOL_CALL_END fires after with results and duration_ms
- The streamed text stays clean — no inline XML
- publish_tool_call() for manual broadcasting (e.g., custom agents)

Run with:
    uv run python examples/tool_call_events.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import (
    AIChannel,
    AIResponse,
    AIToolCall,
    ChannelCategory,
    EphemeralEvent,
    EphemeralEventType,
    InboundMessage,
    MockAIProvider,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

# Collect tool call events
tool_events: list[EphemeralEvent] = []
streamed_chunks: list[str] = []


async def main() -> None:
    # --- Setup ---

    # Define tool call responses: round 1 calls a tool, round 2 returns text
    responses = [
        AIResponse(
            content="Let me look that up.",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[
                AIToolCall(id="tc1", name="search", arguments={"query": "RoomKit features"}),
            ],
        ),
        AIResponse(
            content="RoomKit supports SMS, Email, Voice, WebSocket, and AI channels.",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        ),
    ]

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        """Simulate a tool that takes some time."""
        await asyncio.sleep(0.1)  # Simulate work
        return f"Found 5 results for: {args.get('query', '')}"

    provider = MockAIProvider(ai_responses=responses, streaming=True)
    ai = AIChannel("ai-agent", provider=provider, tool_handler=tool_handler)

    ws = WebSocketChannel("ws-user")

    kit = RoomKit()
    kit.register_channel(ai)
    kit.register_channel(ws)

    # Wire up WebSocket delivery to capture streamed text
    async def capture_stream(conn_id: str, event: Any) -> None:
        pass  # Messages delivered via normal routing

    ws.register_connection("user-conn", capture_stream)

    await kit.create_room(room_id="tool-room")
    await kit.attach_channel("tool-room", "ws-user")
    await kit.attach_channel(
        "tool-room",
        "ai-agent",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "tools": [
                {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ]
        },
    )

    # --- Subscribe to ephemeral events ---
    async def on_ephemeral(event: EphemeralEvent) -> None:
        if event.type in (EphemeralEventType.TOOL_CALL_START, EphemeralEventType.TOOL_CALL_END):
            tool_events.append(event)

    sub_id = await kit.subscribe_room("tool-room", on_ephemeral)

    # --- Send a message that triggers tool use ---
    print("User: What features does RoomKit have?")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user-1",
            content=TextContent(body="What features does RoomKit have?"),
        )
    )

    # Wait for events to propagate
    await asyncio.sleep(0.5)

    # --- Display tool call events ---
    print(f"\nReceived {len(tool_events)} tool call events:\n")

    for ev in tool_events:
        if ev.type == EphemeralEventType.TOOL_CALL_START:
            tools = ev.data["tool_calls"]
            names = ", ".join(t["name"] for t in tools)
            print(f"  TOOL_CALL_START (round {ev.data['round']})")
            print(f"    Tools: {names}")
            for t in tools:
                print(f"    - {t['name']}({t.get('arguments', {})})")

        elif ev.type == EphemeralEventType.TOOL_CALL_END:
            tools = ev.data["tool_calls"]
            duration = ev.data.get("duration_ms", "?")
            print(f"  TOOL_CALL_END (round {ev.data['round']}, {duration}ms)")
            for t in tools:
                result = t.get("result", "")
                print(f"    - {t['name']}: {result[:80]}")

    # --- Manual publishing (for custom agents) ---
    print("\n--- Manual publish_tool_call() ---")
    await kit.publish_tool_call(
        "tool-room",
        "ai-agent",
        [{"id": "manual-1", "name": "fetch_data", "arguments": {"url": "https://example.com"}}],
        EphemeralEventType.TOOL_CALL_START,
    )
    await asyncio.sleep(0.05)

    await kit.publish_tool_call(
        "tool-room",
        "ai-agent",
        [{"id": "manual-1", "name": "fetch_data", "result": "200 OK"}],
        EphemeralEventType.TOOL_CALL_END,
        duration_ms=450,
    )
    await asyncio.sleep(0.05)

    print(f"Total events after manual publish: {len(tool_events)}")

    # --- Verify text stream is clean ---
    timeline = await kit.store.list_events("tool-room")
    for ev in timeline:
        if isinstance(ev.content, TextContent) and ev.source.channel_id == "ai-agent":
            body = ev.content.body
            assert "<invoke" not in body, "XML should not appear in stored text"
            assert "<result>" not in body, "XML should not appear in stored text"
            print(f"\nAgent response (clean text): {body[:100]}")

    # Cleanup
    await kit.unsubscribe_room(sub_id)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
