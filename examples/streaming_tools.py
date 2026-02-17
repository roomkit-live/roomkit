"""Streaming text delivery with tool calls.

Demonstrates the streaming tool loop: AI text deltas are delivered
progressively over WebSocket while tool calls execute between rounds.
Shows:
- MockAIProvider with ai_responses for multi-round tool interactions
- Streaming tool loop yielding text in real time
- WebSocket stream_send_fn receiving progressive chunks
- Tool handler executing between generation rounds

Run with:
    uv run python examples/streaming_tools.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    StreamChunk,
    StreamEnd,
    StreamMessage,
    StreamStart,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

# ---------------------------------------------------------------------------
# 1. Tool handler — simulates order lookup and ticket creation
# ---------------------------------------------------------------------------


async def tool_handler(name: str, arguments: dict[str, Any]) -> str:
    """Execute tool calls from the AI."""
    print(f"    [tool] Executing: {name}({arguments})")
    await asyncio.sleep(0.1)  # simulate API latency

    if name == "lookup_order":
        return json.dumps(
            {
                "order_id": arguments.get("order_id", "???"),
                "status": "shipped",
                "eta": "2026-02-20",
                "carrier": "FedEx",
            }
        )
    if name == "create_ticket":
        return json.dumps(
            {
                "ticket_id": "TKT-9001",
                "status": "open",
                "priority": arguments.get("priority", "medium"),
            }
        )
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# 2. AI responses — simulates a two-round conversation with tool calls
# ---------------------------------------------------------------------------

ai_responses = [
    # Round 1: AI streams "Let me look that up..." then calls lookup_order
    AIResponse(
        content="Let me look up your order right away.",
        finish_reason="tool_calls",
        usage={"prompt_tokens": 50, "completion_tokens": 15},
        tool_calls=[
            AIToolCall(
                id="call_1",
                name="lookup_order",
                arguments={"order_id": "ORD-42"},
            ),
        ],
    ),
    # Round 2: AI streams the final answer with order details
    AIResponse(
        content=(
            "Great news! Your order ORD-42 has shipped via FedEx "
            "and is expected to arrive by February 20th."
        ),
        finish_reason="stop",
        usage={"prompt_tokens": 120, "completion_tokens": 30},
    ),
]


# ---------------------------------------------------------------------------
# 3. Main demo
# ---------------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    provider = MockAIProvider(ai_responses=ai_responses, streaming=True)
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful support agent with access to order lookup.",
        tool_handler=tool_handler,
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    # -- Track streaming delivery on the WebSocket side --

    stream_chunks: list[str] = []

    async def on_stream(conn_id: str, msg: StreamMessage) -> None:
        if isinstance(msg, StreamStart):
            print(f"  [{conn_id}] Stream started")
        elif isinstance(msg, StreamChunk):
            stream_chunks.append(msg.delta)
            print(f"  [{conn_id}] Chunk: {msg.delta!r}")
        elif isinstance(msg, StreamEnd):
            print(f"  [{conn_id}] Stream ended")

    async def on_event(conn_id: str, event: RoomEvent) -> None:
        body = event.content.body if isinstance(event.content, TextContent) else "?"
        print(f"  [{conn_id}] Event: {body!r}")

    ws.register_connection("client", on_event, stream_send_fn=on_stream)

    # -- Set up room with tools --

    await kit.create_room(room_id="support-room")
    await kit.attach_channel("support-room", "ws-user")
    await kit.attach_channel(
        "support-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "tools": [
                {
                    "name": "lookup_order",
                    "description": "Look up an order by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                        },
                        "required": ["order_id"],
                    },
                },
                {
                    "name": "create_ticket",
                    "description": "Create a support ticket",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                    },
                },
            ],
        },
    )

    # -- Send user message --

    print("=== Streaming Tool Loop Demo ===\n")
    print("User: Where is my order ORD-42?\n")

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="customer",
            content=TextContent(body="Where is my order ORD-42?"),
        )
    )

    # -- Summary --

    print("\n=== Summary ===")
    print(f"  Provider calls: {len(provider.calls)}")
    print(f"  Stream chunks received: {len(stream_chunks)}")
    print(f"  Full streamed text: {''.join(stream_chunks)!r}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
