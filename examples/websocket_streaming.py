"""WebSocket streaming text delivery.

Demonstrates progressive text delivery over WebSocket using the
stream_start / stream_chunk / stream_end protocol.  A streaming-capable
connection sees each token arrive in real time, while a legacy connection
receives only the final complete message.

Run with:
    uv run python examples/websocket_streaming.py
"""

from __future__ import annotations

import asyncio
import json

from roomkit import (
    RoomEvent,
    StreamChunk,
    StreamEnd,
    StreamMessage,
    StreamStart,
    WebSocketChannel,
)
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import EventSource, TextContent
from roomkit.models.room import Room


async def main() -> None:
    ws = WebSocketChannel("chat")

    # -- Streaming client (e.g. a modern SPA) ----------------------------------

    async def on_stream_message(conn_id: str, msg: StreamMessage) -> None:
        if isinstance(msg, StreamStart):
            print(f"[{conn_id}] stream started  (id={msg.stream_id[:8]}...)")
        elif isinstance(msg, StreamChunk):
            print(f"[{conn_id}] chunk: {msg.delta!r}  (accumulated: {msg.text!r})")
        elif isinstance(msg, StreamEnd):
            print(f"[{conn_id}] stream ended    final={msg.event.content.body!r}")  # type: ignore[union-attr]

    async def on_event_streaming(conn_id: str, event: RoomEvent) -> None:
        # Streaming connections won't receive events via this path during
        # deliver_stream, but they still need a regular send_fn for non-AI
        # messages (e.g. user chat, system events).
        print(f"[{conn_id}] event: {json.dumps(event.model_dump(), default=str)[:80]}...")

    ws.register_connection(
        "spa-client",
        on_event_streaming,
        stream_send_fn=on_stream_message,
    )

    # -- Legacy client (no streaming support) -----------------------------------

    async def on_event_legacy(conn_id: str, event: RoomEvent) -> None:
        body = event.content.body if isinstance(event.content, TextContent) else "?"
        print(f"[{conn_id}] full message: {body!r}")

    ws.register_connection("legacy-client", on_event_legacy)

    # -- Simulate a streaming AI response ---------------------------------------

    print(f"supports_streaming_delivery = {ws.supports_streaming_delivery}\n")

    async def ai_token_stream() -> ...:  # type: ignore[type-arg]
        tokens = ["Hello", ",", " how", " can", " I", " help", " you", " today", "?"]
        for token in tokens:
            await asyncio.sleep(0.05)  # simulate LLM latency
            yield token

    placeholder_event = RoomEvent(
        room_id="room-1",
        source=EventSource(channel_id="ai", channel_type=ChannelType.AI),
        content=TextContent(body=""),
    )
    binding = ChannelBinding(
        channel_id="chat",
        room_id="room-1",
        channel_type=ChannelType.WEBSOCKET,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
    )
    context = RoomContext(room=Room(id="room-1"), bindings=[binding])

    await ws.deliver_stream(ai_token_stream(), placeholder_event, binding, context)


if __name__ == "__main__":
    asyncio.run(main())
