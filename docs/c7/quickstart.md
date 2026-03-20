# Quickstart

A complete working example: two WebSocket users chatting with an AI assistant in a moderated room.

```python
from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    HookResult,
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


async def main() -> None:
    # 1. Create the framework instance
    kit = RoomKit()

    # 2. Create channels
    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    ai = AIChannel("ai-assistant", provider=MockAIProvider(responses=["Got it!"]))

    # 3. Register channels with the framework
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)
    kit.register_channel(ai)

    # 4. Wire up receive callbacks (in production, WebSocket sends to clients)
    alice_inbox: list[RoomEvent] = []
    bob_inbox: list[RoomEvent] = []

    async def alice_recv(_conn: str, event: RoomEvent) -> None:
        alice_inbox.append(event)

    async def bob_recv(_conn: str, event: RoomEvent) -> None:
        bob_inbox.append(event)

    ws_alice.register_connection("alice-conn", alice_recv)
    ws_bob.register_connection("bob-conn", bob_recv)

    # 5. Create a room and attach channels
    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-alice")
    await kit.attach_channel("demo-room", "ws-bob")
    await kit.attach_channel(
        "demo-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE
    )

    # 6. Add a BEFORE_BROADCAST hook for content moderation
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="profanity_filter")
    async def profanity_filter(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent) and "badword" in event.content.body:
            return HookResult.block("Message blocked by profanity filter")
        return HookResult.allow()

    # 7. Send messages through the inbound pipeline
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Hello everyone!"),
        )
    )
    print(f"Alice sent 'Hello everyone!' -> blocked={result.blocked}")

    # Both Bob and the AI receive Alice's message.
    # The AI responds with "Got it!" which is broadcast to Alice and Bob.

    # 8. Query stored conversation history
    events = await kit.store.list_events("demo-room")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")


if __name__ == "__main__":
    asyncio.run(main())
```

## What Just Happened

1. **RoomKit()** created the framework with in-memory defaults (store, locks, realtime).
2. **Channels** were registered globally, then **attached** to a room.
3. AI channel was attached with `category=ChannelCategory.INTELLIGENCE` — it receives messages and generates responses.
4. A **BEFORE_BROADCAST** hook runs synchronously before every broadcast. It can `block()`, `allow()`, or `modify()` events.
5. **process_inbound()** ran the full pipeline: route -> parse -> identity -> hooks -> store -> broadcast.
6. The AI's response was automatically routed back through the same pipeline (with chain depth tracking to prevent loops).

## Core Pattern

Every RoomKit application follows this pattern:

```python
from roomkit import RoomKit

# 1. Create framework
kit = RoomKit()

# 2. Register channels
kit.register_channel(channel)

# 3. Create rooms and attach channels
await kit.create_room(room_id="my-room")
await kit.attach_channel("my-room", "channel-id")

# 4. Process inbound messages
await kit.process_inbound(InboundMessage(...))
```

## Next Steps

- Add hooks for moderation, logging, or analytics (see Hooks)
- Configure AI providers for real LLM responses (see AI Channels)
- Add voice with STT/TTS (see Voice Channels)
- Set up multi-agent orchestration (see Orchestration)
- Deploy with PostgreSQL storage (see Storage)
