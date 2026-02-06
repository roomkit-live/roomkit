"""Typing indicators with the realtime backend.

Demonstrates how to publish and subscribe to typing start/stop events
using RoomKit's ephemeral event system. Shows:
- publish_typing() for typing start/stop
- subscribe_room() to listen for ephemeral events
- InMemoryRealtime backend (default)
- Typing events are ephemeral — they're NOT stored in conversation history

Run with:
    uv run python examples/typing_indicators.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    EphemeralEvent,
    EphemeralEventType,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

# Collect typing events received by each user
alice_typing_events: list[EphemeralEvent] = []
bob_typing_events: list[EphemeralEvent] = []


async def main() -> None:
    kit = RoomKit()

    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)

    # Wire up message delivery
    ws_alice.register_connection("alice-conn", lambda _c, _e: asyncio.sleep(0))
    ws_bob.register_connection("bob-conn", lambda _c, _e: asyncio.sleep(0))

    await kit.create_room(room_id="typing-room")
    await kit.attach_channel("typing-room", "ws-alice")
    await kit.attach_channel("typing-room", "ws-bob")

    # --- Subscribe both users to ephemeral events ---
    async def alice_ephemeral(event: EphemeralEvent) -> None:
        alice_typing_events.append(event)

    async def bob_ephemeral(event: EphemeralEvent) -> None:
        bob_typing_events.append(event)

    alice_sub = await kit.subscribe_room("typing-room", alice_ephemeral)
    bob_sub = await kit.subscribe_room("typing-room", bob_ephemeral)

    # --- Simulate typing flow ---
    # Alice starts typing
    print("Alice starts typing...")
    await kit.publish_typing("typing-room", "alice", is_typing=True, data={"name": "Alice"})
    await asyncio.sleep(0.05)

    # Alice stops typing and sends a message
    print("Alice stops typing and sends a message...")
    await kit.publish_typing("typing-room", "alice", is_typing=False)
    await asyncio.sleep(0.05)

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Hello Bob!"),
        )
    )

    # Bob starts typing
    print("Bob starts typing...")
    await kit.publish_typing("typing-room", "bob", is_typing=True, data={"name": "Bob"})
    await asyncio.sleep(0.05)

    # Bob sends a message
    print("Bob stops typing and sends a message...")
    await kit.publish_typing("typing-room", "bob", is_typing=False)
    await asyncio.sleep(0.05)

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-bob",
            sender_id="bob",
            content=TextContent(body="Hey Alice!"),
        )
    )

    await asyncio.sleep(0.1)

    # --- Show results ---
    print(f"\nAlice received {len(alice_typing_events)} typing events:")
    for ev in alice_typing_events:
        status = "typing..." if ev.type == EphemeralEventType.TYPING_START else "stopped"
        name = ev.data.get("name", ev.user_id)
        print(f"  {name}: {status}")

    print(f"\nBob received {len(bob_typing_events)} typing events:")
    for ev in bob_typing_events:
        status = "typing..." if ev.type == EphemeralEventType.TYPING_START else "stopped"
        name = ev.data.get("name", ev.user_id)
        print(f"  {name}: {status}")

    # Verify ephemeral events are NOT in conversation history
    events = await kit.store.list_events("typing-room")
    message_events = [e for e in events if e.type.value == "message"]
    print(f"\nStored messages: {len(message_events)} (typing events are ephemeral, not stored)")

    # Cleanup
    await kit.unsubscribe_room(alice_sub)
    await kit.unsubscribe_room(bob_sub)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
