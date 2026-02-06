"""Emoji reactions on messages.

Demonstrates how to add and remove emoji reactions on messages using
RoomKit's ephemeral event system. Shows:
- publish_reaction() to add/remove reactions
- Subscribing to reaction events
- Tracking reactions per message

Run with:
    uv run python examples/reactions.py
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from roomkit import (
    EphemeralEvent,
    EphemeralEventType,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

# Track reactions: message_id -> {emoji: set of users}
reaction_tracker: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
reaction_events: list[EphemeralEvent] = []


async def main() -> None:
    kit = RoomKit()

    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    ws_charlie = WebSocketChannel("ws-charlie")
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)
    kit.register_channel(ws_charlie)

    for ch in [ws_alice, ws_bob, ws_charlie]:
        ch.register_connection(f"{ch.channel_id}-conn", lambda _c, _e: asyncio.sleep(0))

    await kit.create_room(room_id="reaction-room")
    await kit.attach_channel("reaction-room", "ws-alice")
    await kit.attach_channel("reaction-room", "ws-bob")
    await kit.attach_channel("reaction-room", "ws-charlie")

    # --- Subscribe to reaction events ---
    async def on_reaction(event: EphemeralEvent) -> None:
        if event.type != EphemeralEventType.REACTION:
            return
        reaction_events.append(event)
        target = event.data.get("target_event_id", "")
        emoji = event.data.get("emoji", "")
        action = event.data.get("action", "add")

        if action == "add":
            reaction_tracker[target][emoji].add(event.user_id)
        elif action == "remove":
            reaction_tracker[target][emoji].discard(event.user_id)

    sub_id = await kit.subscribe_room("reaction-room", on_reaction)

    # --- Send messages to react to ---
    result1 = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Just deployed the new feature!"),
        )
    )
    msg1_id = result1.event.id if result1.event else "unknown"

    result2 = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-bob",
            sender_id="bob",
            content=TextContent(body="Nice work!"),
        )
    )
    msg2_id = result2.event.id if result2.event else "unknown"

    # --- Add reactions ---
    print("Adding reactions...")
    await kit.publish_reaction("reaction-room", "bob", msg1_id, "thumbsup")
    await kit.publish_reaction("reaction-room", "charlie", msg1_id, "thumbsup")
    await kit.publish_reaction("reaction-room", "charlie", msg1_id, "tada")
    await kit.publish_reaction("reaction-room", "alice", msg2_id, "heart")
    await asyncio.sleep(0.1)

    # --- Remove a reaction ---
    print("Charlie removes thumbsup from message 1...")
    await kit.publish_reaction("reaction-room", "charlie", msg1_id, "thumbsup", action="remove")
    await asyncio.sleep(0.1)

    # --- Show reaction summary ---
    print("\n--- Reaction Summary ---")
    events = await kit.store.list_events("reaction-room")
    message_events = [e for e in events if e.type.value == "message"]

    for ev in message_events:
        body = ev.content.body if isinstance(ev.content, TextContent) else "..."
        print(f'\n  Message: "{body}" (by {ev.source.participant_id})')
        msg_reactions = reaction_tracker.get(ev.id, {})
        if msg_reactions:
            for emoji, users in msg_reactions.items():
                if users:
                    print(f"    :{emoji}: — {', '.join(sorted(users))} ({len(users)})")
        else:
            print("    (no reactions)")

    print(f"\nTotal reaction events: {len(reaction_events)}")

    # Cleanup
    await kit.unsubscribe_room(sub_id)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
