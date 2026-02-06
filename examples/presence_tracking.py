"""Presence tracking — online/away/offline status.

Demonstrates how to track user presence using RoomKit's ephemeral event
system. Shows:
- publish_presence() with "online", "away", "offline" statuses
- Building a "who's online" presence map from events
- Presence events are ephemeral (not persisted)

Run with:
    uv run python examples/presence_tracking.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    EphemeralEvent,
    EphemeralEventType,
    RoomKit,
    WebSocketChannel,
)

# Track current presence status per user
presence_map: dict[str, str] = {}
presence_events: list[EphemeralEvent] = []


async def main() -> None:
    kit = RoomKit()

    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    ws_charlie = WebSocketChannel("ws-charlie")
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)
    kit.register_channel(ws_charlie)

    # Register dummy connections
    for ch in [ws_alice, ws_bob, ws_charlie]:
        ch.register_connection(f"{ch.channel_id}-conn", lambda _c, _e: asyncio.sleep(0))

    await kit.create_room(room_id="presence-room")
    await kit.attach_channel("presence-room", "ws-alice")
    await kit.attach_channel("presence-room", "ws-bob")
    await kit.attach_channel("presence-room", "ws-charlie")

    # --- Subscribe to presence events ---
    async def on_presence(event: EphemeralEvent) -> None:
        presence_events.append(event)
        # Update presence map
        status_map = {
            EphemeralEventType.PRESENCE_ONLINE: "online",
            EphemeralEventType.PRESENCE_AWAY: "away",
            EphemeralEventType.PRESENCE_OFFLINE: "offline",
        }
        status = status_map.get(event.type, "unknown")
        presence_map[event.user_id] = status
        print(f"  [{event.user_id}] -> {status}")

    sub_id = await kit.subscribe_room("presence-room", on_presence)

    # --- Simulate presence changes ---
    print("Users come online:")
    await kit.publish_presence("presence-room", "alice", "online")
    await kit.publish_presence("presence-room", "bob", "online")
    await kit.publish_presence("presence-room", "charlie", "online")
    await asyncio.sleep(0.1)

    print("\nAlice goes away:")
    await kit.publish_presence("presence-room", "alice", "away")
    await asyncio.sleep(0.05)

    print("\nBob goes offline:")
    await kit.publish_presence("presence-room", "bob", "offline")
    await asyncio.sleep(0.05)

    print("\nAlice comes back online:")
    await kit.publish_presence("presence-room", "alice", "online")
    await asyncio.sleep(0.1)

    # --- Show current presence map ---
    print("\n--- Current Presence ---")
    for user, status in sorted(presence_map.items()):
        indicator = {"online": "(+)", "away": "(!)", "offline": "(-)"}
        print(f"  {indicator.get(status, '(?)')} {user}: {status}")

    print(f"\nTotal presence events received: {len(presence_events)}")

    # Cleanup
    await kit.unsubscribe_room(sub_id)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
