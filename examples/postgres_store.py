"""PostgreSQL storage backend.

Demonstrates how to set up PostgresStore for production persistence
instead of the default InMemoryStore. Shows:
- Configuring PostgresStore with a connection URL
- Room, event, and participant CRUD operations
- Querying conversation history with pagination
- Falling back to InMemoryStore when Postgres is unavailable

Run with:
    DATABASE_URL=postgresql://user:pass@localhost/roomkit uv run python examples/postgres_store.py

Without DATABASE_URL, this example falls back to InMemoryStore as a demo.
"""

from __future__ import annotations

import asyncio
import os

from roomkit import (
    InboundMessage,
    InMemoryStore,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)


async def main() -> None:
    database_url = os.environ.get("DATABASE_URL", "")

    if database_url:
        # Production: use PostgreSQL
        from roomkit import PostgresStore

        store = PostgresStore(database_url)
        await store.initialize()  # Creates tables if needed
        print(f"Using PostgresStore ({database_url[:30]}...)")
    else:
        # Fallback: in-memory (for demo purposes)
        store = InMemoryStore()
        print("Using InMemoryStore (set DATABASE_URL for PostgreSQL)")

    # --- RoomKit with custom store ---
    kit = RoomKit(store=store)

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    # --- Create room and send messages ---
    room = await kit.create_room(room_id="persistent-room", metadata={"topic": "Support"})
    print(f"\nRoom created: {room.id} (status={room.status})")

    await kit.attach_channel("persistent-room", "ws-user")

    # Send several messages
    messages = [
        "Hello, I need help with my account",
        "I can't log in since yesterday",
        "I've tried resetting my password",
        "The reset email never arrives",
        "Can someone help me?",
    ]

    for text in messages:
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user",
                content=TextContent(body=text),
            )
        )

    print(f"Sent {len(messages)} messages")

    # --- Query with pagination ---
    print("\n--- Paginated History ---")

    # Page 1: first 3 events
    page1 = await kit.store.list_events("persistent-room", offset=0, limit=3)
    print(f"\nPage 1 ({len(page1)} events):")
    for ev in page1:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] {ev.content.body}")

    # Page 2: next 3 events
    page2 = await kit.store.list_events("persistent-room", offset=3, limit=3)
    print(f"\nPage 2 ({len(page2)} events):")
    for ev in page2:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] {ev.content.body}")

    # --- Get timeline via convenience method ---
    print("\n--- Full Timeline ---")
    timeline = await kit.get_timeline("persistent-room", offset=0, limit=50)
    msg_events = [e for e in timeline if e.type.value == "message"]
    print(f"Total messages: {len(msg_events)}")

    # --- Room metadata ---
    room = await kit.get_room("persistent-room")
    print(f"\nRoom metadata: {room.metadata}")
    print(f"Room event_count: {room.event_count}")

    # --- Participants ---
    participants = await kit.store.list_participants("persistent-room")
    print(f"\nParticipants ({len(participants)}):")
    for p in participants:
        print(f"  {p.id}: role={p.role}, status={p.status}")

    # --- Bindings ---
    bindings = await kit.store.list_bindings("persistent-room")
    print(f"\nBindings ({len(bindings)}):")
    for b in bindings:
        print(f"  {b.channel_id}: type={b.channel_type}, muted={b.muted}")

    # --- Cleanup ---
    if database_url:
        print("\nData is persisted in PostgreSQL and will survive restarts.")
    else:
        print("\nData is in-memory only (will be lost when process exits).")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
