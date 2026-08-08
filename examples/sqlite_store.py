"""Embedded SQLite storage backend.

Demonstrates SQLiteStore, the single-file persistent ConversationStore for
deployments where PostgresStore is a burden and InMemoryStore forgets
everything on exit (desktop apps, edge boxes, small bots). Shows:
- Opening a store on a .db file (the schema is created on first use)
- Running a conversation through RoomKit with it
- History surviving a full process-level restart
- Full-text recall over stored messages via search_events() (FTS5)

Run with:
    uv run python examples/sqlite_store.py

Set ROOMKIT_DB to choose the file (default: roomkit-example.db in the
current directory). The file is removed at startup so each run tells the
same story, and left behind at the end for inspection.

RoomKit logs a warning about pairing SQLiteStore with the default
InMemoryLockManager. That is expected here: the pairing is safe in a single
process, which is exactly what this example is. Only a multi-process
deployment needs a distributed lock manager (and PostgresStore with it).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from roomkit import (
    InboundMessage,
    RoomEvent,
    RoomKit,
    SQLiteStore,
    TextContent,
    WebSocketChannel,
)

ROOM_ID = "support-1"

CONVERSATION = [
    "Hello, my invoice for March is wrong",
    "I was charged twice for the same subscription",
    "Can I get a refund for the duplicate charge?",
    "I also can't log in since yesterday",
    "The password reset email never arrives",
]


async def first_run(db_path: str) -> None:
    """Open the store, hold a conversation, close everything down."""
    store = SQLiteStore(db_path)
    kit = RoomKit(store=store)

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv, room_id=ROOM_ID)

    room = await kit.create_room(room_id=ROOM_ID, metadata={"topic": "Billing"})
    print(f"Room created: {room.id} (status={room.status})")

    await kit.attach_channel(ROOM_ID, "ws-user")

    for text in CONVERSATION:
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user",
                content=TextContent(body=text),
            )
        )
    print(f"Sent {len(CONVERSATION)} messages")

    room = await kit.get_room(ROOM_ID)
    print(f"Room event_count: {room.event_count}, latest_index: {room.latest_index}")

    # Closing the kit closes the store, which closes the database file.
    await kit.close()
    print(f"Store closed. Everything is on disk in {db_path}")


async def after_restart(db_path: str) -> None:
    """Reopen the same file from scratch — nothing is carried over in memory."""
    store = SQLiteStore(db_path)

    room = await store.get_room(ROOM_ID)
    if room is None:  # pragma: no cover - only if the file was tampered with
        print("Room not found — did the file survive?")
        await store.close()
        return

    print(f"Room recovered: {room.id} (topic={room.metadata.get('topic')})")
    print(f"Room event_count: {room.event_count}, latest_index: {room.latest_index}")

    events = await store.list_events(ROOM_ID, offset=0, limit=50)
    messages = [e for e in events if isinstance(e.content, TextContent)]
    print(f"\n--- Recovered timeline ({len(messages)} messages) ---")
    for ev in messages:
        print(f"  [{ev.index}] {ev.content.body}")

    # --- Full-text search (an SQLite extra, not part of the store contract) ---
    # Terms are ANDed, so this matches only the message carrying both words.
    print("\n--- search_events('duplicate charge') ---")
    for ev in await store.search_events("duplicate charge", room_id=ROOM_ID):
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.index}] {ev.content.body}")

    print("\n--- search_events('password') ---")
    for ev in await store.search_events("password", room_id=ROOM_ID):
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.index}] {ev.content.body}")

    await store.close()


async def main() -> None:
    db_path = os.environ.get("ROOMKIT_DB", "roomkit-example.db")

    # Start from a clean file so the example is reproducible. SQLite keeps
    # its WAL and shared-memory files alongside the database.
    for suffix in ("", "-wal", "-shm"):
        Path(db_path + suffix).unlink(missing_ok=True)

    print("=== First run ===")
    await first_run(db_path)

    print("\n=== After restart (new store, new process state) ===")
    await after_restart(db_path)

    print(f"\nThe database is still at {db_path} — run this again to see it recreated.")


if __name__ == "__main__":
    asyncio.run(main())
