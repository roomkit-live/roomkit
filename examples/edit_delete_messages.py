"""Message editing and deletion.

Demonstrates how to edit and delete previously sent messages using
EditContent and DeleteContent. Shows:
- Editing a message with EditContent (references target_event_id)
- Deleting a message with DeleteContent
- Delete types: SENDER, SYSTEM, ADMIN
- How edits/deletes are stored in conversation history

Run with:
    uv run python examples/edit_delete_messages.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    DeleteContent,
    DeleteType,
    EditContent,
    EventType,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

async def main() -> None:
    kit = RoomKit()

    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)

    bob_inbox: list[RoomEvent] = []

    async def bob_recv(_conn: str, event: RoomEvent) -> None:
        bob_inbox.append(event)

    ws_alice.register_connection("alice-conn", lambda _c, _e: asyncio.sleep(0))
    ws_bob.register_connection("bob-conn", bob_recv)

    await kit.create_room(room_id="edit-room")
    await kit.attach_channel("edit-room", "ws-alice")
    await kit.attach_channel("edit-room", "ws-bob")

    # --- Alice sends a message ---
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Hello Bbo!"),  # Typo on purpose
        )
    )
    original_id = result.event.id if result.event else ""
    print(f"1. Alice sent: \"Hello Bbo!\" (id={original_id[:8]}...)")

    # --- Alice edits the message to fix the typo ---
    edit_result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            event_type=EventType.EDIT,
            content=EditContent(
                target_event_id=original_id,
                new_content=TextContent(body="Hello Bob!"),
                edit_source="alice",
            ),
        )
    )
    print(f"2. Alice edited message -> \"Hello Bob!\"")

    # --- Alice sends another message ---
    result2 = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Actually, never mind about that."),
        )
    )
    msg2_id = result2.event.id if result2.event else ""
    print(f"3. Alice sent: \"Actually, never mind about that.\" (id={msg2_id[:8]}...)")

    # --- Alice deletes the second message ---
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            event_type=EventType.DELETE,
            content=DeleteContent(
                target_event_id=msg2_id,
                delete_type=DeleteType.SENDER,
                reason="Changed my mind",
            ),
        )
    )
    print(f"4. Alice deleted message: \"{msg2_id[:8]}...\" (reason: Changed my mind)")

    # --- Show what Bob received ---
    print(f"\nBob's inbox ({len(bob_inbox)} events):")
    for ev in bob_inbox:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.type.value:>8}] {ev.content.body}")
        elif isinstance(ev.content, EditContent):
            new_body = ev.content.new_content.body if isinstance(ev.content.new_content, TextContent) else "..."
            print(f"  [{ev.type.value:>8}] Edit -> \"{new_body}\" (target={ev.content.target_event_id[:8]}...)")
        elif isinstance(ev.content, DeleteContent):
            print(f"  [{ev.type.value:>8}] Delete target={ev.content.target_event_id[:8]}... ({ev.content.delete_type})")

    # --- Show full conversation history ---
    events = await kit.store.list_events("edit-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        content_desc = ""
        if isinstance(ev.content, TextContent):
            content_desc = ev.content.body
        elif isinstance(ev.content, EditContent):
            content_desc = f"Edit target={ev.content.target_event_id[:8]}..."
        elif isinstance(ev.content, DeleteContent):
            content_desc = f"Delete target={ev.content.target_event_id[:8]}..."
        else:
            content_desc = str(ev.content.type)  # type: ignore[union-attr]
        print(f"  [{ev.type.value:>18}] {content_desc}")


if __name__ == "__main__":
    asyncio.run(main())
