"""Read receipts and delivery tracking.

Demonstrates how to implement read receipts using RoomKit's ephemeral
event system and the mark_read/mark_all_read store methods. Shows:
- publish_read_receipt() for ephemeral "seen" notifications
- mark_read() / mark_all_read() for persistent read state
- Subscribing to read receipt events

Run with:
    uv run python examples/read_receipts.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    EphemeralEvent,
    EphemeralEventType,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

read_receipt_log: list[dict[str, str]] = []


async def main() -> None:
    kit = RoomKit()

    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)

    alice_inbox: list[RoomEvent] = []
    bob_inbox: list[RoomEvent] = []

    async def alice_recv(_conn: str, event: RoomEvent) -> None:
        alice_inbox.append(event)

    async def bob_recv(_conn: str, event: RoomEvent) -> None:
        bob_inbox.append(event)

    ws_alice.register_connection("alice-conn", alice_recv)
    ws_bob.register_connection("bob-conn", bob_recv)

    await kit.create_room(room_id="receipt-room")
    await kit.attach_channel("receipt-room", "ws-alice")
    await kit.attach_channel("receipt-room", "ws-bob")

    # --- Subscribe to read receipt events ---
    async def on_ephemeral(event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.READ_RECEIPT:
            read_receipt_log.append(
                {
                    "user": event.user_id,
                    "event_id": event.data.get("event_id", ""),
                }
            )

    sub_id = await kit.subscribe_room("receipt-room", on_ephemeral)

    # --- Alice sends three messages ---
    print("Alice sends messages...")
    msg_ids: list[str] = []
    for text in ["Hello Bob!", "How are you?", "Let me know when you're free"]:
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-alice",
                sender_id="alice",
                content=TextContent(body=text),
            )
        )
        if result.event:
            msg_ids.append(result.event.id)
            print(f'  Sent: "{text}" (id={result.event.id[:8]}...)')

    # --- Bob reads messages one by one ---
    print("\nBob reads messages...")

    # Bob reads the first message
    await kit.publish_read_receipt("receipt-room", "bob", msg_ids[0])
    await kit.mark_read("receipt-room", "ws-bob", msg_ids[0])
    print(f"  Bob read message 1: {msg_ids[0][:8]}...")

    await asyncio.sleep(0.05)

    # Bob marks all as read
    print("  Bob marks all as read")
    await kit.mark_all_read("receipt-room", "ws-bob")
    for mid in msg_ids[1:]:
        await kit.publish_read_receipt("receipt-room", "bob", mid)

    await asyncio.sleep(0.1)

    # --- Show results ---
    print(f"\nRead receipts received ({len(read_receipt_log)}):")
    for receipt in read_receipt_log:
        print(f"  {receipt['user']} read event {receipt['event_id'][:8]}...")

    # Cleanup
    await kit.unsubscribe_room(sub_id)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
