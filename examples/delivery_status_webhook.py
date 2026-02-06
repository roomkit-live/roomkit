"""Delivery status tracking with webhooks.

Demonstrates how to track outbound message delivery using the
on_delivery_status decorator. Shows:
- Registering delivery status handlers
- Processing DeliveryStatus from provider webhooks
- Tracking sent/delivered/failed states

Run with:
    uv run python examples/delivery_status_webhook.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    DeliveryStatus,
    RoomKit,
    WebSocketChannel,
)

# Track delivery states
delivery_log: list[dict[str, str]] = []


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-agent")
    kit.register_channel(ws)
    ws.register_connection("agent-conn", lambda _c, _e: asyncio.sleep(0))

    await kit.create_room(room_id="delivery-room")
    await kit.attach_channel("delivery-room", "ws-agent")

    # --- Register delivery status handler ---
    @kit.on_delivery_status
    async def track_delivery(status: DeliveryStatus) -> None:
        delivery_log.append(
            {
                "provider": status.provider,
                "message_id": status.message_id,
                "status": status.status,
                "recipient": status.recipient,
                "error": status.error_message or "",
            }
        )
        if status.status == "failed":
            print(f"  [ALERT] Message {status.message_id} FAILED: {status.error_message}")

    # --- Simulate delivery status webhooks from an SMS provider ---
    print("Simulating delivery status webhooks...\n")

    # Message 1: Sent -> Delivered -> Read
    statuses_msg1 = [
        DeliveryStatus(
            provider="twilio",
            message_id="SM001",
            status="sent",
            recipient="+15551234567",
            sender="+15559876543",
        ),
        DeliveryStatus(
            provider="twilio",
            message_id="SM001",
            status="delivered",
            recipient="+15551234567",
            sender="+15559876543",
        ),
    ]

    # Message 2: Sent -> Failed
    statuses_msg2 = [
        DeliveryStatus(
            provider="twilio",
            message_id="SM002",
            status="sent",
            recipient="+15550000000",
            sender="+15559876543",
        ),
        DeliveryStatus(
            provider="twilio",
            message_id="SM002",
            status="failed",
            recipient="+15550000000",
            sender="+15559876543",
            error_code="30003",
            error_message="Unreachable destination handset",
        ),
    ]

    for status in statuses_msg1 + statuses_msg2:
        await kit.process_delivery_status(status)

    # --- Show results ---
    print(f"\nDelivery log ({len(delivery_log)} entries):")
    print(f"  {'Message ID':<12} {'Status':<12} {'Recipient':<16} {'Error'}")
    print(f"  {'-' * 60}")
    for entry in delivery_log:
        print(
            f"  {entry['message_id']:<12} {entry['status']:<12} "
            f"{entry['recipient']:<16} {entry['error']}"
        )

    # Summary
    statuses = [e["status"] for e in delivery_log]
    print(
        f"\nSummary: {statuses.count('sent')} sent, "
        f"{statuses.count('delivered')} delivered, "
        f"{statuses.count('failed')} failed"
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
