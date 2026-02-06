"""Location sharing between channels.

Demonstrates how to send and receive geographic locations using
LocationContent. Shows:
- Sending latitude/longitude with optional label and address
- Location content flowing between channels
- How locations can be used in multi-channel scenarios

Run with:
    uv run python examples/location_sharing.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    InboundMessage,
    LocationContent,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)


async def main() -> None:
    kit = RoomKit()

    ws_customer = WebSocketChannel("ws-customer")
    ws_agent = WebSocketChannel("ws-agent")
    kit.register_channel(ws_customer)
    kit.register_channel(ws_agent)

    agent_inbox: list[RoomEvent] = []
    customer_inbox: list[RoomEvent] = []

    async def agent_recv(_conn: str, event: RoomEvent) -> None:
        agent_inbox.append(event)

    async def customer_recv(_conn: str, event: RoomEvent) -> None:
        customer_inbox.append(event)

    ws_customer.register_connection("customer-conn", customer_recv)
    ws_agent.register_connection("agent-conn", agent_recv)

    await kit.create_room(room_id="location-room")
    await kit.attach_channel("location-room", "ws-customer")
    await kit.attach_channel("location-room", "ws-agent")

    # --- Customer asks for directions ---
    print("Customer asks for nearest branch...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-customer",
            sender_id="customer",
            content=TextContent(body="Where is your nearest branch?"),
        )
    )

    # --- Agent shares a location ---
    print("Agent shares branch location...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-agent",
            sender_id="agent",
            content=LocationContent(
                latitude=45.5017,
                longitude=-73.5673,
                label="Montreal Downtown Branch",
                address="1000 Rue Sherbrooke O, Montreal, QC H3A 3G4",
            ),
        )
    )

    # --- Customer shares their current location ---
    print("Customer shares their location...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-customer",
            sender_id="customer",
            content=LocationContent(
                latitude=45.5088,
                longitude=-73.5878,
                label="My Location",
            ),
        )
    )

    # --- Show what each participant received ---
    print(f"\nAgent inbox ({len(agent_inbox)} messages):")
    for ev in agent_inbox:
        if isinstance(ev.content, TextContent):
            print(f"  [text] {ev.content.body}")
        elif isinstance(ev.content, LocationContent):
            print(
                f"  [location] {ev.content.label or 'Unnamed'} "
                f"({ev.content.latitude:.4f}, {ev.content.longitude:.4f})"
            )
            if ev.content.address:
                print(f"             {ev.content.address}")

    print(f"\nCustomer inbox ({len(customer_inbox)} messages):")
    for ev in customer_inbox:
        if isinstance(ev.content, TextContent):
            print(f"  [text] {ev.content.body}")
        elif isinstance(ev.content, LocationContent):
            print(
                f"  [location] {ev.content.label or 'Unnamed'} "
                f"({ev.content.latitude:.4f}, {ev.content.longitude:.4f})"
            )
            if ev.content.address:
                print(f"             {ev.content.address}")

    # --- Show stored history ---
    events = await kit.store.list_events("location-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] text: {ev.content.body}")
        elif isinstance(ev.content, LocationContent):
            print(f"  [{ev.source.channel_id}] location: {ev.content.label}")


if __name__ == "__main__":
    asyncio.run(main())
