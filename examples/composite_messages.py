"""Composite (multi-part) messages.

Demonstrates how to send messages with multiple content parts using
CompositeContent. Shows:
- Combining text + media + location in a single event
- CompositeContent structure with typed parts
- How multi-part messages flow through the pipeline

Run with:
    uv run python examples/composite_messages.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    CompositeContent,
    InboundMessage,
    LocationContent,
    MediaContent,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)


async def main() -> None:
    kit = RoomKit()

    ws_user = WebSocketChannel("ws-user")
    ws_agent = WebSocketChannel("ws-agent")
    kit.register_channel(ws_user)
    kit.register_channel(ws_agent)

    agent_inbox: list[RoomEvent] = []

    async def agent_recv(_conn: str, event: RoomEvent) -> None:
        agent_inbox.append(event)

    ws_user.register_connection("user-conn", lambda _c, _e: asyncio.sleep(0))
    ws_agent.register_connection("agent-conn", agent_recv)

    await kit.create_room(room_id="composite-room")
    await kit.attach_channel("composite-room", "ws-user")
    await kit.attach_channel("composite-room", "ws-agent")

    # --- Example 1: Text + Image ---
    print("1. Sending text + image composite...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=CompositeContent(
                parts=[
                    TextContent(body="Check out this photo from the event!"),
                    MediaContent(
                        url="https://example.com/photos/event.jpg",
                        mime_type="image/jpeg",
                        caption="Team building event 2025",
                    ),
                ]
            ),
        )
    )

    # --- Example 2: Text + Location ---
    print("2. Sending text + location composite...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=CompositeContent(
                parts=[
                    TextContent(body="Let's meet here:"),
                    LocationContent(
                        latitude=45.5017,
                        longitude=-73.5673,
                        label="Montreal Old Port",
                        address="333 Rue de la Commune O, Montreal, QC",
                    ),
                ]
            ),
        )
    )

    # --- Example 3: Text + Image + Location (triple combo) ---
    print("3. Sending text + image + location composite...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=CompositeContent(
                parts=[
                    TextContent(body="Found this amazing restaurant!"),
                    MediaContent(
                        url="https://example.com/photos/restaurant.jpg",
                        mime_type="image/jpeg",
                        caption="The food was incredible",
                    ),
                    LocationContent(
                        latitude=45.5088,
                        longitude=-73.5540,
                        label="Le Restaurant",
                        address="123 Rue Saint-Paul, Montreal, QC",
                    ),
                ]
            ),
        )
    )

    # --- Show what the agent received ---
    print(f"\nAgent received {len(agent_inbox)} composite messages:")
    for i, ev in enumerate(agent_inbox, 1):
        if isinstance(ev.content, CompositeContent):
            part_types = [p.type for p in ev.content.parts]  # type: ignore[union-attr]
            print(f"\n  Message {i}: {len(ev.content.parts)} parts ({', '.join(part_types)})")
            for j, part in enumerate(ev.content.parts, 1):
                if isinstance(part, TextContent):
                    print(f"    Part {j} [text]: {part.body}")
                elif isinstance(part, MediaContent):
                    print(f"    Part {j} [media]: {part.mime_type} - {part.caption}")
                elif isinstance(part, LocationContent):
                    print(f"    Part {j} [location]: {part.label} ({part.latitude}, {part.longitude})")


if __name__ == "__main__":
    asyncio.run(main())
