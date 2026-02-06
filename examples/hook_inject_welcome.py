"""Auto-inject welcome messages with hooks.

Demonstrates how to use ON_ROOM_CREATED and ON_CHANNEL_ATTACHED hooks
to automatically send welcome/system messages when participants join.
Shows:
- Lifecycle hooks (ON_ROOM_CREATED, ON_CHANNEL_ATTACHED)
- InjectedEvent for side-effect messages
- SystemContent for system notifications

Run with:
    uv run python examples/hook_inject_welcome.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    SystemContent,
    TextContent,
    WebSocketChannel,
)


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

    # --- Hook: Welcome on channel attach ---
    @kit.hook(
        HookTrigger.ON_CHANNEL_ATTACHED,
        execution=HookExecution.ASYNC,
        name="welcome_message",
    )
    async def welcome_on_attach(event: RoomEvent, ctx: RoomContext) -> None:
        """Send a welcome message when a new channel is attached."""
        if isinstance(event.content, SystemContent) and event.content.code == "channel_attached":
            channel_id = event.content.data.get("channel_id", "unknown")
            print(f"  [hook] Channel '{channel_id}' attached — sending welcome")

    # --- Hook: Log room creation ---
    @kit.hook(
        HookTrigger.ON_ROOM_CREATED,
        execution=HookExecution.ASYNC,
        name="room_created_logger",
    )
    async def room_created_logger(event: RoomEvent, ctx: RoomContext) -> None:
        if isinstance(event.content, SystemContent):
            print(f"  [hook] Room created: {event.content.data.get('room_id')}")

    # --- Create room and attach channels ---
    print("Creating room...")
    await kit.create_room(room_id="welcome-room")

    # Give async hooks time to fire
    await asyncio.sleep(0.05)

    print("\nAttaching Alice...")
    await kit.attach_channel("welcome-room", "ws-alice")
    await asyncio.sleep(0.05)

    print("Attaching Bob...")
    await kit.attach_channel("welcome-room", "ws-bob")
    await asyncio.sleep(0.05)

    # --- Alice sends a message ---
    print("\nAlice sends a greeting...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Hey everyone, I just joined!"),
        )
    )

    # --- Show results ---
    print(f"\nAlice's inbox ({len(alice_inbox)} messages):")
    for ev in alice_inbox:
        body = getattr(ev.content, "body", str(ev.content))
        print(f"  <- [{ev.source.channel_id}] {body}")

    print(f"\nBob's inbox ({len(bob_inbox)} messages):")
    for ev in bob_inbox:
        body = getattr(ev.content, "body", str(ev.content))
        print(f"  <- [{ev.source.channel_id}] {body}")

    # --- Show stored history ---
    events = await kit.store.list_events("welcome-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        body = getattr(ev.content, "body", str(ev.content))
        print(f"  [{ev.type.value:>18}] {body}")


if __name__ == "__main__":
    asyncio.run(main())
