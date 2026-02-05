"""RoomKit quickstart — two WebSocket users chatting with an AI assistant.

Run with:
    uv run python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    HookResult,
    HookTrigger,
    InboundMessage,
    MockAIProvider,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel


async def main() -> None:
    # --- Setup -----------------------------------------------------------
    kit = RoomKit()

    # Two WebSocket channels (one per user) and one AI channel.
    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    ai = AIChannel("ai-assistant", provider=MockAIProvider(responses=["Got it!"]))

    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)
    kit.register_channel(ai)

    # Wire up fake "send" callbacks so we can see what each user receives.
    alice_inbox: list[RoomEvent] = []
    bob_inbox: list[RoomEvent] = []

    async def alice_recv(_conn: str, event: RoomEvent) -> None:
        alice_inbox.append(event)

    async def bob_recv(_conn: str, event: RoomEvent) -> None:
        bob_inbox.append(event)

    ws_alice.register_connection("alice-conn", alice_recv)
    ws_bob.register_connection("bob-conn", bob_recv)

    # Create a room and attach all three channels.
    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-alice")
    await kit.attach_channel("demo-room", "ws-bob")
    await kit.attach_channel("demo-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    # --- Hook: simple profanity filter -----------------------------------
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="profanity_filter")
    async def profanity_filter(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent) and "badword" in event.content.body:
            return HookResult.block("Message blocked by profanity filter")
        return HookResult.allow()

    # --- Send messages ---------------------------------------------------
    # Alice says hello.
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Hello everyone!"),
        )
    )
    print(f"Alice sent 'Hello everyone!' -> blocked={result.blocked}")

    # Bob replies.
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-bob",
            sender_id="bob",
            content=TextContent(body="Hey Alice!"),
        )
    )
    print(f"Bob sent 'Hey Alice!' -> blocked={result.blocked}")

    # Alice tries to send a bad message.
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="This has a badword in it"),
        )
    )
    print(f"Alice sent profanity -> blocked={result.blocked}, reason={result.reason}")

    # --- Inspect results -------------------------------------------------
    print(f"\nAlice's inbox ({len(alice_inbox)} messages):")
    for ev in alice_inbox:
        print(f"  <- {ev.source.channel_id}: {ev.content.body}")  # type: ignore[union-attr]

    print(f"\nBob's inbox ({len(bob_inbox)} messages):")
    for ev in bob_inbox:
        print(f"  <- {ev.source.channel_id}: {ev.content.body}")  # type: ignore[union-attr]

    # Show stored conversation history.
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
