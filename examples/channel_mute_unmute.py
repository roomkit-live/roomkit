"""Channel muting and unmuting.

Demonstrates how to dynamically mute and unmute channels during a
conversation. A common pattern is muting the AI channel while a human
agent handles the conversation, then unmuting it. Shows:
- mute() / unmute() to control message delivery per channel
- ON_CHANNEL_MUTED / ON_CHANNEL_UNMUTED hooks
- Muted channels don't receive broadcast messages

Run with:
    uv run python examples/channel_mute_unmute.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
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
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider

mute_log: list[str] = []


async def main() -> None:
    kit = RoomKit()

    ws_customer = WebSocketChannel("ws-customer")
    ws_agent = WebSocketChannel("ws-agent")
    ai = AIChannel(
        "ai-bot",
        provider=MockAIProvider(
            responses=[
                "I'm the AI assistant. How can I help?",
                "Let me check that for you.",
                "I'm back! How can I help?",
            ]
        ),
    )
    kit.register_channel(ws_customer)
    kit.register_channel(ws_agent)
    kit.register_channel(ai)

    customer_inbox: list[RoomEvent] = []
    agent_inbox: list[RoomEvent] = []

    async def customer_recv(_conn: str, event: RoomEvent) -> None:
        customer_inbox.append(event)

    async def agent_recv(_conn: str, event: RoomEvent) -> None:
        agent_inbox.append(event)

    ws_customer.register_connection("customer-conn", customer_recv)
    ws_agent.register_connection("agent-conn", agent_recv)

    await kit.create_room(room_id="mute-room")
    await kit.attach_channel("mute-room", "ws-customer")
    await kit.attach_channel("mute-room", "ws-agent")
    await kit.attach_channel("mute-room", "ai-bot", category=ChannelCategory.INTELLIGENCE)

    # --- Hook: Log mute/unmute events ---
    @kit.hook(HookTrigger.ON_CHANNEL_MUTED, execution=HookExecution.ASYNC, name="mute_log")
    async def on_muted(event: RoomEvent, ctx: RoomContext) -> None:
        if isinstance(event.content, SystemContent):
            ch = event.content.data.get("channel_id", "?")
            mute_log.append(f"MUTED: {ch}")

    @kit.hook(HookTrigger.ON_CHANNEL_UNMUTED, execution=HookExecution.ASYNC, name="unmute_log")
    async def on_unmuted(event: RoomEvent, ctx: RoomContext) -> None:
        if isinstance(event.content, SystemContent):
            ch = event.content.data.get("channel_id", "?")
            mute_log.append(f"UNMUTED: {ch}")

    # --- Phase 1: AI handles customer (AI active) ---
    print("=== Phase 1: AI is active ===")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-customer",
            sender_id="customer",
            content=TextContent(body="I need help with my order"),
        )
    )
    print(f"  Customer inbox: {len(customer_inbox)} messages")

    # --- Phase 2: Human agent takes over (mute AI) ---
    print("\n=== Phase 2: Human agent takes over ===")
    await kit.mute("mute-room", "ai-bot")
    await asyncio.sleep(0.05)

    customer_inbox.clear()
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-customer",
            sender_id="customer",
            content=TextContent(body="Can you check order #12345?"),
        )
    )
    print(f"  Customer inbox: {len(customer_inbox)} messages (AI is muted, no AI reply)")

    # Agent replies manually
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-agent",
            sender_id="agent",
            content=TextContent(body="Let me look into order #12345 for you."),
        )
    )
    print(f"  Customer inbox after agent reply: {len(customer_inbox)} messages")

    # --- Phase 3: Agent done, re-enable AI ---
    print("\n=== Phase 3: AI re-enabled ===")
    await kit.unmute("mute-room", "ai-bot")
    await asyncio.sleep(0.05)

    customer_inbox.clear()
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-customer",
            sender_id="customer",
            content=TextContent(body="Thanks! One more question..."),
        )
    )
    print(f"  Customer inbox: {len(customer_inbox)} messages (AI is back)")

    # --- Show mute log ---
    await asyncio.sleep(0.1)
    print("\nMute/unmute log:")
    for entry in mute_log:
        print(f"  {entry}")

    # --- Show full conversation history ---
    events = await kit.store.list_events("mute-room")
    msg_events = [e for e in events if e.type.value == "message"]
    print(f"\nConversation ({len(msg_events)} messages):")
    for ev in msg_events:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] {ev.content.body}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
