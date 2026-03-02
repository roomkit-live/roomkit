"""Response visibility — control where AI responses are delivered.

Demonstrates using ``response_visibility`` to route an AI response to a
specific channel (WebSocket) without triggering delivery on other channels.
This is useful in hybrid voice+text setups where typed text should produce
a text-only reply.

The example sets up three transport channels and one AI channel in a room.
A BEFORE_BROADCAST hook stamps ``response_visibility`` on the trigger event
so the AI's response is delivered only to the targeted WebSocket channel.

Run with:
    uv run python examples/response_visibility.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    HookResult,
    HookTrigger,
    InboundMessage,
    MockAIProvider,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.context import RoomContext


async def main() -> None:
    kit = RoomKit()

    # --- Channels ------------------------------------------------------------

    # Source channel — simulates a text input widget in a voice+text UI
    source = WebSocketChannel("text-input")

    # Target channel — where we want the AI response delivered
    ws = WebSocketChannel("ws-ui")
    ws_inbox: list[RoomEvent] = []

    async def on_ws_receive(_conn: str, event: RoomEvent) -> None:
        ws_inbox.append(event)

    ws.register_connection("browser", on_ws_receive)

    # Another transport — should NOT receive the AI response
    other = WebSocketChannel("voice-out")
    other_inbox: list[RoomEvent] = []

    async def on_other_receive(_conn: str, event: RoomEvent) -> None:
        other_inbox.append(event)

    other.register_connection("speaker", on_other_receive)

    # AI channel
    ai_provider = MockAIProvider(responses=["The answer is 42."])
    ai = AIChannel("ai", provider=ai_provider, system_prompt="Be concise.")

    # --- Room setup ----------------------------------------------------------

    kit.register_channel(source)
    kit.register_channel(ws)
    kit.register_channel(other)
    kit.register_channel(ai)

    await kit.create_room(room_id="hybrid-room")
    await kit.attach_channel("hybrid-room", "text-input")
    await kit.attach_channel("hybrid-room", "ws-ui")
    await kit.attach_channel("hybrid-room", "voice-out")
    await kit.attach_channel("hybrid-room", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hook to stamp response_visibility -----------------------------------
    # In a real app, this logic would inspect metadata (e.g. a "reply_to"
    # header) to decide where the response should go.

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def stamp_response_visibility(event: RoomEvent, context: RoomContext) -> HookResult:
        if event.source.channel_id == "text-input":
            return HookResult(
                action="modify",
                event=event.model_copy(
                    update={
                        "visibility": "ai",
                        "response_visibility": "ws-ui",
                    }
                ),
            )
        return HookResult(action="allow")

    # --- Send inbound message ------------------------------------------------

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="text-input",
            sender_id="user-1",
            content=TextContent(body="What is the meaning of life?"),
        )
    )
    print(f"Inbound processed: blocked={result.blocked}")

    # --- Verify delivery scope -----------------------------------------------

    ws_ai = [e for e in ws_inbox if e.source.channel_id == "ai"]
    other_ai = [e for e in other_inbox if e.source.channel_id == "ai"]

    print(f"\nws-ui received {len(ws_ai)} AI response(s)")
    for ev in ws_ai:
        body = ev.content.body if isinstance(ev.content, TextContent) else "?"
        print(f"  -> {body}")

    print(f"voice-out received {len(other_ai)} AI response(s)  (expected: 0)")

    # --- Conversation store --------------------------------------------------

    events = await kit.get_timeline("hybrid-room")
    print(f"\nTimeline ({len(events)} events):")
    for ev in events:
        body = getattr(ev.content, "body", "?")
        print(f"  [{ev.source.channel_id}] vis={ev.visibility!r} -> {body}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
