"""Smoke test that mirrors the README quickstart exactly.

If this test breaks, the README quickstart is lying to new users.
Only imports/APIs from the base ``pip install roomkit`` are used here —
no dev extras allowed.
"""

from __future__ import annotations

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


class TestReadmeQuickstart:
    """Mirrors the README quickstart snippet step-by-step."""

    async def test_quickstart_flow(self) -> None:
        kit = RoomKit()

        # Register channels
        ws = WebSocketChannel("ws-user")
        ai = AIChannel("ai-bot", provider=MockAIProvider(responses=["Hello!"]))
        kit.register_channel(ws)
        kit.register_channel(ai)

        # Create a room and attach channels
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "ws-user")
        await kit.attach_channel("room-1", "ai-bot", category=ChannelCategory.INTELLIGENCE)

        # Add a broadcast hook
        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="filter")
        async def block_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and "spam" in event.content.body:
                return HookResult.block("spam detected")
            return HookResult.allow()

        # Process a message — it gets stored, broadcast, and the AI responds
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user", sender_id="user-1", content=TextContent(body="Hi")
            )
        )
        assert not result.blocked

        # View conversation history
        events = await kit.store.list_events("room-1")
        assert len(events) >= 2  # user message + AI response

    async def test_spam_blocked(self) -> None:
        """Hook blocks a spam message — mirrors the hook demo in the README."""
        kit = RoomKit()

        ws = WebSocketChannel("ws-user")
        kit.register_channel(ws)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "ws-user")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="filter")
        async def block_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and "spam" in event.content.body:
                return HookResult.block("spam detected")
            return HookResult.allow()

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user-1",
                content=TextContent(body="buy cheap spam now"),
            )
        )
        assert result.blocked
