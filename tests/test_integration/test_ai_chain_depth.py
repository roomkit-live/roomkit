"""Integration: AI-to-AI recursion with depth limit."""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider


class TestAIChainDepth:
    async def test_ai_response_stored(self) -> None:
        """AI response events are properly stored."""
        kit = RoomKit(max_chain_depth=3)
        ai_provider = MockAIProvider(responses=["AI reply"])
        ws = WebSocketChannel("ws1")
        ai = AIChannel("ai1", provider=ai_provider)

        kit.register_channel(ws)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Hello AI"),
        )
        await kit.process_inbound(msg)

        events = await kit.store.list_events("r1")
        assert len(events) >= 2
        ai_events = [e for e in events if e.source.channel_id == "ai1"]
        assert len(ai_events) >= 1

    async def test_two_ai_channels_respond(self) -> None:
        """Two AI channels both respond to a human message."""
        kit = RoomKit(max_chain_depth=3)
        provider1 = MockAIProvider(responses=["AI-1 reply"])
        provider2 = MockAIProvider(responses=["AI-2 reply"])
        ws = WebSocketChannel("ws1")
        ai1 = AIChannel("ai1", provider=provider1)
        ai2 = AIChannel("ai2", provider=provider2)

        kit.register_channel(ws)
        kit.register_channel(ai1)
        kit.register_channel(ai2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        await kit.attach_channel("r1", "ai2", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Hello both AIs"),
        )
        await kit.process_inbound(msg)

        assert len(provider1.calls) >= 1
        assert len(provider2.calls) >= 1

    async def test_chain_depth_limits_recursion(self) -> None:
        """Chain depth prevents infinite AI-to-AI recursion."""
        kit = RoomKit(max_chain_depth=2)
        provider = MockAIProvider(responses=["AI response"])
        ws = WebSocketChannel("ws1")
        ai = AIChannel("ai1", provider=provider)

        kit.register_channel(ws)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Start chain"),
        )
        await kit.process_inbound(msg)

        # AI should be called at least once, but chain depth limits recursion
        assert len(provider.calls) >= 1
        events = await kit.store.list_events("r1")
        # Total events should be bounded by chain depth
        assert len(events) < 20  # Sanity check

    async def test_depth_zero_allows_single_response(self) -> None:
        """Chain depth 0 event triggers AI, but AI response at depth 1 still works within limit."""
        kit = RoomKit(max_chain_depth=5)
        provider = MockAIProvider(responses=["reply"])
        ws = WebSocketChannel("ws1")
        ai = AIChannel("ai1", provider=provider)

        kit.register_channel(ws)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(provider.calls) >= 1
