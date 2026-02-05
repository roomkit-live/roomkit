"""Integration: Human+Human+AI assistant with visibility control."""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.mock import MockAIProvider


class TestAIAssistant:
    async def test_ai_visible_to_all(self) -> None:
        """AI receives messages from both humans when visible to all."""
        kit = RoomKit()
        ai_provider = MockAIProvider(responses=["AI sees all"])
        ws1 = WebSocketChannel("ws1")
        ws2 = WebSocketChannel("ws2")
        ai = AIChannel("ai1", provider=ai_provider)

        kit.register_channel(ws1)
        kit.register_channel(ws2)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ws2")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        for ch_id in ["ws1", "ws2"]:
            msg = InboundMessage(
                channel_id=ch_id,
                sender_id=f"user_{ch_id}",
                content=TextContent(body=f"Message from {ch_id}"),
            )
            await kit.process_inbound(msg)

        assert len(ai_provider.calls) == 2

    async def test_ai_invisible_to_transport(self) -> None:
        """AI with transport-only visibility is not seen by other transport channels."""
        kit = RoomKit()
        ai_provider = MockAIProvider(responses=["AI whispers"])
        ws1 = WebSocketChannel("ws1")
        ws2 = WebSocketChannel("ws2")
        ai = AIChannel("ai1", provider=ai_provider)
        ws2_received: list[RoomEvent] = []

        async def ws2_send(conn_id: str, event: RoomEvent) -> None:
            ws2_received.append(event)

        ws2.register_connection("conn2", ws2_send)

        kit.register_channel(ws1)
        kit.register_channel(ws2)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ws2")
        await kit.attach_channel(
            "r1",
            "ai1",
            category=ChannelCategory.INTELLIGENCE,
            visibility="intelligence",
        )

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Hello team"),
        )
        await kit.process_inbound(msg)
        # ws2 should receive the human message but not the AI response
        # because AI visibility is "intelligence" (invisible to transport)
        human_messages = [e for e in ws2_received if e.source.channel_id == "ws1"]
        ai_messages = [e for e in ws2_received if e.source.channel_id == "ai1"]
        assert len(human_messages) == 1
        assert len(ai_messages) == 0

    async def test_three_party_conversation(self) -> None:
        """Two humans + AI in same room, all visible to all."""
        kit = RoomKit()
        ai_provider = MockAIProvider(responses=["AI contributes"])
        ws1 = WebSocketChannel("ws1")
        ws2 = WebSocketChannel("ws2")
        ai = AIChannel("ai1", provider=ai_provider)
        ws1_received: list[RoomEvent] = []
        ws2_received: list[RoomEvent] = []

        async def ws1_send(conn_id: str, event: RoomEvent) -> None:
            ws1_received.append(event)

        async def ws2_send(conn_id: str, event: RoomEvent) -> None:
            ws2_received.append(event)

        ws1.register_connection("conn1", ws1_send)
        ws2.register_connection("conn2", ws2_send)

        kit.register_channel(ws1)
        kit.register_channel(ws2)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ws2")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="What do you think?"),
        )
        await kit.process_inbound(msg)

        # ws2 should receive the human message
        assert len(ws2_received) >= 1
