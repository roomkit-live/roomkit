"""Tests for WebSocket channel."""

from __future__ import annotations

from roomkit.channels.websocket import WebSocketChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from tests.conftest import make_event


class TestWebSocketChannel:
    async def test_handle_inbound(self) -> None:
        ch = WebSocketChannel("ws1")
        ctx = RoomContext(room=Room(id="r1"))
        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        event = await ch.handle_inbound(msg, ctx)
        assert event.source.channel_type == ChannelType.WEBSOCKET

    async def test_deliver_to_connections(self) -> None:
        ch = WebSocketChannel("ws1")
        sent: list[tuple[str, RoomEvent]] = []

        async def mock_send(conn_id: str, event: RoomEvent) -> None:
            sent.append((conn_id, event))

        ch.register_connection("conn1", mock_send)
        ch.register_connection("conn2", mock_send)

        binding = ChannelBinding(
            channel_id="ws1", room_id="r1", channel_type=ChannelType.WEBSOCKET
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(channel_id="sms1")
        await ch.deliver(event, binding, ctx)
        assert len(sent) == 2

    async def test_connection_management(self) -> None:
        ch = WebSocketChannel("ws1")

        async def mock_send(conn_id: str, event: RoomEvent) -> None:
            pass

        ch.register_connection("conn1", mock_send)
        assert ch.connection_count == 1
        ch.unregister_connection("conn1")
        assert ch.connection_count == 0
