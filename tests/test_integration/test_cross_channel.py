"""Integration: Human-to-Human SMS <-> WebSocket cross-channel messaging."""

from __future__ import annotations

from roomkit.channels import SMSChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.sms.mock import MockSMSProvider


class TestCrossChannel:
    async def test_sms_to_websocket(self) -> None:
        """SMS message should be delivered to WebSocket connection."""
        kit = RoomKit()
        sms_provider = MockSMSProvider()
        sms = SMSChannel("sms1", provider=sms_provider, from_number="+15550001111")
        ws = WebSocketChannel("ws1")
        ws_received: list[RoomEvent] = []

        async def ws_send(conn_id: str, event: RoomEvent) -> None:
            ws_received.append(event)

        ws.register_connection("conn1", ws_send)

        kit.register_channel(sms)
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="+15559999999",
            content=TextContent(body="Hello from SMS"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert len(ws_received) == 1
        assert isinstance(ws_received[0].content, TextContent)
        assert ws_received[0].content.body == "Hello from SMS"

    async def test_websocket_to_sms(self) -> None:
        """WebSocket message should be delivered to SMS provider."""
        kit = RoomKit()
        sms_provider = MockSMSProvider()
        sms = SMSChannel("sms1", provider=sms_provider, from_number="+15550001111")
        ws = WebSocketChannel("ws1")

        kit.register_channel(sms)
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1", metadata={"phone_number": "+15559999999"})
        await kit.attach_channel("r1", "ws1")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Hello from WebSocket"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert len(sms_provider.sent) == 1

    async def test_multiple_websocket_connections(self) -> None:
        """Multiple WebSocket connections should all receive the message."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        all_received: list[tuple[str, RoomEvent]] = []

        async def ws_send(conn_id: str, event: RoomEvent) -> None:
            all_received.append((conn_id, event))

        ws.register_connection("conn1", ws_send)
        ws.register_connection("conn2", ws_send)

        sms = SMSChannel("sms1")
        kit.register_channel(sms)
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="broadcast"),
        )
        await kit.process_inbound(msg)
        assert len(all_received) == 2
        conn_ids = {r[0] for r in all_received}
        assert conn_ids == {"conn1", "conn2"}
