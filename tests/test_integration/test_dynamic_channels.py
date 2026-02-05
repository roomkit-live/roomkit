"""Integration: Dynamic channel management - attach, mute, unmute, visibility."""

from __future__ import annotations

from roomkit.channels import SMSChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent


class TestDynamicChannels:
    async def test_attach_detach(self) -> None:
        """Channels can be attached and detached dynamically."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        sms = SMSChannel("sms1")

        kit.register_channel(ws)
        kit.register_channel(sms)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "sms1")

        bindings = await kit.store.list_bindings("r1")
        assert len(bindings) == 2

        await kit.detach_channel("r1", "sms1")
        bindings = await kit.store.list_bindings("r1")
        assert len(bindings) == 1
        assert bindings[0].channel_id == "ws1"

    async def test_mute_prevents_delivery(self) -> None:
        """Muted source channel's messages are not broadcast."""
        kit = RoomKit()
        ws1 = WebSocketChannel("ws1")
        ws2 = WebSocketChannel("ws2")
        ws2_received: list[RoomEvent] = []

        async def ws2_send(conn_id: str, event: RoomEvent) -> None:
            ws2_received.append(event)

        ws2.register_connection("conn2", ws2_send)

        kit.register_channel(ws1)
        kit.register_channel(ws2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ws2")

        # Mute ws1
        await kit.mute("r1", "ws1")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Should not be delivered"),
        )
        await kit.process_inbound(msg)
        assert len(ws2_received) == 0

    async def test_unmute_restores_delivery(self) -> None:
        """Unmuting restores normal delivery."""
        kit = RoomKit()
        ws1 = WebSocketChannel("ws1")
        ws2 = WebSocketChannel("ws2")
        ws2_received: list[RoomEvent] = []

        async def ws2_send(conn_id: str, event: RoomEvent) -> None:
            ws2_received.append(event)

        ws2.register_connection("conn2", ws2_send)

        kit.register_channel(ws1)
        kit.register_channel(ws2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ws2")

        await kit.mute("r1", "ws1")
        await kit.unmute("r1", "ws1")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Should be delivered now"),
        )
        await kit.process_inbound(msg)
        assert len(ws2_received) == 1

    async def test_visibility_change(self) -> None:
        """Changing visibility dynamically affects routing."""
        kit = RoomKit()
        ws1 = WebSocketChannel("ws1")
        ws2 = WebSocketChannel("ws2")
        ws2_received: list[RoomEvent] = []

        async def ws2_send(conn_id: str, event: RoomEvent) -> None:
            ws2_received.append(event)

        ws2.register_connection("conn2", ws2_send)

        kit.register_channel(ws1)
        kit.register_channel(ws2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ws2")

        # Set ws1 visibility to "none"
        await kit.set_visibility("r1", "ws1", "none")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Invisible message"),
        )
        await kit.process_inbound(msg)
        assert len(ws2_received) == 0

        # Restore visibility
        await kit.set_visibility("r1", "ws1", "all")
        msg2 = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Visible message"),
        )
        await kit.process_inbound(msg2)
        assert len(ws2_received) == 1
