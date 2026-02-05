"""Tests for WebSocket lifecycle integration (Area 4.7)."""

from __future__ import annotations

import pytest

from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import ChannelNotRegisteredError, RoomKit
from roomkit.models.event import RoomEvent
from roomkit.models.framework_event import FrameworkEvent
from tests.test_framework import SimpleChannel


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestConnectWebSocket:
    async def test_connect_registers_connection(self, kit: RoomKit) -> None:
        """connect_websocket registers a connection on the channel."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send)
        assert ws.connection_count == 1

    async def test_connect_emits_framework_event(self, kit: RoomKit) -> None:
        """connect_websocket emits a 'channel_connected' framework event."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        received: list[FrameworkEvent] = []

        @kit.on("channel_connected")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send)
        assert len(received) == 1
        assert received[0].type == "channel_connected"
        assert received[0].data["connection_id"] == "conn1"

    async def test_connect_non_websocket_raises(self, kit: RoomKit) -> None:
        """connect_websocket raises for non-WebSocket channels."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        with pytest.raises(ChannelNotRegisteredError):
            await kit.connect_websocket("sms1", "conn1", send)

    async def test_connect_unregistered_raises(self, kit: RoomKit) -> None:
        """connect_websocket raises for unregistered channel."""

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        with pytest.raises(ChannelNotRegisteredError):
            await kit.connect_websocket("nope", "conn1", send)


class TestDisconnectWebSocket:
    async def test_disconnect_unregisters_connection(self, kit: RoomKit) -> None:
        """disconnect_websocket removes the connection."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send)
        assert ws.connection_count == 1

        await kit.disconnect_websocket("ws1", "conn1")
        assert ws.connection_count == 0

    async def test_disconnect_emits_framework_event(self, kit: RoomKit) -> None:
        """disconnect_websocket emits 'channel_disconnected' framework event."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        received: list[FrameworkEvent] = []

        @kit.on("channel_disconnected")
        async def handler(fe: FrameworkEvent) -> None:
            if fe.type == "channel_disconnected":
                received.append(fe)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send)
        await kit.disconnect_websocket("ws1", "conn1")
        assert len(received) == 1
        assert received[0].data["connection_id"] == "conn1"

    async def test_disconnect_nonexistent_is_safe(self, kit: RoomKit) -> None:
        """disconnect_websocket does not raise for unknown channel/conn."""
        # Should not raise even for unregistered channel
        await kit.disconnect_websocket("nope", "conn1")

    async def test_multiple_connections(self, kit: RoomKit) -> None:
        """Multiple WebSocket connections can be registered and tracked."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send)
        await kit.connect_websocket("ws1", "conn2", send)
        assert ws.connection_count == 2

        await kit.disconnect_websocket("ws1", "conn1")
        assert ws.connection_count == 1
