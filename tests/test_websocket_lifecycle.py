"""Tests for WebSocket lifecycle integration (Area 4.7)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels.websocket import (
    StreamChunk,
    StreamEnd,
    StreamMessage,
    StreamStart,
    WebSocketChannel,
)
from roomkit.core.framework import ChannelNotRegisteredError, RoomKit
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.room import Room
from tests.test_framework import SimpleChannel


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


def _make_event(room_id: str = "room-1") -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="ai", channel_type=ChannelType.AI),
        content=TextContent(body=""),
    )


def _make_binding(channel_id: str = "ws1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id="room-1",
        channel_type=ChannelType.WEBSOCKET,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
    )


def _make_context() -> RoomContext:
    return RoomContext(room=Room(id="room-1"), bindings=[_make_binding()])


async def _noop_send(conn_id: str, event: RoomEvent) -> None:
    pass


async def _noop_stream_send(conn_id: str, msg: StreamMessage) -> None:
    pass


async def _text_stream(*chunks: str) -> Any:
    for chunk in chunks:
        yield chunk


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


class TestStreamingDelivery:
    """Tests for WebSocket streaming text delivery."""

    async def test_supports_streaming_delivery_false_by_default(self) -> None:
        """No stream_send_fn registered → supports_streaming_delivery is False."""
        ws = WebSocketChannel("ws1")
        ws.register_connection("c1", _noop_send)
        assert ws.supports_streaming_delivery is False

    async def test_supports_streaming_delivery_true(self) -> None:
        """At least one stream_send_fn → supports_streaming_delivery is True."""
        ws = WebSocketChannel("ws1")
        ws.register_connection("c1", _noop_send, stream_send_fn=_noop_stream_send)
        assert ws.supports_streaming_delivery is True

    async def test_deliver_stream_sends_protocol_messages(self) -> None:
        """deliver_stream sends stream_start, stream_chunk(s), stream_end."""
        ws = WebSocketChannel("ws1")
        received: list[StreamMessage] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            received.append(msg)

        ws.register_connection("c1", _noop_send, stream_send_fn=stream_send)

        event = _make_event()
        binding = _make_binding()
        context = _make_context()

        await ws.deliver_stream(_text_stream("Hello", " world"), event, binding, context)

        assert len(received) == 4  # start + 2 chunks + end
        assert isinstance(received[0], StreamStart)
        assert isinstance(received[1], StreamChunk)
        assert isinstance(received[2], StreamChunk)
        assert isinstance(received[3], StreamEnd)

        # All share the same stream_id
        stream_id = received[0].stream_id
        assert all(m.stream_id == stream_id for m in received)

    async def test_deliver_stream_accumulated_text(self) -> None:
        """StreamChunk.text accumulates progressively."""
        ws = WebSocketChannel("ws1")
        chunks: list[StreamChunk] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            if isinstance(msg, StreamChunk):
                chunks.append(msg)

        ws.register_connection("c1", _noop_send, stream_send_fn=stream_send)
        await ws.deliver_stream(
            _text_stream("a", "b", "c"), _make_event(), _make_binding(), _make_context()
        )

        assert [c.text for c in chunks] == ["a", "ab", "abc"]

    async def test_deliver_stream_delta_text(self) -> None:
        """StreamChunk.delta contains only the new text for each chunk."""
        ws = WebSocketChannel("ws1")
        chunks: list[StreamChunk] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            if isinstance(msg, StreamChunk):
                chunks.append(msg)

        ws.register_connection("c1", _noop_send, stream_send_fn=stream_send)
        await ws.deliver_stream(
            _text_stream("Hello", " ", "world"), _make_event(), _make_binding(), _make_context()
        )

        assert [c.delta for c in chunks] == ["Hello", " ", "world"]

    async def test_deliver_stream_non_streaming_connections_get_final(self) -> None:
        """Connections without stream_send_fn receive the final event via send_fn."""
        ws = WebSocketChannel("ws1")
        final_events: list[RoomEvent] = []

        async def regular_send(conn_id: str, event: RoomEvent) -> None:
            final_events.append(event)

        # c1 has no stream_send_fn — should receive final event
        ws.register_connection("c1", regular_send)

        await ws.deliver_stream(
            _text_stream("Hi", " there"), _make_event(), _make_binding(), _make_context()
        )

        assert len(final_events) == 1
        assert isinstance(final_events[0].content, TextContent)
        assert final_events[0].content.body == "Hi there"

    async def test_deliver_stream_error_removes_connection(self) -> None:
        """Streaming send failures increment error count and remove after threshold."""
        ws = WebSocketChannel("ws1")

        async def failing_stream_send(conn_id: str, msg: StreamMessage) -> None:
            raise ConnectionError("gone")

        ws.register_connection("c1", _noop_send, stream_send_fn=failing_stream_send)
        assert ws.connection_count == 1

        # stream_start (err 1), then 3 chunks each cause an error
        # After 3 consecutive errors the connection is removed
        await ws.deliver_stream(
            _text_stream("a", "b", "c"), _make_event(), _make_binding(), _make_context()
        )

        assert ws.connection_count == 0

    async def test_deliver_stream_empty_stream(self) -> None:
        """Empty text_stream sends start + end with empty text content."""
        ws = WebSocketChannel("ws1")
        received: list[StreamMessage] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            received.append(msg)

        ws.register_connection("c1", _noop_send, stream_send_fn=stream_send)

        await ws.deliver_stream(_text_stream(), _make_event(), _make_binding(), _make_context())

        assert len(received) == 2  # start + end only
        assert isinstance(received[0], StreamStart)
        assert isinstance(received[1], StreamEnd)
        assert isinstance(received[1].event.content, TextContent)
        assert received[1].event.content.body == ""

    async def test_deliver_stream_mixed_connections(self) -> None:
        """Mixed: streaming connections get protocol, non-streaming get final event."""
        ws = WebSocketChannel("ws1")
        stream_msgs: list[StreamMessage] = []
        regular_events: list[RoomEvent] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            stream_msgs.append(msg)

        async def regular_send(conn_id: str, event: RoomEvent) -> None:
            regular_events.append(event)

        # c1 = streaming, c2 = regular
        ws.register_connection("c1", _noop_send, stream_send_fn=stream_send)
        ws.register_connection("c2", regular_send)

        await ws.deliver_stream(
            _text_stream("foo", "bar"), _make_event(), _make_binding(), _make_context()
        )

        # Streaming connection got start + 2 chunks + end
        assert len(stream_msgs) == 4
        assert isinstance(stream_msgs[0], StreamStart)
        assert isinstance(stream_msgs[-1], StreamEnd)

        # Regular connection got the final event
        assert len(regular_events) == 1
        assert isinstance(regular_events[0].content, TextContent)
        assert regular_events[0].content.body == "foobar"

    async def test_unregister_clears_stream_send_fn(self) -> None:
        """unregister_connection removes stream_send_fn too."""
        ws = WebSocketChannel("ws1")
        ws.register_connection("c1", _noop_send, stream_send_fn=_noop_stream_send)
        assert ws.supports_streaming_delivery is True

        ws.unregister_connection("c1")
        assert ws.supports_streaming_delivery is False
