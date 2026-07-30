"""A WebSocket connection receives its rooms, and only its rooms.

The registry used to be a flat ``{connection_id: send_fn}``, with no room
dimension anywhere in the API — so ``deliver()`` had nothing to filter on and
sent every room's events to every socket the channel held. A channel shared
across conversations therefore leaked them into each other, durably: the
client saw them, and so did anything reading that socket.
"""

from __future__ import annotations

import pytest

from roomkit.channels.websocket import WebSocketChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room


def _binding(room_id: str, channel_id: str = "ws") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id, room_id=room_id, channel_type=ChannelType.WEBSOCKET
    )


def _context(room_id: str) -> RoomContext:
    return RoomContext(room=Room(id=room_id))


def _event(room_id: str, body: str) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="other", channel_type=ChannelType.WEBSOCKET),
        content=TextContent(body=body),
    )


class TestDeliveryIsScopedToTheRoom:
    async def test_a_room_reaches_only_its_own_connections(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[tuple[str, str]] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append((conn_id, event.content.body))

        ws.register_connection("alice", send, room_id="room-a")
        ws.register_connection("bob", send, room_id="room-b")

        await ws.deliver(_event("room-a", "secret"), _binding("room-a"), _context("room-a"))

        assert seen == [("alice", "secret")]

    async def test_several_connections_in_one_room_all_receive(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        ws.register_connection("a1", send, room_id="room-a")
        ws.register_connection("a2", send, room_id="room-a")
        ws.register_connection("b1", send, room_id="room-b")

        await ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a"))

        assert sorted(seen) == ["a1", "a2"]

    async def test_a_room_with_no_connections_delivers_to_nobody(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        ws.register_connection("alice", send, room_id="room-a")

        await ws.deliver(_event("room-z", "hi"), _binding("room-z"), _context("room-z"))

        assert seen == []


class TestSubscription:
    async def test_one_socket_can_follow_several_rooms(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(event.room_id)

        ws.register_connection("alice", send, room_id="room-a")
        ws.subscribe("alice", "room-b")

        await ws.deliver(_event("room-a", "x"), _binding("room-a"), _context("room-a"))
        await ws.deliver(_event("room-b", "y"), _binding("room-b"), _context("room-b"))

        assert seen == ["room-a", "room-b"]

    async def test_unsubscribing_stops_delivery(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(event.room_id)

        ws.register_connection("alice", send, room_id="room-a")
        ws.subscribe("alice", "room-b")
        ws.unsubscribe("alice", "room-b")

        await ws.deliver(_event("room-b", "y"), _binding("room-b"), _context("room-b"))

        assert seen == []
        assert ws.rooms_for("alice") == {"room-a"}

    async def test_unregistering_drops_every_subscription(self) -> None:
        ws = WebSocketChannel("ws")

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        ws.register_connection("alice", send, room_id="room-a")
        ws.subscribe("alice", "room-b")
        ws.unregister_connection("alice")

        assert ws.rooms_for("alice") == set()
        assert ws._room_connections == {}

    async def test_room_id_is_required(self) -> None:
        """The parameter exists so a connection can never be unplaceable."""
        ws = WebSocketChannel("ws")

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        with pytest.raises(TypeError):
            ws.register_connection("alice", send)  # ty: ignore[missing-argument]


class TestStreamingIsScopedToo:
    async def test_streaming_capability_is_answered_per_room(self) -> None:
        ws = WebSocketChannel("ws")

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        async def stream_send(conn_id: str, msg: object) -> None:
            pass

        ws.register_connection("alice", send, room_id="room-a", stream_send_fn=stream_send)
        ws.register_connection("bob", send, room_id="room-b")

        assert ws.supports_streaming_delivery_for("room-a") is True
        assert ws.supports_streaming_delivery_for("room-b") is False
        # The channel-wide answer stays true — some client somewhere streams.
        assert ws.supports_streaming_delivery is True

    async def test_stream_messages_stay_in_their_room(self) -> None:
        ws = WebSocketChannel("ws")
        streamed: list[tuple[str, str]] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        async def stream_send(conn_id: str, msg: object) -> None:
            streamed.append((conn_id, msg.type))

        ws.register_connection("alice", send, room_id="room-a", stream_send_fn=stream_send)
        ws.register_connection("bob", send, room_id="room-b", stream_send_fn=stream_send)

        async def text_stream():
            yield "hello"

        await ws.deliver_stream(
            text_stream(), _event("room-a", ""), _binding("room-a"), _context("room-a")
        )

        assert {conn_id for conn_id, _ in streamed} == {"alice"}
