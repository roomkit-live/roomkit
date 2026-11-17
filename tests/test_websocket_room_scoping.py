"""A WebSocket connection receives its rooms, and only its rooms.

The registry used to be a flat ``{connection_id: send_fn}``, with no room
dimension anywhere in the API — so ``deliver()`` had nothing to filter on and
sent every room's events to every socket the channel held. A channel shared
across conversations therefore leaked them into each other, durably: the
client saw them, and so did anything reading that socket.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.websocket import SendFn, WebSocketChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from tests.test_framework import SimpleChannel


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


class TestFanOutIsBounded:
    """A socket that hangs must not take the room with it.

    Broadcast runs under the room lock and is unbounded by design (RFC §10.1),
    so an unbounded send does not merely delay other clients — it stops the
    room from accepting anything at all.
    """

    async def test_a_hanging_connection_does_not_block_the_others(self) -> None:
        ws = WebSocketChannel("ws", send_timeout=0.05)
        delivered: list[str] = []

        async def hang(conn_id: str, event: RoomEvent) -> None:
            await asyncio.sleep(30)

        async def fast(conn_id: str, event: RoomEvent) -> None:
            delivered.append(conn_id)

        ws.register_connection("zombie", hang, room_id="room-a")
        ws.register_connection("alice", fast, room_id="room-a")

        # Back once the hung socket's send_timeout ran out, never at its pace.
        await asyncio.wait_for(
            ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a")), 1.0
        )

        assert delivered == ["alice"]

    async def test_a_hanging_connection_is_eventually_evicted(self) -> None:
        """A half-open socket never raises, so only a timeout can retire it."""
        ws = WebSocketChannel("ws", send_timeout=0.01)

        async def hang(conn_id: str, event: RoomEvent) -> None:
            await asyncio.sleep(30)

        ws.register_connection("zombie", hang, room_id="room-a")

        for _ in range(WebSocketChannel._MAX_CONSECUTIVE_ERRORS):
            await ws.deliver(_event("room-a", "x"), _binding("room-a"), _context("room-a"))

        assert "zombie" not in ws._connections

    async def test_a_recovering_connection_keeps_its_place(self) -> None:
        ws = WebSocketChannel("ws", send_timeout=0.05)
        calls = {"n": 0}

        async def flaky(conn_id: str, event: RoomEvent) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                await asyncio.sleep(30)

        ws.register_connection("flaky", flaky, room_id="room-a")

        await ws.deliver(_event("room-a", "1"), _binding("room-a"), _context("room-a"))
        await ws.deliver(_event("room-a", "2"), _binding("room-a"), _context("room-a"))

        assert "flaky" in ws._connections
        assert ws._error_counts.get("flaky", 0) == 0

    async def test_slow_clients_do_not_add_up(self) -> None:
        """Concurrent, not sequential: each socket's send is under way while the
        others start, so none waits for another to finish."""
        ws = WebSocketChannel("ws", send_timeout=5.0)
        started: list[str] = []
        delivered: list[str] = []
        all_started = asyncio.Event()

        async def slow(conn_id: str, event: RoomEvent) -> None:
            started.append(conn_id)
            if len(started) == 5:
                all_started.set()
            # A sequential fan-out never starts the others while this one waits.
            await asyncio.wait_for(all_started.wait(), 1.0)
            delivered.append(conn_id)

        for i in range(5):
            ws.register_connection(f"c{i}", slow, room_id="room-a")

        await ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a"))

        assert sorted(delivered) == [f"c{i}" for i in range(5)], "fan-out was sequential"


async def _two_rooms_on_a_socket_channel() -> tuple[RoomKit, WebSocketChannel]:
    kit = RoomKit()
    ws = WebSocketChannel("ws")
    kit.register_channel(ws)
    kit.register_channel(SimpleChannel("sms1"))
    for room_id, org in (("acme-room", "acme"), ("globex-room", "globex")):
        await kit.create_room(room_id=room_id, organization_id=org)
        await kit.attach_channel(room_id, "ws")
        await kit.attach_channel(room_id, "sms1")
    return kit, ws


async def _say_in(kit: RoomKit, room_id: str) -> None:
    message = InboundMessage(
        channel_id="sms1", sender_id="+15550001", content=TextContent(body="hi")
    )
    await kit.process_inbound(message, room_id=room_id)


class _Socket:
    """A client socket: what it was sent."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def send(self, conn_id: str, event: RoomEvent) -> None:
        self.seen.append(event.room_id)


SubscribeDoor = Callable[[RoomKit, WebSocketChannel, str, SendFn, str | None], None]


def _by_register(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="acme-room", organization_id=org)


def _by_subscribe(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="lobby", organization_id=org)
    ws.subscribe(conn, "acme-room", organization_id=org)


def _by_kit_subscribe(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="lobby", organization_id=org)
    kit.subscribe_websocket("ws", conn, "acme-room", organization_id=org)


_SUBSCRIBE_DOORS: dict[str, SubscribeDoor] = {
    "register_connection": _by_register,
    "subscribe": _by_subscribe,
    "subscribe_websocket": _by_kit_subscribe,
}


async def _taken_over(door: str, kit: RoomKit, ws: WebSocketChannel, intruder: _Socket) -> None:
    """Globex's host acting on the connection acme registered."""
    if door == "register_connection":
        ws.register_connection(
            "acme-socket", intruder.send, room_id="acme-room", organization_id="globex"
        )
    elif door == "connect_websocket":
        await kit.connect_websocket(
            "ws", "acme-socket", intruder.send, room_id="globex-room", organization_id="globex"
        )
    elif door == "subscribe":
        ws.subscribe("acme-socket", "acme-room", organization_id="globex")
    elif door == "subscribe_websocket":
        kit.subscribe_websocket("ws", "acme-socket", "acme-room", organization_id="globex")
    elif door == "unsubscribe":
        ws.unsubscribe("acme-socket", "acme-room", organization_id="globex")
    elif door == "unsubscribe_websocket":
        kit.unsubscribe_websocket("ws", "acme-socket", "acme-room", organization_id="globex")
    elif door == "unregister_connection":
        ws.unregister_connection("acme-socket", organization_id="globex")
    else:
        await kit.disconnect_websocket("ws", "acme-socket", organization_id="globex")


_TAKEOVER_DOORS = [
    "connect_websocket",
    "disconnect_websocket",
    "register_connection",
    "unregister_connection",
    "subscribe",
    "subscribe_websocket",
    "unsubscribe",
    "unsubscribe_websocket",
]


class _SharedHost:
    """A host that passes one send function, the same object, for every
    socket and dispatches by connection id, as a server holding a dict of its
    sockets does. A socket that left stays in the dict until another takes
    its id: what still reaches it is recorded under its name."""

    def __init__(self, ws: WebSocketChannel) -> None:
        self.ws = ws
        self.sockets: dict[str, str] = {}
        self.got: dict[str, list[str]] = {}

        async def send(conn_id: str, event: RoomEvent) -> None:
            self.got.setdefault(self.sockets[conn_id], []).append(f"event:{event.content.body}")

        async def stream(conn_id: str, msg: Any) -> None:
            delta = getattr(msg, "delta", None)
            line = msg.type if delta is None else f"{msg.type}:{delta}"
            self.got.setdefault(self.sockets[conn_id], []).append(line)

        self.send = send
        self.stream = stream

    def register(
        self, conn_id: str, who: str, room_id: str, org: str | None, *, streaming: bool = True
    ) -> None:
        self.sockets[conn_id] = who
        self.ws.register_connection(
            conn_id,
            self.send,
            room_id=room_id,
            stream_send_fn=self.stream if streaming else None,
            organization_id=org,
        )

    def unregister(self, conn_id: str) -> None:
        self.ws.unregister_connection(conn_id)


def _registered_for_globex(host: _SharedHost) -> None:
    host.unregister("c1")
    host.register("c1", "globex", "globex-room", "globex")


_REGISTRATION_CHANGES: dict[str, Callable[[_SharedHost], None]] = {
    "another-organization": _registered_for_globex,
    "same-organization": lambda host: host.register("c1", "acme-new", "acme-room", "acme"),
    "no-organization": lambda host: host.register("c1", "unscoped-new", "other-room", None),
    "unregistered": lambda host: host.unregister("c1"),
}


class TestOrganizationScope:
    """A connection registered for an organization receives only the rooms
    of that organization, and only that organization subscribes or
    unsubscribes it (RFC §17.2, RMK-491). The registry is synchronous and
    reads no room, so the room is matched at delivery."""

    @pytest.mark.parametrize("door", sorted(_SUBSCRIBE_DOORS))
    async def test_a_connection_of_another_organization_receives_nothing(self, door: str) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        for conn, org in (("acme-socket", "acme"), ("globex-socket", "globex"), ("any", None)):
            _SUBSCRIBE_DOORS[door](kit, ws, conn, send, org)

        await _say_in(kit, "acme-room")

        assert sorted(seen) == ["acme-socket", "any"]
        await kit.close()

    async def test_a_stream_reaches_no_connection_of_another_organization(self) -> None:
        ws = WebSocketChannel("ws")
        streamed: list[str] = []
        sent: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            sent.append(conn_id)

        async def stream_send(conn_id: str, msg: object) -> None:
            streamed.append(conn_id)

        for conn, org in (("acme-live", "acme"), ("globex-live", "globex")):
            ws.register_connection(
                conn, send, room_id="room-a", stream_send_fn=stream_send, organization_id=org
            )
        for conn, org in (("acme-final", "acme"), ("globex-final", "globex")):
            ws.register_connection(conn, send, room_id="room-a", organization_id=org)

        async def text_stream() -> AsyncIterator[str | RoomEvent]:
            yield "hello"
            yield _event("room-a", "hello")  # the row the stream hands once stored

        context = RoomContext(room=Room(id="room-a", organization_id="acme"))
        await ws.deliver_stream(text_stream(), _event("room-a", ""), _binding("room-a"), context)

        assert set(streamed) == {"acme-live"}
        assert sorted(sent) == ["acme-final", "acme-live"]

    @pytest.mark.parametrize("door", _TAKEOVER_DOORS)
    async def test_another_organization_cannot_take_over_a_connection(self, door: str) -> None:
        """Refused as a connection that does not exist, nothing changed: the
        socket keeps its sender, its organization and its rooms."""
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme, intruder = _Socket(), _Socket()
        ws.register_connection(
            "acme-socket", acme.send, room_id="acme-room", organization_id="acme"
        )

        with pytest.raises(ValueError, match="No connection acme-socket for organization globex"):
            await _taken_over(door, kit, ws, intruder)

        await _say_in(kit, "acme-room")
        await _say_in(kit, "globex-room")
        assert (acme.seen, intruder.seen) == (["acme-room"], [])
        assert ws.rooms_for("acme-socket") == {"acme-room"}
        await kit.close()

    @pytest.mark.parametrize("kit_door", [False, True], ids=["channel", "kit"])
    async def test_an_unscoped_subscription_never_widens_a_scoped_connection(
        self, kit_door: bool
    ) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme = _Socket()
        ws.register_connection(
            "acme-socket", acme.send, room_id="acme-room", organization_id="acme"
        )

        if kit_door:
            kit.subscribe_websocket("ws", "acme-socket", "globex-room")
            kit.subscribe_websocket("ws", "acme-socket", "acme-room")
        else:
            ws.subscribe("acme-socket", "globex-room")
            ws.subscribe("acme-socket", "acme-room")
        await _say_in(kit, "globex-room")
        await _say_in(kit, "acme-room")

        assert acme.seen == ["acme-room"]
        await kit.close()

    @pytest.mark.parametrize("kit_door", [False, True], ids=["channel", "kit"])
    async def test_the_connections_own_organization_subscribes_and_unsubscribes_it(
        self, kit_door: bool
    ) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme = _Socket()
        ws.register_connection("acme-socket", acme.send, room_id="lobby", organization_id="acme")

        if kit_door:
            kit.subscribe_websocket("ws", "acme-socket", "acme-room", organization_id="acme")
            await _say_in(kit, "acme-room")
            kit.unsubscribe_websocket("ws", "acme-socket", "acme-room", organization_id="acme")
        else:
            ws.subscribe("acme-socket", "acme-room", organization_id="acme")
            await _say_in(kit, "acme-room")
            ws.unsubscribe("acme-socket", "acme-room", organization_id="acme")
        await _say_in(kit, "acme-room")

        assert acme.seen == ["acme-room"]
        assert ws.rooms_for("acme-socket") == {"lobby"}
        await kit.close()

    @pytest.mark.parametrize(
        ("before", "registered_for", "rooms"),
        [
            pytest.param("acme", None, {"lobby"}, id="scoped-then-unscoped"),
            pytest.param("unregistered", "globex", {"lobby"}, id="subscribed-then-scoped"),
            pytest.param("acme", "acme", {"acme-room", "lobby"}, id="same-organization"),
        ],
    )
    async def test_a_registration_that_changes_the_scope_drops_the_old_subscriptions(
        self, before: str, registered_for: str | None, rooms: set[str]
    ) -> None:
        """A new scope on an id still known starts with no subscription: what
        the id followed before is no room of the new registration's."""
        kit, ws = await _two_rooms_on_a_socket_channel()
        old, new = _Socket(), _Socket()
        if before == "unregistered":
            ws.subscribe("c", "acme-room")
        else:
            ws.register_connection("c", old.send, room_id="acme-room", organization_id=before)

        ws.register_connection("c", new.send, room_id="lobby", organization_id=registered_for)
        await _say_in(kit, "acme-room")

        assert ws.rooms_for("c") == rooms
        assert new.seen == (["acme-room"] if "acme-room" in rooms else [])
        await kit.close()

    async def test_an_evicted_connection_is_forgotten_with_its_rooms(self) -> None:
        """As unregister_connection() forgets it: an id registered again after
        the eviction starts with no room and no organization."""
        kit, ws = await _two_rooms_on_a_socket_channel()
        new = _Socket()

        async def gone(conn_id: str, event: RoomEvent) -> None:
            raise ConnectionError("closed")

        ws.register_connection("c", gone, room_id="acme-room", organization_id="acme")
        for _ in range(WebSocketChannel._MAX_CONSECUTIVE_ERRORS):
            await _say_in(kit, "acme-room")
        assert ws.rooms_for("c") == set()

        ws.register_connection("c", new.send, room_id="lobby")
        await _say_in(kit, "acme-room")
        assert new.seen == []
        await kit.close()

    @pytest.mark.parametrize("kit_door", [False, True], ids=["channel", "kit"])
    async def test_the_connections_own_organization_unregisters_it(self, kit_door: bool) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme = _Socket()
        ws.register_connection(
            "acme-socket", acme.send, room_id="acme-room", organization_id="acme"
        )

        if kit_door:
            await kit.disconnect_websocket("ws", "acme-socket", organization_id="acme")
        else:
            ws.unregister_connection("acme-socket", organization_id="acme")
        await _say_in(kit, "acme-room")

        assert (acme.seen, ws.connection_count, ws.rooms_for("acme-socket")) == ([], 0, set())
        await kit.close()

    @pytest.mark.parametrize("streaming", [True, False], ids=["stream", "plain"])
    @pytest.mark.parametrize("change", sorted(_REGISTRATION_CHANGES))
    async def test_a_stream_sends_only_to_the_registrations_it_started_with(
        self, change: str, streaming: bool
    ) -> None:
        """An id unregistered, or registered again for any organization, the
        same one included, while acme's stream runs gets nothing more of it:
        a new registration never saw the stream's start. The change is ready
        on the loop just before the next fan-out, and the host shares one
        send function across its sockets (review probes P1, D1, D3)."""
        ws = WebSocketChannel("ws")
        host = _SharedHost(ws)
        host.register("c1", "acme", "acme-room", "acme", streaming=streaming)
        resume = asyncio.Event()

        async def text() -> AsyncIterator[str]:
            yield "secret-1 "
            await resume.wait()
            asyncio.get_running_loop().call_soon(_REGISTRATION_CHANGES[change], host)
            yield "secret-2"

        context = RoomContext(room=Room(id="acme-room", organization_id="acme"))
        streamed = asyncio.create_task(
            ws.deliver_stream(text(), _event("acme-room", ""), _binding("acme-room"), context)
        )
        while not host.got and streaming:
            await asyncio.sleep(0)
        resume.set()
        await asyncio.wait_for(streamed, 1)

        before_the_change = ["stream_start", "stream_chunk:secret-1 "] if streaming else []
        assert host.got == ({"acme": before_the_change} if streaming else {})

    async def test_a_client_reconnecting_mid_stream_gets_nothing_more_without_organizations(
        self,
    ) -> None:
        """A host that uses no organization: the same id registered again for
        the same room while an answer streams to it gets neither the rest of
        the stream nor its stream_end (0.108.0 sent it both). It reads the
        stored answer from the room's history."""
        ws = WebSocketChannel("ws")
        host = _SharedHost(ws)
        host.register("c1", "first", "room", None)
        resume = asyncio.Event()

        async def text() -> AsyncIterator[str]:
            yield "part-1 "
            await resume.wait()
            host.register("c1", "reconnected", "room", None)
            yield "part-2"

        streamed = asyncio.create_task(
            ws.deliver_stream(text(), _event("room", ""), _binding("room"), _context("room"))
        )
        while not host.got:
            await asyncio.sleep(0)
        resume.set()
        await asyncio.wait_for(streamed, 1)

        assert host.got == {"first": ["stream_start", "stream_chunk:part-1 "]}

    async def test_a_delivery_reaches_no_id_registered_again_before_its_send(self) -> None:
        """The registration is checked as each send starts: one made on the
        loop between the fan-out and the send gets nothing (review probe
        D1, deliver())."""
        ws = WebSocketChannel("ws")
        host = _SharedHost(ws)
        host.register("c1", "acme", "acme-room", "acme", streaming=False)
        asyncio.get_running_loop().call_soon(_REGISTRATION_CHANGES["another-organization"], host)

        context = RoomContext(room=Room(id="acme-room", organization_id="acme"))
        await ws.deliver(_event("acme-room", "acme secret"), _binding("acme-room"), context)

        assert host.got == {}

    async def test_failures_of_a_registration_gone_do_not_evict_the_new_one(self) -> None:
        """Sends to acme's dead socket still under way when the id is
        registered again for globex time out against acme's registration,
        not globex's (review probe D2)."""
        ws = WebSocketChannel("ws", send_timeout=0.05)
        hang = asyncio.Event()
        got: list[str] = []

        async def dead(conn_id: str, event: RoomEvent) -> None:
            await hang.wait()

        async def alive(conn_id: str, event: RoomEvent) -> None:
            got.append(event.room_id)

        rooms = [f"acme-{n}" for n in range(WebSocketChannel._MAX_CONSECUTIVE_ERRORS)]
        ws.register_connection("c1", dead, room_id=rooms[0], organization_id="acme")
        for room_id in rooms[1:]:
            ws.subscribe("c1", room_id)
        context = {r: RoomContext(room=Room(id=r, organization_id="acme")) for r in rooms}
        sends = [
            asyncio.create_task(ws.deliver(_event(r, "x"), _binding(r), context[r])) for r in rooms
        ]
        for _ in range(3):
            await asyncio.sleep(0)
        ws.unregister_connection("c1", organization_id="acme")
        ws.register_connection("c1", alive, room_id="globex-room", organization_id="globex")
        await asyncio.wait_for(asyncio.gather(*sends), 1)
        globex = RoomContext(room=Room(id="globex-room", organization_id="globex"))
        await ws.deliver(_event("globex-room", "hi"), _binding("globex-room"), globex)

        assert (ws.rooms_for("c1"), got) == ({"globex-room"}, ["globex-room"])

    async def test_a_room_of_no_organization_reaches_only_unscoped_connections(self) -> None:
        """A room of none is no organization's (RFC §17.2): a connection
        registered for one receives nothing from it, an unscoped one does."""
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        ws.register_connection("acme-socket", send, room_id="room-a", organization_id="acme")
        ws.register_connection("any", send, room_id="room-a")

        await ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a"))

        assert seen == ["any"]
