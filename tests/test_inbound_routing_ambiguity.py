"""A router does not guess (RFC §10.4).

Returning one of several candidate rooms is a durable cross-room disclosure:
the message is stored in the wrong room, broadcast to that room's channels,
and read back as context by that room's agent. Null is the safe answer — the
framework creates a new room, which is recoverable.
"""

from __future__ import annotations

from roomkit.core.inbound_router import DefaultInboundRoomRouter
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore


async def _store_with_rooms(*room_ids: str, channel_id: str = "ws") -> InMemoryStore:
    store = InMemoryStore()
    for rid in room_ids:
        await store.create_room(Room(id=rid))
        await store.add_binding(
            ChannelBinding(
                channel_id=channel_id,
                room_id=rid,
                channel_type=ChannelType.WEBSOCKET,
            )
        )
    return store


class TestAmbiguityIsRefused:
    async def test_one_bound_room_routes(self) -> None:
        store = await _store_with_rooms("r1")
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET) == "r1"

    async def test_two_bound_rooms_refuse_to_route(self) -> None:
        store = await _store_with_rooms("r1", "r2")
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET) is None

    async def test_a_closed_room_does_not_make_it_ambiguous(self) -> None:
        """Only ACTIVE rooms are candidates, so closing one disambiguates."""
        store = await _store_with_rooms("r1", "r2")
        room = await store.get_room("r2")
        assert room is not None
        await store.update_room(room.model_copy(update={"status": RoomStatus.CLOSED}))
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET) == "r1"

    async def test_no_binding_at_all_returns_none(self) -> None:
        store = InMemoryStore()
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET) is None


class TestParticipantRuleWins:
    """RFC §10.4 tries the sender's own room first — a binding is only a pipe."""

    async def test_the_senders_room_is_preferred_over_the_binding(self) -> None:
        store = await _store_with_rooms("r1", "r2")
        # Alice belongs to r2; the channel is bound to both, so the binding
        # rule alone would be ambiguous and give up.
        await store.add_binding(
            ChannelBinding(
                channel_id="ws-alice",
                room_id="r2",
                channel_type=ChannelType.WEBSOCKET,
                participant_id="alice",
            )
        )
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET, participant_id="alice") == "r2"


class TestDeterminism:
    async def test_the_same_state_always_gives_the_same_answer(self) -> None:
        store = await _store_with_rooms("r1")
        router = DefaultInboundRoomRouter(store)

        answers = {await router.route("ws", ChannelType.WEBSOCKET) for _ in range(20)}
        assert answers == {"r1"}

    async def test_candidates_are_ordered_by_room_age(self) -> None:
        """The store's order must not be an accident of insertion."""
        store = await _store_with_rooms("z-room", "a-room")

        ids = await store.find_room_ids_by_channel("ws", status=str(RoomStatus.ACTIVE), limit=10)

        rooms = [await store.get_room(i) for i in ids]
        created = [r.created_at for r in rooms if r is not None]
        assert created == sorted(created)
