"""A router does not guess (RFC §10.4).

Returning one of several candidate rooms is a durable cross-room disclosure:
the message is stored in the wrong room, broadcast to that room's channels,
and read back as context by that room's agent. Null is the safe answer — the
framework creates a new room, which is recoverable.
"""

from __future__ import annotations

from roomkit.core.inbound_router import _HISTORY_PAGE, DefaultInboundRoomRouter
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, EventStatus, RoomStatus
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore
from tests.conftest import make_event


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
        await store.add_participant(Participant(id="alice", room_id="r2", channel_id="ws-alice"))
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET, participant_id="alice") == "r2"

    async def test_the_binding_of_this_channel_naming_the_sender_finds_their_room(
        self,
    ) -> None:
        store = await _store_with_rooms("r1", "r2")
        binding = await store.get_binding("r2", "ws")
        assert binding is not None
        await store.update_binding(binding.model_copy(update={"participant_id": "alice"}))
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET, participant_id="alice") == "r2"

    async def test_a_binding_of_another_channel_does_not(self) -> None:
        """A correspondent of one number is not taken to that room when they
        write to another number of the same type (RFC §10.4 step 1)."""
        store = await _store_with_rooms("r1", "r2")
        await store.add_binding(
            ChannelBinding(
                channel_id="ws-alice",
                room_id="r2",
                channel_type=ChannelType.WEBSOCKET,
                participant_id="alice",
            )
        )
        router = DefaultInboundRoomRouter(store)

        assert await router.route("ws", ChannelType.WEBSOCKET, participant_id="alice") is None


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


class TestStepThreeAdmitsOneConversation:
    """The one room bound to a channel is someone's conversation once anything
    names a correspondent (RFC §10.4 step 3)."""

    async def _route(self, store: InMemoryStore, sender: str | None = "bob") -> str | None:
        return await DefaultInboundRoomRouter(store).route(
            "ws", ChannelType.WEBSOCKET, participant_id=sender
        )

    async def test_a_binding_naming_another_sender_refuses(self) -> None:
        store = await _store_with_rooms("r1")
        binding = await store.get_binding("r1", "ws")
        assert binding is not None
        await store.update_binding(binding.model_copy(update={"participant_id": "alice"}))

        assert await self._route(store) is None

    async def test_a_participant_joined_through_the_channel_refuses(self) -> None:
        store = await _store_with_rooms("r1")
        await store.add_participant(Participant(id="alice", room_id="r1", channel_id="ws"))

        assert await self._route(store) is None

    async def test_a_participant_joined_through_another_channel_does_not(self) -> None:
        store = await _store_with_rooms("r1")
        await store.add_participant(Participant(id="alice", room_id="r1", channel_id="voice"))

        assert await self._route(store) == "r1"

    async def test_a_message_received_from_another_sender_refuses(self) -> None:
        store = await _store_with_rooms("r1")
        await store.add_event_auto_index(
            "r1", make_event(room_id="r1", channel_id="ws", participant_id="alice")
        )

        assert await self._route(store) is None

    async def test_the_senders_own_messages_do_not(self) -> None:
        store = await _store_with_rooms("r1")
        await store.add_event_auto_index(
            "r1", make_event(room_id="r1", channel_id="ws", participant_id="bob")
        )

        assert await self._route(store) == "r1"

    async def test_a_message_naming_no_one_or_blocked_does_not(self) -> None:
        store = await _store_with_rooms("r1")
        await store.add_event_auto_index("r1", make_event(room_id="r1", channel_id="ws"))
        await store.add_event_auto_index(
            "r1",
            make_event(
                room_id="r1",
                channel_id="ws",
                participant_id="alice",
                status=EventStatus.BLOCKED,
            ),
        )

        assert await self._route(store) == "r1"

    async def test_another_sender_found_past_the_first_page_refuses(self) -> None:
        store = await _store_with_rooms("r1")
        for _ in range(_HISTORY_PAGE + 1):
            await store.add_event_auto_index(
                "r1", make_event(room_id="r1", channel_id="ws", participant_id="bob")
            )
        await store.add_event_auto_index(
            "r1", make_event(room_id="r1", channel_id="ws", participant_id="alice")
        )

        assert await self._route(store) is None

    async def test_a_sender_with_no_address_is_held_to_the_same_rule(self) -> None:
        store = await _store_with_rooms("r1")
        assert await self._route(store, sender=None) == "r1"
        await store.add_event_auto_index(
            "r1", make_event(room_id="r1", channel_id="ws", participant_id="alice")
        )

        assert await self._route(store, sender=None) is None

    async def test_a_group_binding_admits_everyone(self) -> None:
        store = await _store_with_rooms("r1")
        binding = await store.get_binding("r1", "ws")
        assert binding is not None
        await store.update_binding(
            binding.model_copy(update={"group": True, "participant_id": "alice"})
        )
        await store.add_event_auto_index(
            "r1", make_event(room_id="r1", channel_id="ws", participant_id="alice")
        )

        assert await self._route(store) == "r1"
