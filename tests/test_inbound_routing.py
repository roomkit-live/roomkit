"""Tests for inbound room routing (Phase C, Area 8)."""

from __future__ import annotations

from typing import Any

from roomkit.core.framework import RoomKit
from roomkit.core.inbound_router import (
    DefaultInboundRoomRouter,
    InboundRoomRouter,
)
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.models.event import TextContent
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel


def _msg(channel_id: str = "sms1", sender_id: str = "user1") -> InboundMessage:
    return InboundMessage(
        channel_id=channel_id,
        sender_id=sender_id,
        content=TextContent(body="hello"),
    )


# ---- DefaultInboundRoomRouter unit tests ----


class TestDefaultInboundRoomRouter:
    async def test_finds_room_by_binding(self) -> None:
        """Default router finds room by channel binding."""
        store = InMemoryStore()
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.room import Room

        await store.create_room(Room(id="r1"))
        await store.add_binding(
            ChannelBinding(channel_id="sms1", room_id="r1", channel_type=ChannelType.SMS)
        )

        router = DefaultInboundRoomRouter(store)
        result = await router.route("sms1", ChannelType.SMS)
        assert result == "r1"

    async def test_skips_closed_rooms(self) -> None:
        """Default router skips closed rooms."""
        store = InMemoryStore()
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.room import Room

        room = Room(id="r1", status=RoomStatus.CLOSED)
        await store.create_room(room)
        await store.add_binding(
            ChannelBinding(channel_id="sms1", room_id="r1", channel_type=ChannelType.SMS)
        )

        router = DefaultInboundRoomRouter(store)
        result = await router.route("sms1", ChannelType.SMS)
        assert result is None

    async def test_returns_none_for_unknown_channel(self) -> None:
        """Default router returns None for unbound channel."""
        store = InMemoryStore()
        router = DefaultInboundRoomRouter(store)
        result = await router.route("sms999", ChannelType.SMS)
        assert result is None


# ---- Framework integration tests ----


class TestInboundRoutingIntegration:
    async def test_existing_binding_routes_correctly(self) -> None:
        """process_inbound routes to existing room via binding."""
        kit = RoomKit()
        ch = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "sms2")

        result = await kit.process_inbound(_msg())
        assert result.event is not None
        assert result.event.room_id == "r1"

    async def test_auto_create_room_when_no_match(self) -> None:
        """process_inbound auto-creates room when router returns None."""
        kit = RoomKit()
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)

        # No room or binding exists
        result = await kit.process_inbound(_msg())
        assert result.event is not None
        # Verify a room was created
        rooms = await kit.store.list_rooms()
        assert len(rooms) == 1
        # Verify the channel was attached
        binding = await kit.store.get_binding(rooms[0].id, "sms1")
        assert binding is not None

    async def test_custom_router_used(self) -> None:
        """A custom InboundRoomRouter is used when provided."""

        class FixedRouter(InboundRoomRouter):
            async def route(
                self,
                channel_id: str,
                channel_type: ChannelType,
                participant_id: str | None = None,
                channel_data: dict[str, Any] | None = None,
            ) -> str | None:
                return "custom-room"

        kit = RoomKit(inbound_router=FixedRouter())
        ch = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch)
        kit.register_channel(ch2)
        await kit.create_room(room_id="custom-room")
        await kit.attach_channel("custom-room", "sms1")
        await kit.attach_channel("custom-room", "sms2")

        result = await kit.process_inbound(_msg())
        assert result.event is not None
        assert result.event.room_id == "custom-room"

    async def test_room_latest_index_updated(self) -> None:
        """process_inbound updates room.latest_index."""
        kit = RoomKit()
        ch = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "sms2")

        await kit.process_inbound(_msg())
        room = await kit.get_room("r1")
        assert room.latest_index >= 0

    async def test_room_last_activity_updated(self) -> None:
        """process_inbound updates room.timers.last_activity_at."""
        kit = RoomKit()
        ch = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "sms2")

        room_before = await kit.get_room("r1")
        initial_activity = room_before.timers.last_activity_at

        await kit.process_inbound(_msg())

        room_after = await kit.get_room("r1")
        assert room_after.timers.last_activity_at is not None
        if initial_activity is not None:
            assert room_after.timers.last_activity_at >= initial_activity
