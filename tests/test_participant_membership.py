"""Tests for explicit room membership (join/leave) and read-marker aggregation.

Covers the framework additions:
- ``add_member`` / ``remove_member`` — explicit join/leave emitting
  ``PARTICIPANT_JOINED`` / ``PARTICIPANT_LEFT`` and firing the matching hooks.
- ``list_members`` / ``is_member`` — active-roster enumeration.
- ``list_read_markers`` — per-channel read high-water-marks used to aggregate
  "seen by" receipts.
"""

from __future__ import annotations

import pytest

from roomkit.core.exceptions import ParticipantNotFoundError
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    EventType,
    HookExecution,
    HookTrigger,
    IdentificationStatus,
    ParticipantRole,
    ParticipantStatus,
)
from roomkit.models.event import RoomEvent
from tests.conftest import make_event


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


def _async_hook(trigger: HookTrigger, fn: object) -> HookRegistration:
    return HookRegistration(
        trigger=trigger,
        execution=HookExecution.ASYNC,
        fn=fn,
        name=f"test_{trigger}",
    )


class TestMembership:
    async def test_add_member_creates_active_participant(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        p = await kit.add_member(
            "r1", "ws:u1:r1", "u1", identity_id="u1", display_name="Alice"
        )
        assert p.status == ParticipantStatus.ACTIVE
        assert p.role == ParticipantRole.MEMBER
        assert p.identity_id == "u1"
        members = await kit.list_members("r1")
        assert [m.id for m in members] == ["u1"]
        assert await kit.is_member("r1", "u1") is True

    async def test_add_member_emits_joined_event(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        events = await kit.store.list_events("r1")
        assert any(e.type == EventType.PARTICIPANT_JOINED for e in events)

    async def test_add_member_fires_hook(self, kit: RoomKit) -> None:
        fired: list[RoomEvent] = []

        async def on_joined(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_PARTICIPANT_JOINED, on_joined))
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        assert len(fired) == 1
        assert fired[0].room_id == "r1"

    async def test_add_member_twice_is_idempotent(self, kit: RoomKit) -> None:
        fired: list[RoomEvent] = []

        async def on_joined(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_PARTICIPANT_JOINED, on_joined))
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        # a second add on an already-active member is a no-op: one join event,
        # one participant row — safe to call on every room open
        assert len(fired) == 1
        assert [m.id for m in await kit.list_members("r1")] == ["u1"]

    async def test_add_member_with_identity_is_identified(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        p = await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        assert p.identification == IdentificationStatus.IDENTIFIED

    async def test_rejoin_flips_left_back_to_active(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        first = await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        await kit.remove_member("r1", "u1")
        assert await kit.is_member("r1", "u1") is False
        rejoined = await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        assert rejoined.status == ParticipantStatus.ACTIVE
        # joined_at is preserved across a re-join (idempotent membership)
        assert rejoined.joined_at == first.joined_at
        assert await kit.is_member("r1", "u1") is True

    async def test_remove_member_soft_flips_to_left(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        left = await kit.remove_member("r1", "u1")
        assert left.status == ParticipantStatus.LEFT
        # the row is NOT deleted — still visible when including those who left
        all_members = await kit.list_members("r1", include_left=True)
        assert [m.id for m in all_members] == ["u1"]
        # but it drops out of the active roster
        assert await kit.list_members("r1") == []

    async def test_remove_member_emits_left_event_and_hook(self, kit: RoomKit) -> None:
        fired: list[RoomEvent] = []

        async def on_left(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_PARTICIPANT_LEFT, on_left))
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        await kit.remove_member("r1", "u1")
        assert len(fired) == 1
        events = await kit.store.list_events("r1")
        assert any(e.type == EventType.PARTICIPANT_LEFT for e in events)

    async def test_remove_unknown_member_raises(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        with pytest.raises(ParticipantNotFoundError):
            await kit.remove_member("r1", "ghost")

    async def test_is_member_false_for_unknown(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        assert await kit.is_member("r1", "nobody") is False

    async def test_ban_excludes_from_active_roster(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        banned = await kit.remove_member("r1", "u1", status=ParticipantStatus.BANNED)
        assert banned.status == ParticipantStatus.BANNED
        assert await kit.is_member("r1", "u1") is False


class TestReadMarkerAggregation:
    async def test_list_read_markers_empty(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        assert await kit.list_read_markers("r1") == {}

    async def test_list_read_markers_per_channel(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        events = []
        for i in range(3):
            e = make_event(room_id="r1", body=f"m{i}")
            await kit.store.add_event(e)
            events.append(e)
        # u1 read up to the last message, u2 only the first
        await kit.mark_read("r1", "ws:u1:r1", events[2].id)
        await kit.mark_read("r1", "ws:u2:r1", events[0].id)
        markers = await kit.list_read_markers("r1")
        assert markers == {"ws:u1:r1": 2, "ws:u2:r1": 0}
