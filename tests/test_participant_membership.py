"""Tests for explicit room membership (join/leave) and read-marker aggregation.

Covers the framework additions:
- ``add_member`` / ``remove_member`` — explicit join/leave emitting
  ``PARTICIPANT_JOINED`` / ``PARTICIPANT_LEFT`` and firing the matching hooks.
- ``list_members`` / ``is_member`` — active-roster enumeration.
- ``ensure_participant`` / ``add_member`` reached through a second channel —
  one record, ``connected_via``, and the warning that names both (RFC §5.5).
- ``list_read_markers`` — per-channel read high-water-marks used to aggregate
  "seen by" receipts.
"""

from __future__ import annotations

import logging

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
        p = await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1", display_name="Alice")
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


class TestRenameMember:
    """Change what a member is called — never who they are (RMK-73, RFC §5.5).

    ``add_member()`` on an ACTIVE member is deliberately a no-op, so a display
    name set at join stays put; ``rename_member()`` is the verb that changes
    it in place, with the event and hook an interface reflects the room from.
    """

    async def test_rename_changes_the_display_name_in_place(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1", display_name="Alice")
        renamed = await kit.rename_member("r1", "u1", "Alice Tremblay")
        assert renamed.display_name == "Alice Tremblay"
        members = await kit.list_members("r1")
        assert members[0].display_name == "Alice Tremblay"
        # presentation only: who they are is untouched
        assert renamed.id == "u1"
        assert renamed.identity_id == "u1"

    async def test_rename_emits_updated_event_and_fires_the_hook(self, kit: RoomKit) -> None:
        fired: list[RoomEvent] = []

        async def on_updated(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_PARTICIPANT_UPDATED, on_updated))
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1", display_name="Alice")
        await kit.rename_member("r1", "u1", "Alice T.")
        events = await kit.store.list_events("r1")
        assert any(e.type == EventType.PARTICIPANT_UPDATED for e in events)
        assert len(fired) == 1
        assert fired[0].content.data["display_name"] == "Alice T."

    async def test_rename_to_the_name_already_held_is_a_noop(self, kit: RoomKit) -> None:
        fired: list[RoomEvent] = []

        async def on_updated(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_PARTICIPANT_UPDATED, on_updated))
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1", display_name="Alice")
        await kit.rename_member("r1", "u1", "Alice")
        events = await kit.store.list_events("r1")
        assert not any(e.type == EventType.PARTICIPANT_UPDATED for e in events)
        assert fired == []

    async def test_rename_of_an_unknown_member_raises(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        with pytest.raises(ParticipantNotFoundError):
            await kit.rename_member("r1", "ghost", "Nobody")


class TestChannelReuse:
    """One record, several channels — and it says so (RMK-108, RFC §5.5).

    ``ensure_participant`` looks a participant up by (room, id), so a caller
    naming a channel the record was not created on gets that record back. That
    is the cross-channel identity working as intended; what it must not be is
    silent, because a caller that keeps a lifecycle on the record it received is
    driving another channel's record with it.
    """

    async def test_ensure_participant_does_not_rehome_an_existing_record(
        self, kit: RoomKit
    ) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        got = await kit.ensure_participant("r1", "conference:r1", "u1")
        # the record is handed back as it stands — primary channel included
        assert got.channel_id == "ws:u1:r1"
        # and there is still exactly one record for u1
        assert [p.id for p in await kit.store.list_participants("r1")] == ["u1"]

    async def test_ensure_participant_records_the_channel_that_asked(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        got = await kit.ensure_participant("r1", "conference:r1", "u1")
        assert got.connected_via == ["ws:u1:r1", "conference:r1"]
        # persisted, not only returned
        stored = await kit.store.get_participant("r1", "u1")
        assert stored is not None
        assert stored.connected_via == ["ws:u1:r1", "conference:r1"]

    async def test_ensure_participant_warns_naming_both_channels(
        self, kit: RoomKit, caplog: pytest.LogCaptureFixture
    ) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
            await kit.ensure_participant("r1", "conference:r1", "u1")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        # the diagnostic is worthless unless it names *both* channels
        message = warnings[0].getMessage()
        assert "ws:u1:r1" in message
        assert "conference:r1" in message

    async def test_ensure_participant_on_the_same_channel_is_quiet(
        self, kit: RoomKit, caplog: pytest.LogCaptureFixture
    ) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
            got = await kit.ensure_participant("r1", "ws:u1:r1", "u1")
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
        assert got.connected_via == ["ws:u1:r1"]

    async def test_a_repeated_reach_does_not_duplicate_the_entry(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        await kit.ensure_participant("r1", "conference:r1", "u1")
        got = await kit.ensure_participant("r1", "conference:r1", "u1")
        assert got.connected_via == ["ws:u1:r1", "conference:r1"]

    async def test_ensure_participant_creating_a_record_seeds_the_list(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        created = await kit.ensure_participant("r1", "sms-main", "u1")
        assert created.channel_id == "sms-main"
        assert created.connected_via == ["sms-main"]

    async def test_recording_a_channel_emits_nothing(self, kit: RoomKit) -> None:
        """Bookkeeping, not presentation (RFC §5.5)."""
        fired: list[RoomEvent] = []

        async def on_updated(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_PARTICIPANT_UPDATED, on_updated))
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        before = len(await kit.store.list_events("r1"))
        await kit.ensure_participant("r1", "conference:r1", "u1")
        assert len(await kit.store.list_events("r1")) == before
        assert fired == []

    async def test_add_member_moves_the_primary_but_keeps_the_one_it_replaces(
        self, kit: RoomKit, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A deliberate join MAY re-home — and says so (RFC §5.5)."""
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        await kit.remove_member("r1", "u1")
        with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
            rejoined = await kit.add_member("r1", "conference:r1", "u1", identity_id="u1")
        assert rejoined.channel_id == "conference:r1"
        assert rejoined.connected_via == ["ws:u1:r1", "conference:r1"]
        message = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert "ws:u1:r1" in message
        assert "conference:r1" in message

    async def test_the_shared_id_scenario_leaves_a_readable_trail(self, kit: RoomKit) -> None:
        """The defect this test exists for (RMK-108).

        A WebSocket channel writes team-channel membership, a conference then
        asks for a participant under the same id, and the conference lifecycle
        drives the record from there — leaving the call flipped the membership
        to LEFT. The library cannot stop a host handing over its own id, but the
        record now carries the evidence that two channels share it.
        """
        await kit.create_room(room_id="r1")
        await kit.add_member("r1", "ws:u1:r1", "u1", identity_id="u1")
        await kit.ensure_participant("r1", "conference:r1", "u1")
        # the conference's departure flips the shared record, as it always did
        left = await kit.remove_member("r1", "u1")
        assert left.status == ParticipantStatus.LEFT
        # but the record now names both channels, so the collision is legible
        assert left.channel_id == "ws:u1:r1"
        assert left.connected_via == ["ws:u1:r1", "conference:r1"]


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
