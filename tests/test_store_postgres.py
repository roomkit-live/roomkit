"""Tests for PostgresStore.

These tests require a running PostgreSQL instance. Set POSTGRES_DSN to run them:

    POSTGRES_DSN=postgresql://user:pass@localhost/roomkit_test \
    pytest tests/test_store_postgres.py -v
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

import pytest

from roomkit.core.mixins.verification import HELD_KEY
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task
from tests.conftest import PoolCheckoutRecorder

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set"),
    pytest.mark.asyncio,
]


@pytest.fixture
async def store():
    from roomkit.store.postgres import PostgresStore

    _index_counters.clear()
    s = PostgresStore(dsn=POSTGRES_DSN)
    await s.init(min_size=1, max_size=3)
    # Clean tables before each test
    async with s._pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE rooms, events, bindings, participants, "
            "identities, identity_addresses, tasks, observations, read_markers CASCADE"
        )
    yield s
    await s.close()


# Per-room monotonic index, mirroring what the framework assigns — so a test
# that adds several events via ``add_event`` gets distinct indices and does not
# trip the UNIQUE(room_id, index) constraint. Reset per test by the fixture.
_index_counters: dict[str, int] = {}


def _make_event(room_id: str = "r1", body: str = "hello", **kwargs):
    from roomkit.models.event import EventSource, RoomEvent, TextContent

    if "index" not in kwargs:
        kwargs["index"] = _index_counters.get(room_id, 0)
        _index_counters[room_id] = kwargs["index"] + 1
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
        content=TextContent(body=body),
        **kwargs,
    )


class TestRoomOperations:
    async def test_create_and_get(self, store) -> None:
        room = Room(id="r1")
        await store.create_room(room)
        result = await store.get_room("r1")
        assert result is not None
        assert result.id == "r1"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_room("nope") is None

    async def test_update_room(self, store) -> None:
        room = Room(id="r1")
        await store.create_room(room)
        room.status = RoomStatus.CLOSED
        await store.update_room(room)
        result = await store.get_room("r1")
        assert result is not None
        assert result.status == RoomStatus.CLOSED

    async def test_an_empty_room_stores_no_latest_index(self, store) -> None:
        await store.create_room(Room(id="r1"))
        async with store._pool.acquire() as conn:
            stored = await conn.fetchval("SELECT latest_index FROM rooms WHERE id = 'r1'")
        assert stored is None

    async def test_init_clears_the_zero_an_empty_room_was_stored_with(self, store) -> None:
        """A deployment from before latest_index was nullable (RFC §5.1)."""
        async with store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO rooms (id, event_count, latest_index) VALUES ('old', 0, 0)"
            )
            await conn.execute("ALTER TABLE rooms ALTER COLUMN latest_index SET NOT NULL")
            await conn.execute("ALTER TABLE rooms ALTER COLUMN latest_index SET DEFAULT 0")
        await store.init(min_size=1, max_size=3)
        async with store._pool.acquire() as conn:
            stored = await conn.fetchval("SELECT latest_index FROM rooms WHERE id = 'old'")
        assert stored is None

    async def test_init_stamps_the_visibility_of_events_stored_before(self, store) -> None:
        """An event stored ``all`` under a narrowed binding takes the
        binding's scope, once (RFC §7.3)."""
        await store.create_room(Room(id="r1"))
        await store.add_binding(
            ChannelBinding(
                room_id="r1", channel_id="ch1", channel_type=ChannelType.SMS, visibility="ws"
            )
        )
        await store.add_event(_make_event(body="under the binding"))
        await store.add_event(_make_event(body="its own scope", visibility="ai"))
        await store.add_event(
            _make_event(
                body="held",
                visibility="transport",
                metadata={HELD_KEY: {"visibility": "all", "identity_id": "i1"}},
            )
        )
        async with store._pool.acquire() as conn:
            await conn.execute("UPDATE events SET visibility = 'all' WHERE visibility = 'ws'")
            await conn.execute("DELETE FROM schema_version WHERE version >= 3")
        await store.init(min_size=1, max_size=3)
        events = await store.list_events("r1")
        assert {e.content.body: e.visibility for e in events} == {
            "under the binding": "ws",
            "its own scope": "ai",
            "held": "transport",
        }
        held = next(e for e in events if e.content.body == "held")
        assert held.metadata[HELD_KEY]["visibility"] == "ws"

    async def test_delivered_index_defaults_to_minus_one(self, store) -> None:
        await store.create_room(Room(id="r1"))
        room = await store.get_room("r1")
        assert room is not None
        assert room.delivered_index == -1

    async def test_advance_delivered_index_cas(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.advance_delivered_index("r1", 0) is True
        assert await store.advance_delivered_index("r1", 1) is True
        assert await store.advance_delivered_index("r1", 3) is False  # gap
        assert await store.advance_delivered_index("r1", 1) is False  # regress
        room = await store.get_room("r1")
        assert room is not None
        assert room.delivered_index == 1

    async def test_advance_delivered_index_force_skips_gap(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.advance_delivered_index("r1", 5, force=True) is True
        assert await store.advance_delivered_index("r1", 3, force=True) is False
        room = await store.get_room("r1")
        assert room is not None
        assert room.delivered_index == 5

    async def test_update_room_never_rewinds_delivered_index(self, store) -> None:
        """delivered_index is absent from update_room's column list."""
        await store.create_room(Room(id="r1"))
        stale = await store.get_room("r1")
        assert stale is not None
        await store.advance_delivered_index("r1", 0)
        await store.update_room(stale.model_copy(update={"status": RoomStatus.CLOSED}))
        room = await store.get_room("r1")
        assert room is not None
        assert room.status == RoomStatus.CLOSED
        assert room.delivered_index == 0

    async def test_patch_room_metadata_merges(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"a": 1, "keep": "x"}))
        result = await store.patch_room_metadata("r1", {"a": 2, "b": 3})
        assert result is not None
        assert result.metadata == {"a": 2, "keep": "x", "b": 3}
        stored = await store.get_room("r1")
        assert stored is not None
        assert stored.metadata == {"a": 2, "keep": "x", "b": 3}

    async def test_patch_room_metadata_unset(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"drop": 1, "keep": 2}))
        result = await store.patch_room_metadata("r1", {"new": 3}, unset=["drop"])
        assert result is not None
        assert result.metadata == {"keep": 2, "new": 3}

    async def test_patch_room_metadata_nonexistent(self, store) -> None:
        assert await store.patch_room_metadata("nope", {"a": 1}) is None

    async def test_patch_room_metadata_does_not_touch_counters(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"seed": True}))
        for i in range(3):
            await store.commit_event("r1", _make_event(body=f"m{i}"))
        await store.patch_room_metadata("r1", {"title": "t"})
        room = await store.get_room("r1")
        assert room is not None
        assert room.event_count == 3
        assert room.latest_index == 2
        assert room.metadata == {"seed": True, "title": "t"}

    async def test_patch_room_metadata_concurrent_with_commits(self, store) -> None:
        """Interleaved patches and commits across pool connections must not
        clobber each other — the read-modify-write regression this API exists
        to prevent. Every patched key and every committed event must survive.
        """
        await store.create_room(Room(id="r1"))
        patches = [store.patch_room_metadata("r1", {f"k{i}": i}) for i in range(10)]
        commits = [store.commit_event("r1", _make_event(body=f"m{i}")) for i in range(10)]
        await asyncio.gather(*patches, *commits)
        room = await store.get_room("r1")
        assert room is not None
        assert room.event_count == 10
        assert room.latest_index == 9
        assert {f"k{i}": i for i in range(10)}.items() <= room.metadata.items()

    async def test_delete_room(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.delete_room("r1") is True
        assert await store.get_room("r1") is None

    async def test_delete_nonexistent(self, store) -> None:
        assert await store.delete_room("nope") is False

    async def test_list_rooms(self, store) -> None:
        for i in range(5):
            await store.create_room(Room(id=f"r{i}"))
        rooms = await store.list_rooms()
        assert len(rooms) == 5

    async def test_list_rooms_pagination(self, store) -> None:
        for i in range(10):
            await store.create_room(Room(id=f"r{i}"))
        page1 = await store.list_rooms(offset=0, limit=3)
        page2 = await store.list_rooms(offset=3, limit=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id


class TestEventOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        event = _make_event(room_id="r1")
        await store.add_event(event)
        result = await store.get_event(event.id)
        assert result is not None
        assert result.id == event.id

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_event("nope") is None

    async def test_list_events(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(5):
            await store.add_event(_make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1")
        assert len(events) == 5

    async def test_list_events_pagination(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event(_make_event(room_id="r1", body=f"msg{i}"))
        page = await store.list_events("r1", offset=2, limit=3)
        assert len(page) == 3

    async def test_idempotency_check(self, store) -> None:
        await store.create_room(Room(id="r1"))
        event = _make_event(room_id="r1", idempotency_key="key1")
        await store.add_event(event)
        assert await store.check_idempotency("r1", "key1", channel_id="ch1") is True
        assert await store.check_idempotency("r1", "key2", channel_id="ch1") is False
        # A key is unique per room and per source channel (RFC §13.4).
        assert await store.check_idempotency("r1", "key1", channel_id="ch2") is False

    async def test_event_count(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.get_event_count("r1") == 0
        await store.add_event(_make_event(room_id="r1"))
        assert await store.get_event_count("r1") == 1

    async def test_update_event_persists_source(self, store) -> None:
        from roomkit.models.enums import ChannelType as CT

        await store.create_room(Room(id="r1"))
        event = _make_event(room_id="r1")
        await store.add_event(event)
        patched = event.model_copy(
            update={
                "source": event.source.model_copy(
                    update={"channel_type": CT.SYSTEM, "participant_id": "task_result"}
                )
            }
        )
        await store.update_event(patched)
        stored = await store.get_event(event.id)
        assert stored is not None
        assert stored.source.channel_type is CT.SYSTEM
        assert stored.source.participant_id == "task_result"

    async def test_commit_after_deleting_latest_keeps_index_monotonic(self, store) -> None:
        await store.create_room(Room(id="r1"))
        first = await store.commit_event("r1", _make_event(room_id="r1", body="first"))
        second = await store.commit_event("r1", _make_event(room_id="r1", body="second"))

        await store.delete_event("r1", second.id)
        third = await store.commit_event("r1", _make_event(room_id="r1", body="third"))

        assert [first.index, second.index, third.index] == [0, 1, 2]
        assert [event.index for event in await store.list_events("r1")] == [0, 2]

    async def test_auto_index_after_deleting_latest_keeps_index_monotonic(self, store) -> None:
        await store.create_room(Room(id="r1"))
        first = await store.add_event_auto_index("r1", _make_event(room_id="r1"))
        second = await store.add_event_auto_index("r1", _make_event(room_id="r1"))

        await store.delete_event("r1", second.id)
        third = await store.add_event_auto_index("r1", _make_event(room_id="r1"))

        assert [first.index, second.index, third.index] == [0, 1, 2]


class TestBindingOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        binding = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        await store.add_binding(binding)
        result = await store.get_binding("r1", "ch1")
        assert result is not None
        assert result.channel_id == "ch1"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_binding("r1", "nope") is None

    async def test_update_binding(self, store) -> None:
        await store.create_room(Room(id="r1"))
        binding = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        await store.add_binding(binding)
        binding.muted = True
        await store.update_binding(binding)
        result = await store.get_binding("r1", "ch1")
        assert result is not None
        assert result.muted is True

    async def test_remove_binding(self, store) -> None:
        await store.create_room(Room(id="r1"))
        binding = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        await store.add_binding(binding)
        assert await store.remove_binding("r1", "ch1") is True
        assert await store.get_binding("r1", "ch1") is None

    async def test_remove_nonexistent(self, store) -> None:
        assert await store.remove_binding("r1", "nope") is False

    async def test_list_bindings(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(3):
            await store.add_binding(
                ChannelBinding(channel_id=f"ch{i}", room_id="r1", channel_type=ChannelType.SMS)
            )
        bindings = await store.list_bindings("r1")
        assert len(bindings) == 3


class TestParticipantOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1")
        await store.add_participant(p)
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.id == "p1"

    async def test_list_participants(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(3):
            await store.add_participant(Participant(id=f"p{i}", room_id="r1", channel_id=f"ch{i}"))
        participants = await store.list_participants("r1")
        assert len(participants) == 3

    async def test_update_participant(self, store) -> None:
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1")
        await store.add_participant(p)
        p.display_name = "Alice"
        await store.update_participant(p)
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.display_name == "Alice"

    async def test_connected_via_round_trips(self, store) -> None:
        """RFC §5.5: the channels a participant was reached through MUST persist.

        A list that only lived in the model would make the multi-channel record
        true for one process and empty for the next one to read it.
        """
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1", connected_via=["ch1"])
        await store.add_participant(p)
        stored = await store.get_participant("r1", "p1")
        assert stored is not None
        assert stored.connected_via == ["ch1"]

        await store.update_participant(p.model_copy(update={"connected_via": ["ch1", "ch2"]}))
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.connected_via == ["ch1", "ch2"]
        # order is order of first sight, not set order
        assert result.connected_via[0] == "ch1"

    @pytest.mark.parametrize("candidates", [None, [], ["alice-id", "bob-id"]])
    async def test_candidates_round_trip(self, store, candidates: list[str] | None) -> None:
        """RFC §11.2: the identities an ambiguous sender could be persist, and
        none offered (NULL) is not none left ([])."""
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1")
        await store.add_participant(p.model_copy(update={"candidates": candidates}))
        stored = await store.get_participant("r1", "p1")
        assert stored is not None
        assert stored.candidates == candidates

        await store.update_participant(p.model_copy(update={"candidates": ["carol-id"]}))
        stored = await store.get_participant("r1", "p1")
        assert stored is not None
        assert stored.candidates == ["carol-id"]

    async def test_connected_via_repairs_a_row_that_predates_it(self, store) -> None:
        """RFC §5.5: an old empty list reads with its primary included."""
        await store.create_room(Room(id="r1"))
        await store.add_participant(Participant(id="p1", room_id="r1", channel_id="ch1"))
        async with store._pool.acquire() as conn:
            await conn.execute(
                "UPDATE participants SET connected_via='{}' WHERE room_id='r1' AND id='p1'"
            )
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.connected_via == ["ch1"]

    async def test_init_backfills_connected_via_on_a_legacy_row(self, store) -> None:
        """The additive schema pass persists the primary on pre-column rows."""
        await store.create_room(Room(id="r1"))
        await store.add_participant(Participant(id="p1", room_id="r1", channel_id="ch1"))
        async with store._pool.acquire() as conn:
            await conn.execute(
                "UPDATE participants SET connected_via='{}' WHERE room_id='r1' AND id='p1'"
            )

        await store.init()

        async with store._pool.acquire() as conn:
            connected_via = await conn.fetchval(
                "SELECT connected_via FROM participants WHERE room_id='r1' AND id='p1'"
            )
        assert connected_via == ["ch1"]


class TestReadTracking:
    async def test_unread_count_initial(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for _ in range(3):
            await store.add_event(_make_event(room_id="r1"))
        count = await store.get_unread_count("r1", "ch1")
        assert count == 3

    async def test_mark_read(self, store) -> None:
        await store.create_room(Room(id="r1"))
        events = []
        for _ in range(3):
            e = _make_event(room_id="r1")
            await store.add_event(e)
            events.append(e)
        await store.mark_read("r1", "ch1", events[1].id)
        count = await store.get_unread_count("r1", "ch1")
        assert count == 1

    async def test_mark_all_read(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for _ in range(3):
            await store.add_event(_make_event(room_id="r1"))
        await store.mark_all_read("r1", "ch1")
        count = await store.get_unread_count("r1", "ch1")
        assert count == 0

    async def test_list_read_markers(self, store) -> None:
        await store.create_room(Room(id="r1"))
        events = []
        for _ in range(3):
            e = _make_event(room_id="r1")
            await store.add_event(e)
            events.append(e)
        await store.mark_read("r1", "ch1", events[2].id)
        await store.mark_read("r1", "ch2", events[0].id)
        markers = await store.list_read_markers("r1")
        assert set(markers) == {"ch1", "ch2"}
        # ch1 read further than ch2; a channel with no marker is absent
        assert markers["ch1"] > markers["ch2"]
        assert "ch3" not in markers


class TestIdentityOperations:
    async def test_create_and_get(self, store) -> None:
        identity = Identity(
            id="id1",
            display_name="Alice",
            channel_addresses={"sms": ["+15551234567"]},
        )
        await store.create_identity(identity)
        result = await store.get_identity("id1")
        assert result is not None
        assert result.display_name == "Alice"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_identity("nope") is None

    async def test_resolve_by_address(self, store) -> None:
        identity = Identity(
            id="id1",
            channel_addresses={"sms": ["+15551234567"]},
        )
        await store.create_identity(identity)
        result = await store.resolve_identity("sms", "+15551234567")
        assert result is not None
        assert result.id == "id1"

    async def test_resolve_unknown_address(self, store) -> None:
        result = await store.resolve_identity("sms", "+10000000000")
        assert result is None

    async def test_link_address(self, store) -> None:
        identity = Identity(id="id1")
        await store.create_identity(identity)
        await store.link_address("id1", "email", "alice@example.com")

        result = await store.resolve_identity("email", "alice@example.com")
        assert result is not None
        assert result.id == "id1"

        fetched = await store.get_identity("id1")
        assert fetched is not None
        assert "alice@example.com" in fetched.channel_addresses.get("email", [])

    async def test_multiple_addresses_same_channel(self, store) -> None:
        identity = Identity(
            id="id1",
            channel_addresses={"sms": ["+15551111111", "+15552222222"]},
        )
        await store.create_identity(identity)
        r1 = await store.resolve_identity("sms", "+15551111111")
        r2 = await store.resolve_identity("sms", "+15552222222")
        assert r1 is not None and r1.id == "id1"
        assert r2 is not None and r2.id == "id1"


class TestIdentityDetails:
    async def test_what_an_identity_holds_beside_its_addresses_round_trips(self, store) -> None:
        """RFC §5.6: as the other stores keep them, through get and resolve."""
        identity = Identity(
            id="i1",
            phone="+15551234567",
            external_id="crm_42",
            external_ids={"crm": "42", "erp": "77"},
            channel_addresses={"sms": ["+15551234567"]},
        )
        await store.create_identity(identity)
        assert await store.get_identity("i1") == identity
        assert await store.resolve_identity("sms", "+15551234567") == identity

    async def test_an_identity_created_again_replaces_the_stored_one(self, store) -> None:
        await store.create_identity(Identity(id="i1", phone="+1555", external_id="crm_42"))
        renamed = Identity(id="i1", display_name="Alice B.")
        assert await store.create_identity(renamed) == renamed
        assert await store.get_identity("i1") == renamed


class TestTaskOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        task = Task(id="t1", room_id="r1", title="Follow up")
        await store.add_task(task)
        result = await store.get_task("t1")
        assert result is not None
        assert result.title == "Follow up"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_task("nope") is None

    async def test_list_tasks_by_room(self, store) -> None:
        await store.create_room(Room(id="r1"))
        await store.create_room(Room(id="r2"))
        await store.add_task(Task(id="t1", room_id="r1", title="Task 1"))
        await store.add_task(Task(id="t2", room_id="r1", title="Task 2"))
        await store.add_task(Task(id="t3", room_id="r2", title="Task 3"))
        tasks = await store.list_tasks("r1")
        assert len(tasks) == 2

    async def test_list_tasks_by_status(self, store) -> None:
        await store.create_room(Room(id="r1"))
        await store.add_task(Task(id="t1", room_id="r1", title="Pending", status="pending"))
        await store.add_task(Task(id="t2", room_id="r1", title="Done", status="completed"))
        pending = await store.list_tasks("r1", status="pending")
        assert len(pending) == 1
        assert pending[0].title == "Pending"

    async def test_update_task(self, store) -> None:
        await store.create_room(Room(id="r1"))
        task = Task(id="t1", room_id="r1", title="Follow up")
        await store.add_task(task)
        task.status = "completed"
        await store.update_task(task)
        result = await store.get_task("t1")
        assert result is not None
        assert result.status == "completed"


class TestObservationOperations:
    async def test_add_and_list(self, store) -> None:
        await store.create_room(Room(id="r1"))
        obs = Observation(
            id="obs1",
            room_id="r1",
            channel_id="ai1",
            content="sentiment positive",
            category="sentiment",
        )
        await store.add_observation(obs)
        observations = await store.list_observations("r1")
        assert len(observations) == 1
        assert observations[0].category == "sentiment"

    async def test_list_by_room(self, store) -> None:
        await store.create_room(Room(id="r1"))
        await store.create_room(Room(id="r2"))
        await store.add_observation(
            Observation(id="obs1", room_id="r1", channel_id="ai1", content="a")
        )
        await store.add_observation(
            Observation(id="obs2", room_id="r2", channel_id="ai1", content="b")
        )
        r1_obs = await store.list_observations("r1")
        r2_obs = await store.list_observations("r2")
        assert len(r1_obs) == 1
        assert len(r2_obs) == 1


class TestFindRooms:
    async def test_find_by_status(self, store) -> None:
        await store.create_room(Room(id="r1", status=RoomStatus.ACTIVE))
        await store.create_room(Room(id="r2", status=RoomStatus.CLOSED))
        await store.create_room(Room(id="r3", status=RoomStatus.ACTIVE))
        rooms = await store.find_rooms(status=RoomStatus.ACTIVE)
        assert len(rooms) == 2
        assert all(r.status == RoomStatus.ACTIVE for r in rooms)

    async def test_find_by_metadata(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"org": "acme", "tier": "premium"}))
        await store.create_room(Room(id="r2", metadata={"org": "globex", "tier": "basic"}))
        rooms = await store.find_rooms(metadata_filter={"org": "acme"})
        assert len(rooms) == 1
        assert rooms[0].id == "r1"

    async def test_find_no_match(self, store) -> None:
        await store.create_room(Room(id="r1"))
        rooms = await store.find_rooms(status=RoomStatus.CLOSED)
        assert len(rooms) == 0


class TestListEventsVisibilityFilter:
    async def test_visibility_filter(self, store) -> None:
        await store.create_room(Room(id="r1"))
        e1 = _make_event(room_id="r1", body="public", visibility="all")
        e2 = _make_event(room_id="r1", body="internal", visibility="internal")
        e3 = _make_event(room_id="r1", body="public2", visibility="all")
        await store.add_event(e1)
        await store.add_event(e2)
        await store.add_event(e3)

        all_events = await store.list_events("r1")
        assert len(all_events) == 3

        public_only = await store.list_events("r1", visibility_filter="all")
        assert len(public_only) == 2

        internal_only = await store.list_events("r1", visibility_filter="internal")
        assert len(internal_only) == 1


class TestCursorPagination:
    async def test_after_index(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(5):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", after_index=1, limit=50)
        assert len(events) == 3
        assert all(e.index > 1 for e in events)

    async def test_before_index(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(5):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", before_index=3, limit=50)
        assert len(events) == 3
        assert all(e.index < 3 for e in events)
        assert [e.index for e in events] == [0, 1, 2]

    async def test_before_index_returns_last_n(self, store) -> None:
        """before_index returns the *last* limit events before cursor."""
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", before_index=8, limit=3)
        assert len(events) == 3
        assert [e.index for e in events] == [5, 6, 7]

    async def test_after_index_with_limit(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", after_index=2, limit=3)
        assert len(events) == 3
        assert events[0].index == 3

    async def test_mutually_exclusive(self, store) -> None:
        await store.create_room(Room(id="r1"))
        with pytest.raises(ValueError, match="mutually exclusive"):
            await store.list_events("r1", after_index=0, before_index=5)

    async def test_cursor_with_visibility_filter(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(6):
            vis = "all" if i % 2 == 0 else "internal"
            await store.add_event_auto_index(
                "r1", _make_event(room_id="r1", body=f"msg{i}", visibility=vis)
            )
        events = await store.list_events("r1", after_index=0, visibility_filter="all")
        assert len(events) == 2
        assert all(e.visibility == "all" for e in events)

    async def test_empty_result(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(3):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", after_index=10)
        assert events == []


class TestConversationRecency:
    """RMK-99 — against a real database: SQL decides this ordering, not Python."""

    async def test_get_conversation_returns_the_tail(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))

        convo = await store.get_conversation("r1", limit=3)

        assert [e.index for e in convo] == [7, 8, 9]
        assert convo[-1].content.body == "msg9"

    async def test_get_conversation_with_cursor_reads_forward(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))

        page = await store.get_conversation("r1", limit=3, after_index=5)

        assert [e.index for e in page] == [6, 7, 8]


class TestDeleteRoomCleanup:
    async def test_cascading_delete(self, store) -> None:
        """ON DELETE CASCADE removes all child rows."""
        await store.create_room(Room(id="r1"))
        await store.add_event(_make_event(room_id="r1", idempotency_key="k1"))
        await store.add_task(Task(id="t1", room_id="r1", title="T"))
        await store.add_observation(
            Observation(id="obs1", room_id="r1", channel_id="c1", content="x")
        )
        await store.add_binding(
            ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        )
        await store.add_participant(Participant(id="p1", room_id="r1", channel_id="ch1"))

        assert await store.delete_room("r1") is True

        assert await store.get_room("r1") is None
        assert await store.get_event_count("r1") == 0
        assert await store.list_tasks("r1") == []
        assert await store.list_observations("r1") == []
        assert await store.list_bindings("r1") == []
        assert await store.list_participants("r1") == []


@pytest.fixture
def checkouts(monkeypatch) -> Callable[[Any], PoolCheckoutRecorder]:
    """Return ``recorder_for(store)`` — real pool checkouts, per store instance."""
    from roomkit.store.postgres import PostgresStore

    recorders: dict[int, PoolCheckoutRecorder] = {}

    def recorder_for(s: Any) -> PoolCheckoutRecorder:
        if id(s) not in recorders:
            recorders[id(s)] = PoolCheckoutRecorder(s._pool)
        return recorders[id(s)]

    monkeypatch.setattr(PostgresStore, "_ensure_pool", recorder_for)
    return recorder_for


class TestConnectionTenure:
    """RMK-97 — ``connection()`` bounds how long a pooled connection is held.

    asyncpg resets a connection on release, so every checkout is a full extra
    round trip; on the inbound path that overhead is the same order as the
    reads it brackets.
    """

    async def test_the_block_pays_one_checkout_for_every_call(self, store, checkouts) -> None:
        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        async with store.connection():
            await store.get_room("r1")
            await store.list_bindings("r1")
            await store.list_participants("r1")

        assert recorder.total == 1

    async def test_the_same_calls_outside_pay_one_checkout_each(self, store, checkouts) -> None:
        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        await store.get_room("r1")
        await store.list_bindings("r1")
        await store.list_participants("r1")

        assert recorder.total == 3

    async def test_a_nested_block_joins_the_outer_one(self, store, checkouts) -> None:
        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        async with store.connection():
            await store.get_room("r1")
            async with store.connection():
                await store.list_bindings("r1")
            await store.list_participants("r1")

        assert recorder.total == 1

    async def test_the_default_load_room_context_rides_the_block(self, store, checkouts) -> None:
        """The ABC default composes three getters and brackets them itself.

        ``PostgresStore`` overrides the method, so call the default explicitly:
        this is what a third-party pooled backend inherits without writing any
        SQL of its own.
        """
        from roomkit.store.base import ConversationStore

        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        room, bindings, participants = await ConversationStore.load_room_context(store, "r1")

        assert room is not None
        assert (bindings, participants) == ([], [])
        assert recorder.total == 1

    async def test_a_failure_inside_the_block_returns_the_connection(self, store) -> None:
        await store.get_room("warm-the-pool")
        idle_before = store._pool.get_idle_size()

        with pytest.raises(RuntimeError, match="boom"):
            async with store.connection():
                await store.get_room("r1")
                raise RuntimeError("boom")

        assert store._pool.get_idle_size() == idle_before
        assert await store.get_room("r1") is None

    async def test_concurrent_blocks_do_not_share_one_connection(self, store) -> None:
        """Two tasks, two connections — a shared one would raise on the second
        query ("another operation is in progress")."""
        await store.create_room(Room(id="r1"))
        seen: list[int] = []

        async def read_twice() -> None:
            async with store.connection():
                seen.append(await store.get_event_count("r1"))
                await asyncio.sleep(0)  # hand over inside the block
                seen.append(await store.get_event_count("r1"))

        await asyncio.gather(read_twice(), read_twice())

        assert seen == [0, 0, 0, 0]

    async def test_another_store_is_not_lent_this_one_s_connection(self, store, checkouts) -> None:
        """The binding carries its owner: two stores may hold two pools onto
        two different databases, and lending across them would query the wrong
        one."""
        from roomkit.store.postgres import PostgresStore

        other = PostgresStore(dsn=POSTGRES_DSN)
        await other.init(min_size=1, max_size=2)
        try:
            await store.create_room(Room(id="r1"))
            recorder = checkouts(other)
            recorder.reset()

            async with store.connection():
                assert await other.get_room("r1") is not None

            assert recorder.total == 1
        finally:
            await other.close()


class TestContextManager:
    async def test_async_context_manager(self) -> None:
        from roomkit.store.postgres import PostgresStore

        store = PostgresStore(dsn=POSTGRES_DSN)
        async with store as s:
            room = Room(id="ctx-test")
            await s.create_room(room)
            result = await s.get_room("ctx-test")
            assert result is not None


class TestThreadQueries:
    """Flat two-level threading — the ``parent_event_id`` filters + summaries."""

    async def test_top_level_only_excludes_replies(self, store) -> None:
        from roomkit.models.store_filter import EventFilter

        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        await store.add_event(_make_event(body="standalone"))
        await store.add_event(_make_event(body="reply1", parent_event_id=root.id))
        await store.add_event(_make_event(body="reply2", parent_event_id=root.id))

        top = await store.list_events("r1", event_filter=EventFilter(top_level_only=True))
        assert {e.content.body for e in top} == {"root", "standalone"}

    async def test_parent_event_id_returns_only_that_thread(self, store) -> None:
        from roomkit.models.store_filter import EventFilter

        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        await store.add_event(_make_event(body="reply1", parent_event_id=root.id))
        await store.add_event(_make_event(body="reply2", parent_event_id=root.id))

        replies = await store.list_events("r1", event_filter=EventFilter(parent_event_id=root.id))
        assert {e.content.body for e in replies} == {"reply1", "reply2"}
        assert all(e.parent_event_id == root.id for e in replies)

    async def test_thread_summaries_counts_and_last_reply(self, store) -> None:
        from datetime import UTC, datetime

        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        await store.add_event(
            _make_event(
                body="r1",
                parent_event_id=root.id,
                created_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
            )
        )
        last = _make_event(
            body="r2",
            parent_event_id=root.id,
            created_at=datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC),
        )
        await store.add_event(last)

        summaries = await store.get_thread_summaries("r1", [root.id])
        assert set(summaries) == {root.id}
        assert summaries[root.id].reply_count == 2
        assert summaries[root.id].last_reply_at == last.created_at

    async def test_thread_summaries_empty_cases(self, store) -> None:
        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        assert await store.get_thread_summaries("r1", [root.id]) == {}
        assert await store.get_thread_summaries("r1", []) == {}


async def test_a_confidence_reads_as_written(store) -> None:
    await store.create_room(Room(id="r1"))
    written = Observation(id="o1", room_id="r1", channel_id="ch1", content="c", confidence=0.42)
    await store.add_observation(written)
    assert await store.list_observations("r1") == [written]


async def test_init_widens_the_confidence_of_a_store_from_before(store) -> None:
    """A value a REAL column holds reads as written once widened, through its
    shortest digits, not as the nearest double of the single."""
    await store.create_room(Room(id="r1"))
    written = Observation(id="o1", room_id="r1", channel_id="ch1", content="c", confidence=0.42)
    await store.add_observation(written)
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE observations ALTER COLUMN confidence TYPE REAL")
    await store.init()
    await store.init()
    assert await store.list_observations("r1") == [written]


async def test_init_adds_what_an_identity_holds_to_a_table_without_it(store) -> None:
    await store.create_identity(Identity(id="i1", display_name="Alice"))
    async with store._pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE identities DROP COLUMN phone, DROP COLUMN external_id,"
            " DROP COLUMN external_ids"
        )
    await store.init()
    await store.init()
    assert await store.get_identity("i1") == Identity(id="i1", display_name="Alice")


async def test_binding_policy_migration_preserves_old_rows(store) -> None:
    from roomkit.models.channel import RateLimit, RetryPolicy

    await store.create_room(Room(id="r1"))
    binding = ChannelBinding(room_id="r1", channel_id="sms", channel_type="sms")
    await store.add_binding(binding)
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE bindings DROP COLUMN rate_limit, DROP COLUMN retry_policy")
    # Reinitializing the store upgrades the old schema, including repeated runs.
    await store.init()
    await store.init()
    assert await store.get_binding("r1", "sms") == binding
    changed = binding.model_copy(
        update={
            "rate_limit": RateLimit(max_per_minute=30),
            "retry_policy": RetryPolicy(max_retries=7),
        }
    )
    await store.update_binding(changed)
    assert await store.get_binding("r1", "sms") == changed


async def test_init_creates_each_index_it_lacks_and_leaves_the_others_be(store) -> None:
    """init() creates every index of its schema that is missing, as it
    defines it (a unique one unique, a partial one partial, the thread
    index composite), and finds an index present as it is."""
    from roomkit.store.postgres_schema import V1_TABLES

    tables = [*V1_TABLES, "verifications", "verification_requests", "voice_deliveries"]
    indexes = (
        "SELECT indexname, indexdef FROM pg_indexes"
        " WHERE schemaname = current_schema() AND tablename = ANY($1::text[])"
    )
    async with store._pool.acquire() as conn:
        before = {row["indexname"]: row["indexdef"] for row in await conn.fetch(indexes, tables)}
        for name in before:
            if name.startswith("idx_"):
                await conn.execute(f'DROP INDEX "{name}"')
    await store.init()
    await store.init()
    async with store._pool.acquire() as conn:
        after = {row["indexname"]: row["indexdef"] for row in await conn.fetch(indexes, tables)}
    assert after == before
    assert "(parent_event_id, index) WHERE" in after["idx_events_parent_index"]


async def test_the_legacy_parent_index_is_dropped_on_request_only(store) -> None:
    """init() never drops: the single-column idx_events_parent the composite
    supersedes stays until drop_legacy_parent_index() is asked, which a dry
    run only reports."""
    legacy = "SELECT 1 FROM pg_indexes WHERE schemaname = current_schema() AND indexname = $1"
    async with store._pool.acquire() as conn:
        await conn.execute("CREATE INDEX idx_events_parent ON events(parent_event_id)")
    try:
        await store.init()
        dry = await store.drop_legacy_parent_index()
        async with store._pool.acquire() as conn:
            kept = await conn.fetchval(legacy, "idx_events_parent")
        dropped = await store.drop_legacy_parent_index(dry_run=False)
        again = await store.drop_legacy_parent_index(dry_run=False)
        async with store._pool.acquire() as conn:
            gone = await conn.fetchval(legacy, "idx_events_parent") is None
    finally:
        async with store._pool.acquire() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_events_parent")
    assert (dry, dropped, again) == (
        {"action": "dry_run"},
        {"action": "dropped"},
        {"action": "noop"},
    )
    assert kept == 1 and gone


async def test_init_seeds_the_index_marks_of_a_store_from_before_them(store) -> None:
    """A store from before the per-room high-water marks gains them at its
    first init(), seeded from its rooms and events, so the next commit takes
    the next index."""
    await store.create_room(Room(id="r1"))
    await store.create_room(Room(id="r2"))
    for _ in range(3):
        await store.commit_event("r1", _make_event())
    async with store._pool.acquire() as conn:
        await conn.execute("DROP TABLE event_sequences")
    await store.init()

    committed = await store.commit_event("r1", _make_event())

    assert committed.index == 3
    async with store._pool.acquire() as conn:
        seeded = await conn.fetch("SELECT room_id, next_index FROM event_sequences")
    assert {(row["room_id"], row["next_index"]) for row in seeded} == {("r1", 4), ("r2", 0)}


async def test_a_store_from_before_group_bindings_upgrades_its_rows(store) -> None:
    """The additive ``is_group`` column (RFC §5.7): an existing binding reads as
    not a group, and a group binding round-trips once the column is there."""
    await store.create_room(Room(id="r1"))
    binding = ChannelBinding(room_id="r1", channel_id="sms", channel_type="sms")
    await store.add_binding(binding)
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE bindings DROP COLUMN is_group")
    await store.init()
    await store.init()
    assert await store.get_binding("r1", "sms") == binding
    grouped = binding.model_copy(update={"group": True})
    await store.update_binding(grouped)
    assert await store.get_binding("r1", "sms") == grouped


class TestEventFromRow:
    """A row the host reads with a query of its own maps as the store maps it."""

    async def test_a_raw_row_is_the_event_get_event_reads(self, store) -> None:
        from roomkit.store.postgres import PostgresStore

        await store.create_room(Room(id="r1"))
        committed = await store.commit_event("r1", _make_event(body="from a page"))
        async with store._pool.acquire() as conn:
            # A host's page query: every column of events, plus one of its own.
            row = await conn.fetchrow(
                "SELECT events.*, 'page-1' AS page FROM events WHERE id = $1", committed.id
            )

        assert PostgresStore.event_from_row(row) == await store.get_event(committed.id)


# -- RFC §8.5: an answer names the event it answers ---------------------------


async def test_an_answer_keeps_and_finds_the_event_it_answers(store) -> None:
    from roomkit.models.store_filter import EventFilter

    await store.create_room(Room(id="r1"))
    asked = await store.add_event(_make_event(body="la météo ?"))
    answer = await store.add_event(_make_event(body="8 degrés", responds_to=asked.id))
    await store.add_event(_make_event(body="autre chose"))

    stored = await store.get_event(answer.id)
    answers = await store.list_events(
        "r1", offset=0, limit=10, event_filter=EventFilter(responds_to=asked.id)
    )
    assert stored is not None and stored.responds_to == asked.id
    assert [e.id for e in answers] == [answer.id]


@pytest.mark.parametrize("column", ["addressed_to", "responds_to", "delivery_results"])
async def test_init_adds_the_column_to_an_events_table_without_it(store, column: str) -> None:
    from roomkit.store.postgres import PostgresStore

    await store.create_room(Room(id="r1"))
    old = await store.add_event(_make_event(body="avant"))
    async with store._pool.acquire() as conn:
        await conn.execute("DROP INDEX IF EXISTS idx_events_room_responds_to")
        await conn.execute(f"ALTER TABLE events DROP COLUMN {column}")

    upgraded = PostgresStore(dsn=POSTGRES_DSN)
    await upgraded.init(min_size=1, max_size=2)
    try:
        kept = await upgraded.get_event(old.id)
        async with upgraded._pool.acquire() as conn:
            found = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns"
                " WHERE table_schema = current_schema()"
                " AND table_name = 'events' AND column_name = $1",
                column,
            )
    finally:
        await upgraded.close()
    assert found == 1
    assert kept is not None
    assert (kept.addressed_to, kept.responds_to, kept.delivery_results) == (None, None, {})


async def test_init_records_when_a_room_stored_paused_before_paused_at_paused(store) -> None:
    """RFC §5.1: a store from before ``paused_at`` gains the column, and a
    room stored paused takes its last modification as the instant it paused."""
    from roomkit.models.enums import RoomStatus
    from roomkit.store.postgres import PostgresStore

    paused = await store.create_room(Room(id="r1", status=RoomStatus.PAUSED))
    await store.create_room(Room(id="r2"))
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE rooms DROP COLUMN paused_at")

    upgraded = PostgresStore(dsn=POSTGRES_DSN)
    await upgraded.init(min_size=1, max_size=2)
    try:
        was_paused = await upgraded.get_room("r1")
        active = await upgraded.get_room("r2")
    finally:
        await upgraded.close()
    assert was_paused is not None and active is not None
    assert was_paused.paused_at == paused.updated_at
    assert active.paused_at is None


async def test_init_moves_the_author_registers_to_the_private_state(store) -> None:
    """RMK-631: a store from before keeps a room's register of authors in
    its metadata, which clients may read; init() moves it to the room's
    private state, once, and drops a malformed one. A copy that is not what it moved
    (a process from before wrote it meanwhile) stays for the room's next
    turn to merge: dropped, its ranks would be given again."""
    from roomkit.store.postgres import PostgresStore

    register = {"version": 3, "names": {"a" * 16: "b" * 16 + ":1"}}
    later = {"version": 3, "names": {"a" * 16: "b" * 16 + ":1 " + "c" * 16 + ":2"}}
    await store.create_room(Room(id="r1", metadata={"topic": "x", "author_register": register}))
    await store.create_room(Room(id="r2", metadata={"author_register": "junk"}))
    await store.create_room(Room(id="r3", metadata={"author_register": later}))
    async with store._pool.acquire() as conn:
        await conn.execute("DELETE FROM schema_version WHERE version >= 4")
        await conn.execute(
            "INSERT INTO private_state (room_id, key, value)"
            " VALUES ('r3', 'author_register', $1::jsonb)",
            register,
        )

    upgraded = PostgresStore(dsn=POSTGRES_DSN)
    await upgraded.init(min_size=1, max_size=2)
    try:
        moved, kept = (
            await upgraded.get_private_state("r1", "author_register"),
            await upgraded.get_room("r1"),
        )
        junk, emptied = (
            await upgraded.get_private_state("r2", "author_register"),
            await upgraded.get_room("r2"),
        )
        held, pending = (
            await upgraded.get_private_state("r3", "author_register"),
            await upgraded.get_room("r3"),
        )
    finally:
        await upgraded.close()
    assert moved == register
    assert kept is not None and kept.metadata == {"topic": "x"}
    assert junk is None
    assert emptied is not None and emptied.metadata == {}
    assert held == register
    assert pending is not None and pending.metadata == {"author_register": later}


async def test_a_private_state_write_racing_the_room_s_deletion_writes_nothing(store) -> None:
    """RMK-631: a write of a room's private state while the room is being
    deleted waits for the deletion, then writes nothing, as for a room the
    store does not hold, rather than raise the foreign key's violation."""
    import asyncpg

    await store.create_room(Room(id="r1"))
    deleter = await asyncpg.connect(POSTGRES_DSN)
    try:
        async with deleter.transaction():
            await deleter.execute("DELETE FROM rooms WHERE id = 'r1'")
            write = asyncio.create_task(
                store.set_private_state("r1", "author_register", {"version": 3, "names": {}})
            )
            await _until_blocked_by(deleter)
        await asyncio.wait_for(write, 2)
    finally:
        await deleter.close()
    assert await store.get_private_state("r1", "author_register") is None


async def _until_blocked_by(holder: Any) -> None:
    """Return once a session waits on a lock *holder*'s transaction holds."""
    deadline = asyncio.get_running_loop().time() + 2
    blocked = "SELECT count(*) FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid))"
    async with asyncio.timeout_at(deadline):
        while not await holder.fetchval(blocked, holder.get_server_pid()):
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("write", ["commit_event", "add_event", "add_event_auto_index"])
async def test_a_held_id_is_the_driver_s_refusal_whatever_the_index_order(
    store, write: str
) -> None:
    """RFC §14.3: a second write of a stored event, its key held with it, is
    the database's refusal of the id (``UniqueViolationError``), and another
    event under the key is a taken key, whichever unique index Postgres
    checks first: after ``REINDEX ... CONCURRENTLY`` the primary key is the
    newest index, checked last."""
    import asyncpg

    from roomkit.core.exceptions import IdempotencyKeyTakenError

    async with store._pool.acquire() as conn:
        await conn.execute("REINDEX INDEX CONCURRENTLY events_pkey")
    await store.create_room(Room(id="r1"))
    held = await store.commit_event("r1", _make_event(idempotency_key="k"))
    writes = {
        "commit_event": lambda event: store.commit_event("r1", event),
        "add_event": lambda event: store.add_event(event.model_copy(update={"index": 99})),
        "add_event_auto_index": lambda event: store.add_event_auto_index("r1", event),
    }

    with pytest.raises(asyncpg.UniqueViolationError):
        await writes[write](held)
    with pytest.raises(IdempotencyKeyTakenError):
        await writes[write](_make_event(idempotency_key="k"))
    assert [e.id for e in await store.list_events("r1")] == [held.id]


# Open transactions a booting worker meets: readers of the tables init()
# once added columns to, a writer of a room, and a commit reserving an index.
_OPEN_TRANSACTIONS = {
    **{
        f"reader of {table}": f"SELECT count(*) FROM {table}"
        for table in ("events", "rooms", "bindings", "participants", "identity_addresses")
    },
    "writer of a room": "UPDATE rooms SET updated_at = now() WHERE id = 'r1'",
    "commit reserving an index": (
        "UPDATE event_sequences SET next_index = next_index + 1 WHERE room_id = 'r1'"
    ),
}


@pytest.mark.parametrize("holder", list(_OPEN_TRANSACTIONS))
async def test_a_second_init_waits_on_no_open_transaction(store, holder: str) -> None:
    """``init()`` on an up-to-date schema takes no lock an open transaction
    on its tables holds against it: no ``ALTER TABLE`` (ACCESS EXCLUSIVE,
    even when ``IF NOT EXISTS`` finds the column), no ``CREATE INDEX``
    (SHARE, which waits for every writer, even when ``IF NOT EXISTS`` finds
    the index), and no write of a row a commit holds."""
    import asyncpg

    from roomkit.store.postgres import PostgresStore

    await store.create_room(Room(id="r1"))
    await store.commit_event("r1", _make_event())
    other = await asyncpg.connect(POSTGRES_DSN)
    pool = await asyncpg.create_pool(
        POSTGRES_DSN, min_size=1, max_size=1, server_settings={"lock_timeout": "500ms"}
    )
    try:
        async with other.transaction():
            await other.execute(_OPEN_TRANSACTIONS[holder])
            await PostgresStore(pool=pool).init()
    finally:
        await pool.close()
        await other.close()


# What an older deployment left in its schema: a v1 ``rooms.data`` column,
# the unique idx_participants_channel, a non-unique idx_events_room_index, the
# two-column identity key, the single-column parent index and the v0.107.0
# idempotency index.
_AN_OLDER_DEPLOYMENT = (
    "ALTER TABLE rooms ADD COLUMN data JSONB",
    "DROP INDEX idx_participants_channel",
    "CREATE UNIQUE INDEX idx_participants_channel ON participants(room_id, channel_id)",
    "DROP INDEX idx_events_room_index",
    "CREATE INDEX idx_events_room_index ON events(room_id, index)",
    "ALTER TABLE identity_addresses DROP CONSTRAINT identity_addresses_pkey,"
    " ADD PRIMARY KEY (channel_type, address)",
    "CREATE INDEX idx_events_parent ON events(parent_event_id)",
    "CREATE UNIQUE INDEX idx_events_idempotency ON events(room_id, idempotency_key)"
    " WHERE idempotency_key IS NOT NULL",
)

_GUARDED_INDEXES = [
    "idx_events_idempotency",
    "idx_events_room_idempotency",
    "idx_participants_identity",
    "idx_participants_channel",
    "idx_events_room_index",
    "idx_events_parent",
]


async def _identity_key(conn: Any, schema: str) -> int:
    return await conn.fetchval(
        "SELECT i.indexrelid::int FROM pg_index i"
        " WHERE i.indrelid = to_regclass(format('%I.%I', $1::text, 'identity_addresses'))"
        " AND i.indisprimary",
        schema,
    )


@pytest.fixture
async def second_schema(store) -> Any:
    """A second deployment's schema in the store's database, and a pool whose
    ``search_path`` is ``tenant, <the store's schema>``: the store's own
    deployment shows through it to every name the tenant does not hold."""
    import asyncpg

    tenant = f"tenant_{os.getpid()}"
    async with store._pool.acquire() as conn:
        home = await conn.fetchval("SELECT current_schema()")
        await conn.execute(f'CREATE SCHEMA "{tenant}"')
    pool = await asyncpg.create_pool(
        POSTGRES_DSN, min_size=1, max_size=1, server_settings={"search_path": f"{tenant},{home}"}
    )
    try:
        yield home, tenant, pool
    finally:
        await pool.close()
        async with store._pool.acquire() as conn:
            await conn.execute(f'DROP SCHEMA "{tenant}" CASCADE')


async def test_init_in_a_second_schema_keeps_each_schema_its_own_indexes(
    store, second_schema, caplog: pytest.LogCaptureFixture
) -> None:
    """Two deployments in one database, one per schema: ``init()`` with a
    ``search_path`` of ``tenant, <first schema>`` reads and writes the
    tenant's schema only. The first schema holds what an older deployment
    left: the tenant is not taken for a v1 schema, creates its own indexes,
    leaves the first deployment's as they are, logs no degraded index of
    another schema, and does not rebuild its identity key at each boot; its
    maintenance calls read and drop its own indexes only."""
    from roomkit.store.postgres import PostgresStore

    home, tenant, pool = second_schema
    async with store._pool.acquire() as conn:
        for statement in _AN_OLDER_DEPLOYMENT:
            await conn.execute(statement)
    try:
        other = PostgresStore(pool=pool)
        with caplog.at_level("WARNING", logger="roomkit.store.postgres"):
            await other.init()
        async with pool.acquire() as conn:
            key = await _identity_key(conn, tenant)
        await other.init()
        async with pool.acquire() as conn:
            key_again = await _identity_key(conn, tenant)
        report = await other.dedupe_event_indices()
        dropped = await other.drop_legacy_parent_index(dry_run=False)
        async with store._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT schemaname, indexname, indexdef LIKE '%UNIQUE%' AS is_unique"
                " FROM pg_indexes"
                " WHERE schemaname = ANY($1::text[]) AND indexname = ANY($2::text[])",
                [home, tenant],
                _GUARDED_INDEXES,
            )
    finally:
        async with store._pool.acquire() as conn:
            await conn.execute("ALTER TABLE rooms DROP COLUMN IF EXISTS data")
            await conn.execute("DROP INDEX IF EXISTS idx_events_parent")
        await store.init()
    assert {(row["schemaname"], row["indexname"], row["is_unique"]) for row in rows} == {
        (home, "idx_events_idempotency", True),
        (home, "idx_events_room_idempotency", False),
        (home, "idx_participants_identity", False),
        (home, "idx_participants_channel", True),
        (home, "idx_events_room_index", False),
        (home, "idx_events_parent", False),
        (tenant, "idx_events_room_idempotency", False),
        (tenant, "idx_participants_identity", False),
        (tenant, "idx_participants_channel", False),
        (tenant, "idx_events_room_index", True),
    }
    assert not [r for r in caplog.records if "idx_events_room_index" in r.getMessage()]
    assert key == key_again
    assert report["now_unique"] is True
    assert dropped == {"action": "noop"}


async def test_migrate_in_a_second_schema_drops_its_own_tables_only(store, second_schema) -> None:
    """The destructive v1 → v2 ``migrate()`` drops the tables of the schema
    its session writes to: a table the v1 tenant lacks is not taken from the
    other deployment's schema through ``search_path``."""
    from roomkit.store.postgres import PostgresStore

    home, _tenant, pool = second_schema
    other = PostgresStore(pool=pool)
    await other.init()
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE rooms ADD COLUMN data JSONB")
        await conn.execute("DROP TABLE read_markers")

    report = await other.migrate(dry_run=False, confirm=True)

    async with store._pool.acquire() as conn:
        kept = await conn.fetchval(
            "SELECT to_regclass(format('%I.%I', $1::text, 'read_markers')) IS NOT NULL", home
        )
    assert report["action"] == "migrated"
    assert kept
