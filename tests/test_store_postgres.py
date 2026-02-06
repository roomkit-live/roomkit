"""Tests for PostgresStore.

These tests require a running PostgreSQL instance. Set POSTGRES_DSN to run them:

    POSTGRES_DSN=postgresql://user:pass@localhost/roomkit_test \
    pytest tests/test_store_postgres.py -v
"""

from __future__ import annotations

import os

import pytest

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set"),
    pytest.mark.asyncio,
]


@pytest.fixture
async def store():
    from roomkit.store.postgres import PostgresStore

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


def _make_event(room_id: str = "r1", body: str = "hello", **kwargs):
    from roomkit.models.event import EventSource, RoomEvent, TextContent

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
        assert await store.check_idempotency("r1", "key1") is True
        assert await store.check_idempotency("r1", "key2") is False

    async def test_event_count(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.get_event_count("r1") == 0
        await store.add_event(_make_event(room_id="r1"))
        assert await store.get_event_count("r1") == 1


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


class TestContextManager:
    async def test_async_context_manager(self) -> None:
        from roomkit.store.postgres import PostgresStore

        store = PostgresStore(dsn=POSTGRES_DSN)
        async with store as s:
            room = Room(id="ctx-test")
            await s.create_room(room)
            result = await s.get_room("ctx-test")
            assert result is not None
