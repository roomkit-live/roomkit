"""Tests for SQLiteStore.

The behavioural contract is the one ``InMemoryStore`` defines, so the whole
memory suite runs again here against SQLite — the module-level ``store``
fixture shadows the conftest one for every inherited test class. SQLite-only
behaviour (persistence across connections, FTS5 search) gets its own tests.
"""

from __future__ import annotations

import pytest

from roomkit.models.room import Room
from roomkit.store.sqlite import SQLiteStore
from tests.conftest import make_event
from tests.test_store_memory import (
    TestBindingOperations,
    TestConnectionTenure,
    TestDeleteEvent,
    TestEventOperations,
    TestIdentityOperations,
    TestObservationOperations,
    TestParticipantOperations,
    TestReadTracking,
    TestRoomOperations,
    TestTaskOperations,
)

__all__ = [
    "TestBindingOperations",
    "TestConnectionTenure",
    "TestDeleteEvent",
    "TestEventOperations",
    "TestIdentityOperations",
    "TestObservationOperations",
    "TestParticipantOperations",
    "TestReadTracking",
    "TestRoomOperations",
    "TestTaskOperations",
]


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    await s.close()


class TestPersistence:
    async def test_data_survives_a_close_and_reopen(self, tmp_path) -> None:
        path = tmp_path / "persist.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="premier message"))
        await store.close()

        reopened = SQLiteStore(path)
        try:
            room = await reopened.get_room("r1")
            assert room is not None
            assert room.event_count == 1
            events = await reopened.get_conversation("r1")
            assert [e.content.body for e in events] == ["premier message"]
        finally:
            await reopened.close()

    async def test_close_is_idempotent(self, tmp_path) -> None:
        store = SQLiteStore(tmp_path / "x.db")
        await store.create_room(Room(id="r1"))
        await store.close()
        await store.close()


class TestFullTextSearch:
    async def test_search_finds_by_content_words(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="on a parlé du projet AEC"))
        await store.commit_event(
            "r1", make_event(room_id="r1", body="recette de tarte aux pommes")
        )

        hits = await store.search_events("projet AEC")
        assert [e.content.body for e in hits] == ["on a parlé du projet AEC"]

    async def test_search_scopes_to_a_room(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.create_room(Room(id="r2"))
        await store.commit_event("r1", make_event(room_id="r1", body="meeting notes alpha"))
        await store.commit_event("r2", make_event(room_id="r2", body="meeting notes beta"))

        hits = await store.search_events("meeting", room_id="r2")
        assert [e.room_id for e in hits] == ["r2"]

    async def test_search_survives_hostile_query_syntax(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="hello world"))

        assert await store.search_events('"hello* (world -') != []
        assert await store.search_events("???") == []
        assert await store.search_events("") == []

    async def test_deleted_events_leave_the_index(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="ephemeral thing"))
        assert await store.search_events("ephemeral") != []

        await store.delete_event("r1", event.id)
        assert await store.search_events("ephemeral") == []

    async def test_updated_events_reindex(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="old wording"))
        updated = event.model_copy(deep=True)
        updated.content.body = "new phrasing"
        await store.update_event(updated)

        assert await store.search_events("wording") == []
        assert [e.id for e in await store.search_events("phrasing")] == [event.id]
