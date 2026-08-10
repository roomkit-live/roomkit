"""Tests for SQLiteStore.

The behavioural contract is the one ``InMemoryStore`` defines, so the whole
memory suite runs again here against SQLite — the module-level ``store``
fixture shadows the conftest one for every inherited test class. SQLite-only
behaviour (persistence across connections, FTS5 search) gets its own tests.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

import pytest

from roomkit import RoomKit
from roomkit.models.room import Room
from roomkit.store.sqlite import _SCHEMA_VERSION, SQLiteSchemaError, SQLiteStore
from tests.conftest import make_event
from tests.test_store_memory import (
    TestBindingOperations,
    TestConnectionTenure,
    TestCursorPagination,
    TestDeleteEvent,
    TestDeleteRoomCleanup,
    TestEventOperations,
    TestEventOwnership,
    TestFindLatestRoomIndex,
    TestFindRooms,
    TestIdentityOperations,
    TestListEventsVisibilityFilter,
    TestObservationOperations,
    TestParticipantOperations,
    TestReadTracking,
    TestRoomOperations,
    TestTaskOperations,
)

__all__ = [
    "TestBindingOperations",
    "TestConnectionTenure",
    "TestCursorPagination",
    "TestDeleteEvent",
    "TestDeleteRoomCleanup",
    "TestEventOperations",
    "TestEventOwnership",
    "TestFindLatestRoomIndex",
    "TestFindRooms",
    "TestIdentityOperations",
    "TestListEventsVisibilityFilter",
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

    async def test_v1_database_migrates_sequence_and_unique_index(self, tmp_path) -> None:
        """A development-era v1 file upgrades without reusing deleted history."""
        path = tmp_path / "v1.db"
        room = Room(id="r1", event_count=2, latest_index=1)
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE rooms(
                id TEXT PRIMARY KEY,
                created_ts REAL NOT NULL,
                organization_id TEXT,
                status TEXT NOT NULL,
                delivered_index INTEGER NOT NULL DEFAULT -1,
                data TEXT NOT NULL
            );
            CREATE TABLE events(
                id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                type TEXT NOT NULL,
                visibility TEXT NOT NULL,
                source_channel_id TEXT,
                source_channel_type TEXT,
                participant_id TEXT,
                correlation_id TEXT,
                parent_event_id TEXT,
                idempotency_key TEXT,
                created_ts REAL NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX idx_events_room_idx ON events(room_id, idx);
            PRAGMA user_version=1;
            """
        )
        conn.execute(
            "INSERT INTO rooms(id, created_ts, status, data) VALUES(?, ?, ?, ?)",
            (room.id, room.created_at.timestamp(), str(room.status), room.model_dump_json()),
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            committed = await store.commit_event(
                "r1", make_event(room_id="r1", body="after deleted history")
            )
            assert committed.index == 2
        finally:
            await store.close()

        conn = sqlite3.connect(path)
        try:
            # A v1 file is brought all the way to the current schema, not
            # to the version that happened to follow it.
            assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
            indexes = conn.execute("PRAGMA index_list(events)").fetchall()
            room_index = next(row for row in indexes if row[1] == "idx_events_room_idx")
            assert room_index[2] == 1  # unique
        finally:
            conn.close()

    async def test_newer_schema_is_rejected_without_relabelling(self, tmp_path) -> None:
        path = tmp_path / "future.db"
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version=999")
        conn.close()

        store = SQLiteStore(path)
        with pytest.raises(SQLiteSchemaError, match="newer"):
            await store.get_room("r1")
        await store.close()

        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 999
        finally:
            conn.close()

    async def test_two_connections_allocate_one_monotonic_sequence(self, tmp_path) -> None:
        path = tmp_path / "shared.db"
        first = SQLiteStore(path)
        second = SQLiteStore(path)
        await first.create_room(Room(id="r1"))
        try:
            committed = await asyncio.gather(
                *(
                    (first if i % 2 == 0 else second).commit_event(
                        "r1", make_event(room_id="r1", body=f"message {i}")
                    )
                    for i in range(20)
                )
            )
            assert sorted(event.index for event in committed) == list(range(20))
        finally:
            await first.close()
            await second.close()

    async def test_duplicate_index_is_rejected_by_database(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        await store.add_event(make_event(room_id="r1", body="first", index=0))

        with pytest.raises(sqlite3.IntegrityError):
            await store.add_event(make_event(room_id="r1", body="duplicate", index=0))

    async def test_duplicate_idempotency_key_is_rejected_across_connections(
        self, tmp_path
    ) -> None:
        path = tmp_path / "idempotency.db"
        first = SQLiteStore(path)
        second = SQLiteStore(path)
        await first.create_room(Room(id="r1"))
        one = make_event(room_id="r1", body="one", idempotency_key="same")
        two = make_event(room_id="r1", body="two", idempotency_key="same")
        try:
            results = await asyncio.gather(
                first.commit_event("r1", one),
                second.commit_event("r1", two),
                return_exceptions=True,
            )
            assert sum(isinstance(result, sqlite3.IntegrityError) for result in results) == 1
            assert len(await first.list_events("r1")) == 1
        finally:
            await first.close()
            await second.close()

    async def test_framework_warns_that_shared_sqlite_needs_distributed_lock(
        self, tmp_path, caplog
    ) -> None:
        store = SQLiteStore(tmp_path / "warning.db")
        with caplog.at_level(logging.WARNING, logger="roomkit"):
            RoomKit(store=store)
        assert "safe only in a single process" in caplog.text
        await store.close()

    def test_sqlite_store_is_exported_from_package_root(self) -> None:
        from roomkit import SQLiteSchemaError as RootSQLiteSchemaError
        from roomkit import SQLiteStore as RootSQLiteStore

        assert RootSQLiteStore is SQLiteStore
        assert RootSQLiteSchemaError is SQLiteSchemaError


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
