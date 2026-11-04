"""Tests for SQLiteStore.

The behavioural contract is the one ``InMemoryStore`` defines, so the whole
memory suite runs again here against SQLite — the module-level ``store``
fixture shadows the conftest one for every inherited test class. SQLite-only
behaviour (persistence across connections, FTS5 search) gets its own tests.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from pathlib import Path

import pytest

from roomkit import AddressLinkedError, RoomKit
from roomkit.core.exceptions import IdempotencyKeyTakenError
from roomkit.core.mixins.verification import HELD_KEY
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, EventStatus, EventType, RoomStatus
from roomkit.models.event import DeleteContent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.store_filter import RoomFilter
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

    async def test_a_v5_file_keeps_its_read_markers_as_event_indexes(self, tmp_path) -> None:
        """v5 kept the read event's id; v6 keeps its index. A marker on an
        event that is gone is dropped: it read as all unread, and still does."""
        path = tmp_path / "v5.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        events = [
            await store.add_event(make_event(room_id="r1", index=i, body=str(i))) for i in range(3)
        ]
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            DROP TABLE read_markers;
            CREATE TABLE read_markers(
                room_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(room_id, channel_id)
            );
            PRAGMA user_version=5;
            """
        )
        conn.executemany(
            "INSERT INTO read_markers(room_id, channel_id, event_id) VALUES(?, ?, ?)",
            [("r1", "sms", events[1].id), ("r1", "email", "gone")],
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            assert await store.list_read_markers("r1") == {"sms": 1}
            assert await store.get_unread_count("r1", "sms") == 1
            assert await store.get_unread_count("r1", "email") == 3
        finally:
            await store.close()
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
        finally:
            conn.close()

    async def test_a_v6_file_stamps_its_events_visibility(self, tmp_path) -> None:
        """v7 stores an event's visibility stamped with its source binding's
        (RFC §7.3). An event stored ``all`` under a narrowed binding takes the
        binding's scope once; an event with a scope of its own, or whose
        source is no longer bound, keeps its own."""
        path = tmp_path / "v6.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.add_binding(
            ChannelBinding(
                room_id="r1", channel_id="sms", channel_type=ChannelType.SMS, visibility="ws"
            )
        )
        bodies = {"under the binding": "all", "its own scope": "ai", "gone": "all"}
        for index, (body, scope) in enumerate(bodies.items()):
            source = "gone" if body == "gone" else "sms"
            await store.add_event(
                make_event(
                    room_id="r1", channel_id=source, body=body, index=index, visibility=scope
                )
            )
        held = {HELD_KEY: {"visibility": "all", "identity_id": "i1", "organization_id": None}}
        await store.add_event(
            make_event(
                room_id="r1",
                channel_id="sms",
                body="held",
                index=3,
                visibility="transport",
                metadata=held,
            )
        )
        await store.close()
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version=6")
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            events = await store.list_events("r1")
            assert {e.content.body: e.visibility for e in events} == {
                "under the binding": "ws",
                "its own scope": "ai",
                "gone": "all",
                "held": "transport",
            }
            held_event = next(e for e in events if e.content.body == "held")
            assert held_event.metadata[HELD_KEY]["visibility"] == "ws"
        finally:
            await store.close()

    async def test_v7_database_scopes_its_keys_to_their_source_channel(self, tmp_path) -> None:
        """A v7 file kept one key per room: the keys it recorded stay found
        under the channel that sent them, and a second channel may reuse one."""
        path = tmp_path / "v7.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.add_event(
            make_event(room_id="r1", channel_id="sms", body="first", index=0, idempotency_key="k")
        )
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            DROP INDEX idx_events_idempotency_channel;
            CREATE UNIQUE INDEX idx_events_idempotency
                ON events(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
            DROP TABLE idempotency;
            CREATE TABLE idempotency(
                room_id TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY(room_id, key)
            );
            INSERT INTO idempotency(room_id, key) VALUES('r1', 'k');
            PRAGMA user_version=7;
            """
        )
        conn.close()

        store = SQLiteStore(path)
        try:
            assert await store.check_idempotency("r1", "k", channel_id="sms")
            assert not await store.check_idempotency("r1", "k", channel_id="email")
            await store.add_event(
                make_event(
                    room_id="r1", channel_id="email", body="second", index=1, idempotency_key="k"
                )
            )
            found = await store.get_event_by_idempotency_key("r1", "k", channel_id="email")
            assert found is not None and found.content.body == "second"
        finally:
            await store.close()

    async def test_v8_database_records_when_its_paused_rooms_paused(self, tmp_path) -> None:
        """A room a v8 file stored paused takes its last modification as the
        instant it paused (RFC §5.1), which its close timer counts from."""
        path = tmp_path / "v8.db"
        store = SQLiteStore(path)
        paused = await store.create_room(Room(id="r1", status=RoomStatus.PAUSED))
        await store.create_room(Room(id="r2"))
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            UPDATE rooms SET data = json_remove(data, '$.paused_at');
            PRAGMA user_version=8;
            """
        )
        conn.close()

        store = SQLiteStore(path)
        try:
            upgraded = await store.get_room("r1")
            active = await store.get_room("r2")
            assert upgraded is not None and active is not None
            assert upgraded.paused_at == paused.updated_at
            assert active.paused_at is None
        finally:
            await store.close()

    async def test_a_v9_file_indexes_its_keys_by_room(self, tmp_path) -> None:
        """Schema 10: a proactive key is looked up within its room, whatever
        channel committed it (RFC §22.1), through an index by room and key."""
        path = tmp_path / "v9.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            DROP INDEX idx_idempotency_room_key;
            DROP INDEX idx_events_room_idempotency;
            PRAGMA user_version=9;
            """
        )
        conn.close()

        store = SQLiteStore(path)
        assert await store.get_room("r1") is not None
        await store.close()

        conn = sqlite3.connect(path)
        try:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            assert {"idx_idempotency_room_key", "idx_events_room_idempotency"} <= names
            assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
        finally:
            conn.close()

    async def test_a_v10_file_indexes_its_participants_by_identity(self, tmp_path) -> None:
        """Schema 11: a sender's participant lookup names it by its identity
        through an index (RFC §10.4 step 1), on an upgraded file too."""
        path = tmp_path / "v10.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript("DROP INDEX idx_participants_identity; PRAGMA user_version=10;")
        conn.close()

        store = SQLiteStore(path)
        assert await store.get_room("r1") is not None
        await store.close()

        conn = sqlite3.connect(path)
        try:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            assert "idx_participants_identity" in names
        finally:
            conn.close()

    async def test_a_v0107_file_no_longer_finds_its_soft_deleted_messages(self, tmp_path) -> None:
        """v0.107.0 (schema 6) indexed a soft-deleted message's text again
        when its delete marked it: the upgrade drops it from the index (RFC
        §10.3), and the record keeps its text."""
        path = tmp_path / "v6.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        secret = await store.commit_event("r1", make_event(room_id="r1", body="zebra secret"))
        await store.commit_event("r1", make_event(room_id="r1", body="zebra fresh"))
        await store.update_event(secret.model_copy(update={"metadata": {"deleted": True}}))
        await store.close()
        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT INTO events_fts(text, event_id, room_id) VALUES(?, ?, ?)",
            ("zebra secret", secret.id, "r1"),
        )
        conn.execute("PRAGMA user_version=6")
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        try:
            hits = await store.search_events("zebra")
            kept = await store.get_event(secret.id)
        finally:
            await store.close()
        assert [e.content.body for e in hits] == ["zebra fresh"]
        assert kept is not None and kept.content == secret.content

    async def test_a_v11_file_moves_its_author_registers_to_its_private_state(
        self, tmp_path
    ) -> None:
        """RMK-631: a room's register of authors leaves its metadata, which
        clients may read, for the room's private state; a malformed one is
        dropped, as a reader reads it absent."""
        path = tmp_path / "v11.db"
        register = {"version": 3, "names": {"a" * 16: "b" * 16 + ":1"}}
        store = SQLiteStore(path)
        await store.create_room(
            Room(id="r1", metadata={"topic": "x", "author_register": register})
        )
        await store.create_room(Room(id="r2", metadata={"author_register": "junk"}))
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript("DROP TABLE private_state; PRAGMA user_version=11;")
        conn.close()

        store = SQLiteStore(path)
        try:
            moved, kept = (
                await store.get_private_state("r1", "author_register"),
                await store.get_room("r1"),
            )
            junk, emptied = (
                await store.get_private_state("r2", "author_register"),
                await store.get_room("r2"),
            )
        finally:
            await store.close()
        assert moved == register
        assert kept is not None and kept.metadata == {"topic": "x"}
        assert junk is None
        assert emptied is not None and emptied.metadata == {}

    async def test_a_v12_file_ties_its_search_rows_to_their_events(self, tmp_path) -> None:
        """RMK-1031: a v12 file's search rows were written by `event_id`, their
        rowids their own. The migration gives each its event row's rowid, so
        an edit replaces the old text; a row naming no event is dropped."""
        path = tmp_path / "v12.db"
        store = SQLiteStore(path)
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="old wording"))
        await store.close()
        conn = sqlite3.connect(path)
        conn.executescript(
            f"""DELETE FROM events_fts;
            INSERT INTO events_fts(rowid, text, event_id, room_id)
                VALUES (900, 'old wording', '{event.id}', 'r1'), (901, 'orphan', 'gone', 'r1');
            PRAGMA user_version=12;"""
        )
        conn.close()

        store = SQLiteStore(path)
        try:
            found = [e.id for e in await store.search_events("wording")]
            updated = event.model_copy(deep=True)
            updated.content.body = "new phrasing"
            await store.update_event(updated)
            stale = await store.search_events("wording")
            fresh = [e.id for e in await store.search_events("phrasing")]
        finally:
            await store.close()
        conn = sqlite3.connect(path)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            rows = conn.execute("SELECT event_id FROM events_fts").fetchall()
        finally:
            conn.close()
        assert (found, stale, fresh, version, rows) == (
            [event.id],
            [],
            [event.id],
            _SCHEMA_VERSION,
            [(event.id,)],
        )

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
            assert sum(isinstance(result, IdempotencyKeyTakenError) for result in results) == 1
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

    async def test_search_does_not_find_the_rows_the_room_refused(
        self, store: SQLiteStore
    ) -> None:
        """RFC §14.1 reaches the index too: a body a hook refused is stored
        BLOCKED, out of the timeline by default, and not findable by search."""
        await store.create_room(Room(id="r1"))
        await store.commit_event("r1", make_event(room_id="r1", body="the secret plan"))
        await store.commit_event(
            "r1",
            make_event(
                room_id="r1", body="the secret leak", status=EventStatus.BLOCKED, blocked_by="dlp"
            ),
        )

        hits = await store.search_events("secret")
        assert [e.content.body for e in hits] == ["the secret plan"]

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

    async def test_a_soft_deleted_event_leaves_the_index(self, store: SQLiteStore) -> None:
        """A deleted message's text is withheld (RFC §10.3): the search
        finds it no more, while the stored record keeps it."""
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="ephemeral thing"))
        delete = make_event(room_id="r1", status=EventStatus.DELIVERED).model_copy(
            update={"type": EventType.DELETE, "content": DeleteContent(target_event_id=event.id)}
        )
        deleted = event.model_copy(update={"metadata": {"deleted": True}})

        await store.commit_event("r1", delete, target=deleted)

        assert await store.search_events("ephemeral") == []
        stored = await store.get_event(event.id)
        assert stored is not None and stored.content == event.content

    async def test_updated_events_reindex(self, store: SQLiteStore) -> None:
        await store.create_room(Room(id="r1"))
        event = await store.commit_event("r1", make_event(room_id="r1", body="old wording"))
        updated = event.model_copy(deep=True)
        updated.content.body = "new phrasing"
        await store.update_event(updated)

        assert await store.search_events("wording") == []
        assert [e.id for e in await store.search_events("phrasing")] == [event.id]


class TestSearchIndexPlans:
    async def test_the_search_rows_are_written_by_rowid(
        self, store: SQLiteStore, tmp_path
    ) -> None:
        """RMK-1031: every commit rewrites its event's search row, and the
        event's deletion and its room's remove them. Reached by `event_id` or
        `room_id`, which FTS5 does not index, each write scanned the whole
        index, a cost that grew with every event the file holds (12 ms a
        commit at 60k rows). They go by the event row's rowid."""
        await store.create_room(Room(id="r1"))
        statements: list[str] = []
        await store._run(lambda: store._db().set_trace_callback(statements.append))

        event = await store.commit_event("r1", make_event(room_id="r1", body="old wording"))
        updated = event.model_copy(deep=True)
        updated.content.body = "new phrasing"
        await store.update_event(updated)
        gone = await store.commit_event("r1", make_event(room_id="r1", body="ephemeral thing"))
        await store.delete_event("r1", gone.id)
        await store.delete_room("r1")

        await store._run(lambda: store._db().set_trace_callback(None))
        conn = sqlite3.connect(tmp_path / "test.db")
        try:
            plans = [
                row[3]
                for statement in statements
                # FTS5's own statements are traced as comments.
                if "events_fts" in statement and not statement.startswith("--")
                for row in conn.execute(f"EXPLAIN QUERY PLAN {statement}")
            ]
        finally:
            conn.close()
        # A full scan of the virtual table reads `INDEX n:` with no
        # constraint; a rowid lookup reads `INDEX n:=`.
        scans = [
            plan
            for plan in plans
            if re.fullmatch(r"SCAN events_fts VIRTUAL TABLE INDEX \d+:", plan)
        ]
        assert plans and scans == [], plans


class TestRoutingPlans:
    @pytest.mark.parametrize("identity_id", [None, "idn-1"], ids=["address", "identity"])
    async def test_the_participant_lookup_reads_through_indexes(
        self, store: SQLiteStore, tmp_path, identity_id: str | None
    ) -> None:
        """A sender's participant lookup (RFC §10.4 step 1) reads the rows
        naming it, by id or by identity, through indexes: the default router
        asks it of every new sender, and a scan of the table grows with every
        participant the file holds."""
        await store.create_room(Room(id="r1"))
        await store.add_participant(
            Participant(id="+1555", room_id="r1", channel_id="sms", identity_id="idn-1")
        )
        statements: list[str] = []
        await store._run(lambda: store._db().set_trace_callback(statements.append))

        found = await store.find_room_id_by_participant(
            "sms", "+1555", identity_id, RoomFilter(statuses=[RoomStatus.ACTIVE])
        )

        await store._run(lambda: store._db().set_trace_callback(None))
        assert found == "r1"
        conn = sqlite3.connect(tmp_path / "test.db")
        try:
            plans = [
                row[3]
                for statement in statements
                for row in conn.execute(f"EXPLAIN QUERY PLAN {statement}")
            ]
        finally:
            conn.close()
        # Each table is searched through an index: the json_each over a row's
        # channels and the union's own rows are the only scans.
        scans = [
            plan for plan in plans if re.match(r"SCAN (TABLE )?(p|r|participants|rooms)\b", plan)
        ]
        assert plans and scans == [], plans


def _insert_address(path: Path, *rows: tuple[str, str, str]) -> None:
    """Writes address rows directly, as an older file holds them."""
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO identity_addresses (channel_type, address, identity_id, organization_id)"
        " VALUES ('sms', ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _address_rows(path: Path) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT address, identity_id, organization_id FROM identity_addresses ORDER BY 1, 3"
    ).fetchall()
    conn.close()
    return rows


async def test_a_replacement_removes_a_row_of_another_scope_for_a_kept_address(
    tmp_path: Path,
) -> None:
    """An address it keeps, under its own organization: a row of another scope
    for the same address goes, not only stops resolving."""
    path = tmp_path / "scoped.db"
    store = SQLiteStore(path)
    identity = Identity(id="i1", organization_id="acme", channel_addresses={"sms": ["+1555"]})
    await store.create_identity(identity)
    _insert_address(path, ("+1555", "i1", ""))
    await store.create_identity(identity)
    assert _address_rows(path) == [("+1555", "i1", "acme")]
    assert await store.resolve_identity("sms", "+1555", organization_id="acme") == identity
    await store.close()


@pytest.mark.parametrize(("held_in", "stored"), [("", "acme"), ("globex", "acme"), ("acme", None)])
async def test_a_row_of_another_scope_than_its_identity_resolves_for_nobody(
    tmp_path: Path, held_in: str, stored: str | None
) -> None:
    """RFC §17.2: an identity is read in its own organization only, whatever a
    row a file's migration left or a direct write says."""
    path = tmp_path / "scoped.db"
    store = SQLiteStore(path)
    await store.create_identity(Identity(id="i1", organization_id=stored))
    _insert_address(path, ("+1555", "i1", held_in))
    for scope in (None, "acme", "globex"):
        assert await store.resolve_identity("sms", "+1555", organization_id=scope) is None
    await store.close()


async def test_opening_moves_an_unscoped_address_to_its_identity_s_organization(
    tmp_path: Path,
) -> None:
    """HITL 0f96a597, RFC §17.2: a row version 3's migration left unscoped moves to
    the organization of its identity, once, if the identity still declares the
    address; one whose organization holds the address already yields to that row. A
    link since revoked is not restored: its row goes, and the address is free (HITL
    6928c418). An identity of no organization keeps its unscoped row, and a row of
    another organization is left alone, which no read serves."""
    path = tmp_path / "older.db"
    store = SQLiteStore(path)
    bob = Identity(id="bob", organization_id="acme", channel_addresses={"sms": ["+2"]})
    carol = Identity(id="carol", channel_addresses={"sms": ["+3"]})
    for identity in (Identity(id="alice", organization_id="acme"), bob, carol):
        await store.create_identity(identity)
    await store.close()
    _insert_address(
        path,
        ("+1", "alice", ""),
        ("+2", "alice", ""),
        ("+4", "alice", "globex"),
        ("+5", "alice", ""),
    )
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE identities SET data = json_set(data, '$.channel_addresses',"
        ' json(\'{"sms": ["+1", "+2"]}\')) WHERE id = \'alice\''
    )
    conn.execute("PRAGMA user_version=13")
    conn.commit()
    conn.close()

    reopened = SQLiteStore(path)
    moved = await reopened.resolve_identity("sms", "+1", organization_id="acme")
    assert moved is not None and moved.id == "alice"
    assert await reopened.resolve_identity("sms", "+1") is None
    for scope in (None, "acme"):
        assert await reopened.resolve_identity("sms", "+5", organization_id=scope) is None
    assert _address_rows(path) == [
        ("+1", "alice", "acme"),
        ("+2", "bob", "acme"),
        ("+3", "carol", ""),
        ("+4", "alice", "globex"),
    ]
    dana = Identity(id="dana", channel_addresses={"sms": ["+5"]})
    await reopened.create_identity(dana)
    assert await reopened.resolve_identity("sms", "+5") == dana
    await reopened.close()


async def test_the_rewrite_takes_only_a_list_for_a_declaration(tmp_path: Path) -> None:
    """A declaration no API writes (a text, a map) declares nothing: the row goes,
    as on Postgres, and the file still opens."""
    path = tmp_path / "older.db"
    store = SQLiteStore(path)
    await store.create_identity(Identity(id="alice", organization_id="acme"))
    await store.close()
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO identity_addresses (channel_type, address, identity_id, organization_id)"
        " VALUES (?, ?, 'alice', '')",
        [("sms", "a@b.c"), ("mms", "+2")],
    )
    # Text SQLite cannot read as JSON: json_each on it would raise, and stop the file
    # from opening.
    conn.execute(
        "UPDATE identities SET data = json_set(data, '$.channel_addresses',"
        ' json(\'{"sms": "a@b.c", "mms": {"k": "+2"}}\')) WHERE id = \'alice\''
    )
    conn.execute("PRAGMA user_version=13")
    conn.commit()
    conn.close()
    reopened = SQLiteStore(path)
    assert await reopened.resolve_identity("sms", "a@b.c", organization_id="acme") is None
    await reopened.close()
    assert _address_rows(path) == []


@pytest.mark.parametrize(("held_in", "stored"), [("", "acme"), ("globex", "acme"), ("acme", None)])
@pytest.mark.parametrize("taking", ["create_identity", "link_address"])
async def test_a_row_no_read_serves_leaves_its_address_free(
    tmp_path: Path, held_in: str, stored: str | None, taking: str
) -> None:
    """RFC §5.6: an address is refused for what another identity of the organization
    holds. A row an older release left, whose identity is of another organization,
    resolves for nobody: the organization's own identity takes the address."""
    path = tmp_path / "older.db"
    store = SQLiteStore(path)
    await store.create_identity(Identity(id="i1", organization_id=stored))
    _insert_address(path, ("+1555", "i1", held_in))
    scope = held_in or None
    if taking == "create_identity":
        await store.create_identity(
            Identity(id="dana", organization_id=scope, channel_addresses={"sms": ["+1555"]})
        )
    else:
        await store.create_identity(Identity(id="dana", organization_id=scope))
        await store.link_address("dana", "sms", "+1555", organization_id=scope)
    resolved = await store.resolve_identity("sms", "+1555", organization_id=scope)
    await store.close()
    assert resolved is not None and resolved.id == "dana"
    assert _address_rows(path) == [("+1555", "dana", held_in)]


@pytest.mark.parametrize(
    "record",
    [
        '{"email": "a@globex.example"}',
        '{"id": "i1", "channel_addresses": "x"}',
        '["a@globex.example"]',
    ],
)
async def test_a_holder_the_model_cannot_read_keeps_its_address(
    tmp_path: Path, record: str
) -> None:
    """RFC §5.6: the refusal stands, and quotes no part of the holder's record, which
    may be another organization's: a record with no id, of the wrong type, or no object."""
    path = tmp_path / "older.db"
    store = SQLiteStore(path)
    await store.create_identity(Identity(id="i1", organization_id="globex"))
    conn = sqlite3.connect(path)
    conn.execute("UPDATE identities SET data = ? WHERE id = 'i1'", (record,))
    conn.commit()
    conn.close()
    _insert_address(path, ("+1555", "i1", "acme"))
    with pytest.raises(AddressLinkedError) as refused:
        await store.create_identity(
            Identity(id="dana", organization_id="acme", channel_addresses={"sms": ["+1555"]})
        )
    await store.close()
    assert "globex" not in str(refused.value)
    assert refused.value.__suppress_context__
    assert _address_rows(path) == [("+1555", "i1", "acme")]


async def test_an_unlink_removes_the_identity_s_rows_in_every_scope(tmp_path: Path) -> None:
    """RFC §5.6: a link revoked is revoked, a row a file's migration left in another
    scope with it, so that no later rewrite can restore it."""
    path = tmp_path / "unlink.db"
    store = SQLiteStore(path)
    identity = Identity(id="i1", organization_id="acme", channel_addresses={"sms": ["+1"]})
    await store.create_identity(identity)
    _insert_address(path, ("+1", "i1", ""))
    await store.unlink_address("i1", "sms", "+1", organization_id="acme")
    assert _address_rows(path) == []
    await store.close()
