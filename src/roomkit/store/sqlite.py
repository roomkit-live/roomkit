"""SQLite implementation of ConversationStore.

The embedded persistent backend: one ``.db`` file, no server, stdlib
``sqlite3`` only. Built for single-process deployments — desktop apps,
edge boxes, small bots — where ``PostgresStore`` is a burden and
``InMemoryStore`` forgets everything on exit.

Models are stored as their pydantic JSON with the queryable fields
extracted into indexed columns; reads round-trip through
``model_validate_json`` so stored objects always match what the models
serialise. Message text is additionally mirrored into an FTS5 table, so
conversation history is full-text searchable (:meth:`SQLiteStore.search_events`)
without any extra infrastructure.

Every database call runs on a single dedicated worker thread: SQLite
objects stay on the thread that created them, writes are naturally
serialised, and the event loop never blocks on disk I/O. Multi-statement
operations (``commit_event``, ``delete_room``…) run as one ``BEGIN
IMMEDIATE`` transaction, which also guards against another *process*
sharing the file.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.event import RoomEvent, ThreadSummary
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Observation, Task
from roomkit.store.base import ConversationStore

_SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms(
    id TEXT PRIMARY KEY,
    created_ts REAL NOT NULL,
    organization_id TEXT,
    status TEXT NOT NULL,
    delivered_index INTEGER NOT NULL DEFAULT -1,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
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
CREATE INDEX IF NOT EXISTS idx_events_room ON events(room_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_room_idx ON events(room_id, idx);
CREATE INDEX IF NOT EXISTS idx_events_room_parent ON events(room_id, parent_event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency
    ON events(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE TABLE IF NOT EXISTS event_sequences(
    room_id TEXT PRIMARY KEY,
    next_index INTEGER NOT NULL DEFAULT 0 CHECK(next_index >= 0)
);
CREATE TABLE IF NOT EXISTS bindings(
    room_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    participant_id TEXT,
    channel_type TEXT,
    data TEXT NOT NULL,
    PRIMARY KEY(room_id, channel_id)
);
CREATE INDEX IF NOT EXISTS idx_bindings_channel ON bindings(channel_id);
CREATE TABLE IF NOT EXISTS participants(
    room_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(room_id, id)
);
CREATE TABLE IF NOT EXISTS identities(
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS identity_addresses(
    channel_type TEXT NOT NULL,
    address TEXT NOT NULL,
    identity_id TEXT NOT NULL,
    organization_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(channel_type, address, organization_id)
);
CREATE TABLE IF NOT EXISTS tasks(
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_room ON tasks(room_id);
CREATE TABLE IF NOT EXISTS observations(
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_room ON observations(room_id);
CREATE TABLE IF NOT EXISTS read_markers(
    room_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    PRIMARY KEY(room_id, channel_id)
);
CREATE TABLE IF NOT EXISTS idempotency(
    room_id TEXT NOT NULL,
    key TEXT NOT NULL,
    PRIMARY KEY(room_id, key)
);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
    USING fts5(text, event_id UNINDEXED, room_id UNINDEXED);
"""


def _ts(value: datetime) -> float:
    return value.timestamp()


def _fts_text(event: RoomEvent) -> str:
    """Extract the searchable text of an event's content, if any."""
    parts: list[str] = []
    contents = [event.content]
    contents.extend(getattr(event.content, "parts", None) or [])
    for content in contents:
        for attr in ("body", "text", "caption"):
            value = getattr(content, attr, None)
            if isinstance(value, str) and value.strip():
                parts.append(value)
                break
    return "\n".join(parts)


def _load_event(data: str) -> RoomEvent:
    return RoomEvent.model_validate_json(data)


@contextmanager
def _write_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run one immediate SQLite transaction with reliable rollback."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


class SQLiteSchemaError(RuntimeError):
    """Raised when a SQLite file cannot be migrated or is from a newer schema."""


def _populate_event_sequences(conn: sqlite3.Connection) -> None:
    """Advance sequence rows from both live events and historical room tallies."""
    conn.execute(
        """INSERT INTO event_sequences(room_id, next_index)
           SELECT room_id, MAX(idx) + 1 FROM events GROUP BY room_id
           ON CONFLICT(room_id) DO UPDATE SET
               next_index = MAX(event_sequences.next_index, excluded.next_index)"""
    )
    for room_id, data in conn.execute("SELECT id, data FROM rooms"):
        room = Room.model_validate_json(data)
        # event_count is a never-decremented tally, so it distinguishes an
        # untouched room (latest_index's model default is 0) from one whose
        # highest event was deleted.
        historical_next = room.latest_index + 1 if room.event_count > 0 else 0
        conn.execute(
            """INSERT INTO event_sequences(room_id, next_index) VALUES(?, ?)
               ON CONFLICT(room_id) DO UPDATE SET
                   next_index = MAX(event_sequences.next_index, excluded.next_index)""",
            (room_id, historical_next),
        )


def _create_schema(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{_SCHEMA}")
        _populate_event_sequences(conn)
        conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    duplicate_index = conn.execute(
        """SELECT room_id, idx FROM events GROUP BY room_id, idx
           HAVING COUNT(*) > 1 LIMIT 1"""
    ).fetchone()
    if duplicate_index is not None:
        raise SQLiteSchemaError(
            "SQLite schema v1 contains duplicate (room_id, index) rows; "
            "deduplicate them before upgrading to v2"
        )
    duplicate_key = conn.execute(
        """SELECT room_id, idempotency_key FROM events
           WHERE idempotency_key IS NOT NULL
           GROUP BY room_id, idempotency_key HAVING COUNT(*) > 1 LIMIT 1"""
    ).fetchone()
    if duplicate_key is not None:
        raise SQLiteSchemaError(
            "SQLite schema v1 contains duplicate idempotency keys; "
            "deduplicate them before upgrading to v2"
        )
    try:
        # v1 used this same name for a non-unique index. It must be removed so
        # the v2 CREATE UNIQUE INDEX is not skipped by IF NOT EXISTS.
        migration = "BEGIN IMMEDIATE;\nDROP INDEX IF EXISTS idx_events_room_idx;\n" + _SCHEMA
        conn.executescript(migration)
        _populate_event_sequences(conn)
        conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Scope identity addresses by organization (RFC §17.2).

    v2 keyed ``identity_addresses`` on ``(channel_type, address)``, so an
    address belonged to one identity across every tenant. The key gains
    ``organization_id``; SQLite cannot alter a primary key in place, so the
    table is rebuilt. Existing rows carry the empty string — the unscoped
    tenant — which preserves exactly what they resolved to before.
    """
    # One script: ``executescript`` commits any open transaction before it
    # runs, so a second call would close the one opened here and leave the
    # COMMIT below with nothing to commit.
    rebuild = """BEGIN IMMEDIATE;
CREATE TABLE identity_addresses_v3(
    channel_type TEXT NOT NULL,
    address TEXT NOT NULL,
    identity_id TEXT NOT NULL,
    organization_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(channel_type, address, organization_id)
);
INSERT INTO identity_addresses_v3(channel_type, address, identity_id, organization_id)
    SELECT channel_type, address, identity_id, '' FROM identity_addresses;
DROP TABLE identity_addresses;
ALTER TABLE identity_addresses_v3 RENAME TO identity_addresses;
"""
    try:
        conn.executescript(rebuild + _SCHEMA)
        conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _ensure_schema(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > _SCHEMA_VERSION:
        raise SQLiteSchemaError(
            f"SQLite schema version {version} is newer than supported version "
            f"{_SCHEMA_VERSION}; upgrade RoomKit before opening this file"
        )
    if version == 0:
        _create_schema(conn)
    elif version == 1:
        _migrate_v1_to_v2(conn)
        _migrate_v2_to_v3(conn)
    elif version == 2:
        _migrate_v2_to_v3(conn)
    elif version == _SCHEMA_VERSION:
        # Additive repair for a partially initialized v2 file. The version is
        # never rewritten, and all statements are idempotent.
        conn.executescript(_SCHEMA)
        _populate_event_sequences(conn)
    else:
        raise SQLiteSchemaError(f"Unsupported SQLite schema version {version}")


class SQLiteStore(ConversationStore):
    """Single-file persistent store backed by stdlib ``sqlite3`` + FTS5."""

    def __init__(self, path: str | Path = "roomkit.db") -> None:
        self._path = str(path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="roomkit-sqlite")
        self._conn: sqlite3.Connection | None = None
        self._closed = False

    # -- Plumbing ----------------------------------------------------------

    async def _run(self, fn: Any, *args: Any) -> Any:
        if self._closed:
            raise RuntimeError("SQLiteStore is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args))

    def _db(self) -> sqlite3.Connection:
        """Open (once) and return the connection. Worker thread only."""
        if self._conn is None:
            Path(self._path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(Path(self._path).expanduser()), isolation_level=None)
            try:
                conn.row_factory = sqlite3.Row
                _ensure_schema(conn)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            except BaseException:
                conn.close()
                raise
            self._conn = conn
        return self._conn

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        def _shut() -> None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _shut)
        self._executor.shutdown(wait=False)

    def _row_to_room(self, row: sqlite3.Row) -> Room:
        room = Room.model_validate_json(row["data"])
        # The column is authoritative for the store-managed delivery cursor
        # (advance_delivered_index writes it without rewriting the JSON).
        if room.delivered_index != row["delivered_index"]:
            room = room.model_copy(update={"delivered_index": row["delivered_index"]})
        return room

    _EVENT_INSERT = """
        INSERT INTO events(id, room_id, idx, type, visibility, source_channel_id,
                           source_channel_type, participant_id, correlation_id,
                           parent_event_id, idempotency_key, created_ts, data)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    _EVENT_UPSERT = """
        INSERT INTO events(id, room_id, idx, type, visibility, source_channel_id,
                           source_channel_type, participant_id, correlation_id,
                           parent_event_id, idempotency_key, created_ts, data)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            room_id=excluded.room_id, idx=excluded.idx, type=excluded.type,
            visibility=excluded.visibility,
            source_channel_id=excluded.source_channel_id,
            source_channel_type=excluded.source_channel_type,
            participant_id=excluded.participant_id,
            correlation_id=excluded.correlation_id,
            parent_event_id=excluded.parent_event_id,
            idempotency_key=excluded.idempotency_key,
            created_ts=excluded.created_ts, data=excluded.data
    """

    def _write_event_row(
        self, conn: sqlite3.Connection, event: RoomEvent, *, replace: bool = False
    ) -> None:
        source = event.source
        conn.execute(
            self._EVENT_UPSERT if replace else self._EVENT_INSERT,
            (
                event.id,
                event.room_id,
                event.index,
                str(event.type),
                event.visibility,
                source.channel_id if source else None,
                str(source.channel_type) if source and source.channel_type else None,
                source.participant_id if source else None,
                event.correlation_id,
                event.parent_event_id,
                event.idempotency_key,
                _ts(event.created_at),
                event.model_dump_json(),
            ),
        )
        if event.idempotency_key:
            conn.execute(
                "INSERT OR IGNORE INTO idempotency(room_id, key) VALUES(?, ?)",
                (event.room_id, event.idempotency_key),
            )
        conn.execute("DELETE FROM events_fts WHERE event_id = ?", (event.id,))
        text = _fts_text(event)
        if text:
            conn.execute(
                "INSERT INTO events_fts(text, event_id, room_id) VALUES(?, ?, ?)",
                (text, event.id, event.room_id),
            )
        self._observe_event_index(conn, event.room_id, event.index)

    @staticmethod
    def _reserve_event_index(conn: sqlite3.Connection, room_id: str) -> int:
        """Reserve the next room index inside the caller's write transaction."""
        conn.execute(
            "INSERT OR IGNORE INTO event_sequences(room_id, next_index) VALUES(?, 0)",
            (room_id,),
        )
        row = conn.execute(
            "SELECT next_index FROM event_sequences WHERE room_id = ?", (room_id,)
        ).fetchone()
        if row is None:  # pragma: no cover - guarded by the INSERT above
            raise RuntimeError(f"Could not reserve an event index for room {room_id!r}")
        index = int(row[0])
        conn.execute(
            "UPDATE event_sequences SET next_index = ? WHERE room_id = ?",
            (index + 1, room_id),
        )
        return index

    @staticmethod
    def _observe_event_index(conn: sqlite3.Connection, room_id: str, event_index: int) -> None:
        """Keep auto-indexing ahead of indexes supplied by store callers."""
        conn.execute(
            """INSERT INTO event_sequences(room_id, next_index) VALUES(?, ?)
               ON CONFLICT(room_id) DO UPDATE SET
                   next_index = MAX(event_sequences.next_index, excluded.next_index)""",
            (room_id, event_index + 1),
        )

    # -- Room operations ---------------------------------------------------

    async def create_room(self, room: Room) -> Room:
        return await self._run(self._x_create_room, room)

    def _x_create_room(self, room: Room) -> Room:
        conn = self._db()
        conn.execute(
            "INSERT OR REPLACE INTO rooms(id, created_ts, organization_id, status,"
            " delivered_index, data) VALUES(?, ?, ?, ?, ?, ?)",
            (
                room.id,
                _ts(room.created_at),
                room.organization_id,
                str(room.status),
                room.delivered_index,
                room.model_dump_json(),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO event_sequences(room_id, next_index) VALUES(?, 0)",
            (room.id,),
        )
        return room

    async def get_room(self, room_id: str) -> Room | None:
        return await self._run(self._x_get_room, room_id)

    def _x_get_room(self, room_id: str) -> Room | None:
        row = self._db().execute("SELECT * FROM rooms WHERE id = ?", (room_id,)).fetchone()
        return self._row_to_room(row) if row is not None else None

    async def update_room(self, room: Room) -> Room:
        return await self._run(self._x_update_room, room)

    def _x_update_room(self, room: Room) -> Room:
        # delivered_index is store-managed (advance_delivered_index): the
        # column is not part of this write, so a stale copy cannot rewind it.
        cur = self._db().execute(
            "UPDATE rooms SET created_ts=?, organization_id=?, status=?, data=? WHERE id=?",
            (
                _ts(room.created_at),
                room.organization_id,
                str(room.status),
                room.model_dump_json(),
                room.id,
            ),
        )
        if cur.rowcount == 0:
            from roomkit.core.framework import RoomNotFoundError

            raise RoomNotFoundError(room.id)
        delivered = (
            self._db()
            .execute("SELECT delivered_index FROM rooms WHERE id=?", (room.id,))
            .fetchone()
        )
        if delivered is not None and room.delivered_index != delivered[0]:
            room = room.model_copy(update={"delivered_index": delivered[0]})
        return room

    async def room_exists(self, room_id: str) -> bool:
        return await self._run(self._x_room_exists, room_id)

    def _x_room_exists(self, room_id: str) -> bool:
        return (
            self._db().execute("SELECT 1 FROM rooms WHERE id = ?", (room_id,)).fetchone()
            is not None
        )

    async def get_delivered_index(self, room_id: str) -> int:
        return await self._run(self._x_get_delivered_index, room_id)

    def _x_get_delivered_index(self, room_id: str) -> int:
        row = (
            self._db()
            .execute("SELECT delivered_index FROM rooms WHERE id = ?", (room_id,))
            .fetchone()
        )
        return -1 if row is None else row[0]

    async def advance_delivered_index(
        self, room_id: str, index: int, *, force: bool = False
    ) -> bool:
        return await self._run(self._x_advance_delivered_index, room_id, index, force)

    def _x_advance_delivered_index(self, room_id: str, index: int, force: bool) -> bool:
        if force:
            cur = self._db().execute(
                "UPDATE rooms SET delivered_index=? WHERE id=? AND delivered_index < ?",
                (index, room_id, index),
            )
        else:
            cur = self._db().execute(
                "UPDATE rooms SET delivered_index=? WHERE id=? AND delivered_index = ?",
                (index, room_id, index - 1),
            )
        return cur.rowcount > 0

    async def patch_room_metadata(
        self,
        room_id: str,
        patch: dict[str, Any],
        *,
        unset: Any = (),
    ) -> Room | None:
        return await self._run(self._x_patch_room_metadata, room_id, patch, tuple(unset))

    def _x_patch_room_metadata(
        self, room_id: str, patch: dict[str, Any], unset: tuple[str, ...]
    ) -> Room | None:
        conn = self._db()
        with _write_transaction(conn):
            row = conn.execute("SELECT * FROM rooms WHERE id = ?", (room_id,)).fetchone()
            if row is None:
                return None
            room = self._row_to_room(row)
            metadata = {k: v for k, v in room.metadata.items() if k not in set(unset)}
            metadata.update(patch)
            room = room.model_copy(update={"metadata": metadata, "updated_at": datetime.now(UTC)})
            conn.execute("UPDATE rooms SET data=? WHERE id=?", (room.model_dump_json(), room_id))
        return room

    async def delete_room(self, room_id: str) -> bool:
        return await self._run(self._x_delete_room, room_id)

    def _x_delete_room(self, room_id: str) -> bool:
        conn = self._db()
        with _write_transaction(conn):
            cur = conn.execute("DELETE FROM rooms WHERE id = ?", (room_id,))
            existed = cur.rowcount > 0
            for table in (
                "events",
                "bindings",
                "participants",
                "tasks",
                "observations",
                "read_markers",
                "idempotency",
                "event_sequences",
                "events_fts",
            ):
                conn.execute(f"DELETE FROM {table} WHERE room_id = ?", (room_id,))  # nosec B608 — fragments are internal, values parameterised
        return existed

    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        return await self._run(self._x_list_rooms, offset, limit)

    def _x_list_rooms(self, offset: int, limit: int) -> list[Room]:
        rows = self._db().execute(
            "SELECT * FROM rooms ORDER BY rowid LIMIT ? OFFSET ?", (limit, offset)
        )
        return [self._row_to_room(r) for r in rows]

    async def find_rooms(
        self,
        organization_id: str | None = None,
        status: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Room]:
        return await self._run(
            self._x_find_rooms, organization_id, status, metadata_filter, limit, offset
        )

    def _x_find_rooms(
        self,
        organization_id: str | None,
        status: str | None,
        metadata_filter: dict[str, Any] | None,
        limit: int,
        offset: int,
    ) -> list[Room]:
        where, params = ["1=1"], []
        if organization_id is not None:
            where.append("organization_id = ?")
            params.append(organization_id)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        rows = self._db().execute(
            f"SELECT * FROM rooms WHERE {' AND '.join(where)} ORDER BY rowid",  # nosec B608 — fragments are internal, values parameterised
            params,
        )
        results = [self._row_to_room(r) for r in rows]
        if metadata_filter:
            # Metadata values are arbitrary JSON — compare in Python rather
            # than fight json_extract over type affinities.
            results = [
                room
                for room in results
                if all(room.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
        return results[offset : offset + limit]

    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        return await self._run(self._x_find_latest_room, participant_id, channel_type, status)

    def _x_find_latest_room(
        self, participant_id: str, channel_type: str | None, status: str | None
    ) -> Room | None:
        # A participant row qualifies the room regardless of channel_type;
        # the channel_type constraint only narrows the binding path — same
        # predicate as InMemoryStore.
        where = [
            """(EXISTS(SELECT 1 FROM participants p WHERE p.room_id = rooms.id AND p.id = ?)
                OR EXISTS(SELECT 1 FROM bindings b WHERE b.room_id = rooms.id
                          AND b.participant_id = ?
                          AND (? IS NULL OR b.channel_type = ?)))"""
        ]
        params: list[Any] = [participant_id, participant_id, channel_type, channel_type]
        if status is not None:
            where.append("status = ?")
            params.append(status)
        row = (
            self._db()
            .execute(
                f"SELECT * FROM rooms WHERE {' AND '.join(where)}"  # nosec B608 — fragments are internal, values parameterised
                " ORDER BY created_ts DESC LIMIT 1",
                params,
            )
            .fetchone()
        )
        return self._row_to_room(row) if row is not None else None

    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        matches = await self.find_room_ids_by_channel(channel_id, status=status, limit=1)
        return matches[0] if matches else None

    async def find_room_ids_by_channel(
        self, channel_id: str, status: str | None = None, limit: int = 2
    ) -> list[str]:
        return await self._run(self._x_find_room_ids_by_channel, channel_id, status, limit)

    def _x_find_room_ids_by_channel(
        self, channel_id: str, status: str | None, limit: int
    ) -> list[str]:
        where = ["b.channel_id = ?"]
        params: list[Any] = [channel_id]
        if status is not None:
            where.append("r.status = ?")
            params.append(status)
        params.append(limit)
        rows = self._db().execute(
            f"""SELECT r.id FROM bindings b JOIN rooms r ON r.id = b.room_id
                WHERE {" AND ".join(where)}
                ORDER BY r.created_ts, r.id LIMIT ?""",  # nosec B608 — fragments are internal, values parameterised
            params,
        )
        return [row[0] for row in rows]

    # -- Event operations --------------------------------------------------

    async def add_event(self, event: RoomEvent) -> RoomEvent:
        return await self._run(self._x_add_event, event)

    def _x_add_event(self, event: RoomEvent) -> RoomEvent:
        conn = self._db()
        with _write_transaction(conn):
            self._write_event_row(conn, event)
        return event

    async def get_event(self, event_id: str) -> RoomEvent | None:
        return await self._run(self._x_get_event, event_id)

    def _x_get_event(self, event_id: str) -> RoomEvent | None:
        row = self._db().execute("SELECT data FROM events WHERE id = ?", (event_id,)).fetchone()
        return _load_event(row[0]) if row is not None else None

    async def update_event(self, event: RoomEvent) -> RoomEvent:
        return await self._run(self._x_update_event, event)

    def _x_update_event(self, event: RoomEvent) -> RoomEvent:
        conn = self._db()
        with _write_transaction(conn):
            self._write_event_row(conn, event, replace=True)
        return event

    async def delete_event(
        self, room_id: str, event_id: str, *, cascade_replies: bool = True
    ) -> list[str]:
        return await self._run(self._x_delete_event, room_id, event_id, cascade_replies)

    def _x_delete_event(self, room_id: str, event_id: str, cascade_replies: bool) -> list[str]:
        conn = self._db()
        with _write_transaction(conn):
            root = conn.execute(
                "SELECT 1 FROM events WHERE id = ? AND room_id = ?", (event_id, room_id)
            ).fetchone()
            if root is None:
                return []
            deleted = [event_id]
            if cascade_replies:
                rows = conn.execute(
                    "SELECT id FROM events WHERE room_id = ? AND parent_event_id = ?"
                    " AND id != ? ORDER BY rowid",
                    (room_id, event_id, event_id),
                )
                deleted.extend(row[0] for row in rows)
            marks = ",".join("?" for _ in deleted)
            keys = conn.execute(
                f"SELECT idempotency_key FROM events WHERE id IN ({marks})"  # nosec B608 — fragments are internal, values parameterised
                " AND idempotency_key IS NOT NULL",
                deleted,
            ).fetchall()
            for (key,) in keys:
                conn.execute(
                    "DELETE FROM idempotency WHERE room_id = ? AND key = ?", (room_id, key)
                )
            conn.execute(f"DELETE FROM events WHERE id IN ({marks})", deleted)  # nosec B608 — fragments are internal, values parameterised
            conn.execute(f"DELETE FROM events_fts WHERE event_id IN ({marks})", deleted)  # nosec B608 — fragments are internal, values parameterised
        return deleted

    async def list_events(
        self,
        room_id: str,
        offset: int = 0,
        limit: int = 50,
        visibility_filter: str | None = None,
        *,
        after_index: int | None = None,
        before_index: int | None = None,
        event_filter: EventFilter | None = None,
        newest_first: bool = False,
    ) -> list[RoomEvent]:
        if after_index is not None and before_index is not None:
            raise ValueError("after_index and before_index are mutually exclusive")
        return await self._run(
            self._x_list_events,
            room_id,
            offset,
            limit,
            visibility_filter,
            after_index,
            before_index,
            event_filter,
            newest_first,
        )

    def _x_list_events(
        self,
        room_id: str,
        offset: int,
        limit: int,
        visibility_filter: str | None,
        after_index: int | None,
        before_index: int | None,
        ef: EventFilter | None,
        newest_first: bool,
    ) -> list[RoomEvent]:
        where = ["room_id = ?"]
        params: list[Any] = [room_id]
        if after_index is not None:
            where.append("idx > ?")
            params.append(after_index)
        elif before_index is not None:
            where.append("idx < ?")
            params.append(before_index)
        visibility = (
            ef.visibility if ef is not None and ef.visibility is not None else visibility_filter
        )
        if visibility is not None:
            where.append("visibility = ?")
            params.append(visibility)
        if ef is not None:
            if ef.event_types is not None:
                marks = ",".join("?" for _ in ef.event_types)
                where.append(f"type IN ({marks})")
                params.extend(str(t) for t in ef.event_types)
            if ef.exclude_types is not None:
                marks = ",".join("?" for _ in ef.exclude_types)
                where.append(f"type NOT IN ({marks})")
                params.extend(str(t) for t in ef.exclude_types)
            if ef.source_channel_id is not None:
                where.append("source_channel_id = ?")
                params.append(ef.source_channel_id)
            if ef.source_channel_type is not None:
                where.append("source_channel_type = ?")
                params.append(str(ef.source_channel_type))
            if ef.correlation_id is not None:
                where.append("correlation_id = ?")
                params.append(ef.correlation_id)
            if ef.participant_id is not None:
                where.append("participant_id = ?")
                params.append(ef.participant_id)
            if ef.parent_event_id is not None:
                where.append("parent_event_id = ?")
                params.append(ef.parent_event_id)
            if ef.top_level_only:
                where.append("parent_event_id IS NULL")
            if ef.after_time is not None:
                where.append("created_ts > ?")
                params.append(_ts(ef.after_time))
            if ef.before_time is not None:
                where.append("created_ts < ?")
                params.append(_ts(ef.before_time))

        base = f"SELECT data FROM events WHERE {' AND '.join(where)}"  # nosec B608 — fragments are internal, values parameterised
        if before_index is not None or (
            before_index is None and after_index is None and newest_first
        ):
            # Tail page: newest ``limit`` rows (offset counted from the newest
            # end), returned in chronological order.
            tail_offset = 0 if before_index is not None else offset
            rows = self._db().execute(
                f"{base} ORDER BY rowid DESC LIMIT ? OFFSET ?", [*params, limit, tail_offset]
            )
            return [_load_event(r[0]) for r in rows][::-1]
        head_offset = 0 if after_index is not None else offset
        rows = self._db().execute(
            f"{base} ORDER BY rowid LIMIT ? OFFSET ?", [*params, limit, head_offset]
        )
        return [_load_event(r[0]) for r in rows]

    async def get_thread_summaries(
        self, room_id: str, root_event_ids: list[str]
    ) -> dict[str, ThreadSummary]:
        return await self._run(self._x_get_thread_summaries, room_id, list(root_event_ids))

    def _x_get_thread_summaries(
        self, room_id: str, root_event_ids: list[str]
    ) -> dict[str, ThreadSummary]:
        if not root_event_ids:
            return {}
        marks = ",".join("?" for _ in root_event_ids)
        rows = self._db().execute(
            f"""SELECT parent_event_id, COUNT(*), MAX(created_ts) FROM events
                WHERE room_id = ? AND parent_event_id IN ({marks})
                GROUP BY parent_event_id""",  # nosec B608 — fragments are internal, values parameterised
            [room_id, *root_event_ids],
        )
        return {
            root: ThreadSummary(
                root_event_id=root,
                reply_count=count,
                last_reply_at=datetime.fromtimestamp(last_ts, UTC),
            )
            for root, count, last_ts in rows
        }

    async def check_idempotency(self, room_id: str, key: str) -> bool:
        return await self._run(self._x_check_idempotency, room_id, key)

    def _x_check_idempotency(self, room_id: str, key: str) -> bool:
        row = (
            self._db()
            .execute("SELECT 1 FROM idempotency WHERE room_id = ? AND key = ?", (room_id, key))
            .fetchone()
        )
        return row is not None

    async def get_event_by_idempotency_key(self, room_id: str, key: str) -> RoomEvent | None:
        return await self._run(self._x_get_event_by_idempotency_key, room_id, key)

    def _x_get_event_by_idempotency_key(self, room_id: str, key: str) -> RoomEvent | None:
        row = (
            self._db()
            .execute(
                "SELECT data FROM events WHERE room_id = ? AND idempotency_key = ?",
                (room_id, key),
            )
            .fetchone()
        )
        return _load_event(row[0]) if row is not None else None

    async def get_event_count(self, room_id: str) -> int:
        return await self._run(self._x_get_event_count, room_id)

    def _x_get_event_count(self, room_id: str) -> int:
        return (
            self._db()
            .execute("SELECT COUNT(*) FROM events WHERE room_id = ?", (room_id,))
            .fetchone()[0]
        )

    async def add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        return await self._run(self._x_add_event_auto_index, room_id, event)

    def _x_add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        conn = self._db()
        with _write_transaction(conn):
            index = self._reserve_event_index(conn, room_id)
            indexed = event.model_copy(update={"index": index})
            self._write_event_row(conn, indexed)
        return indexed

    async def commit_event(self, room_id: str, event: RoomEvent) -> RoomEvent:
        return await self._run(self._x_commit_event, room_id, event)

    def _x_commit_event(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Atomic commit (RFC §10.1 step 12): index, insert and counter bump
        in one SQLite transaction, so the timeline and the room counters can
        never be observed diverged."""
        conn = self._db()
        with _write_transaction(conn):
            index = self._reserve_event_index(conn, room_id)
            indexed = event.model_copy(update={"index": index})
            self._write_event_row(conn, indexed)
            row = conn.execute("SELECT * FROM rooms WHERE id = ?", (room_id,)).fetchone()
            if row is not None:
                room = self._row_to_room(row)
                timers = room.timers.model_copy(update={"last_activity_at": datetime.now(UTC)})
                room = room.model_copy(
                    update={
                        # The tally, incremented — never a recount (see base).
                        "event_count": room.event_count + 1,
                        "latest_index": indexed.index,
                        "timers": timers,
                    }
                )
                conn.execute(
                    "UPDATE rooms SET data=? WHERE id=?", (room.model_dump_json(), room_id)
                )
        return indexed

    # -- Full-text search --------------------------------------------------

    async def search_events(
        self,
        query: str,
        *,
        room_id: str | None = None,
        limit: int = 20,
    ) -> list[RoomEvent]:
        """Full-text search over stored event text (FTS5, relevance-ranked).

        Not part of the :class:`ConversationStore` contract — an SQLite
        extra. *query* is free text: it is tokenised and matched as an AND
        of terms, so user input can be passed through verbatim.
        """
        return await self._run(self._x_search_events, query, room_id, limit)

    def _x_search_events(self, query: str, room_id: str | None, limit: int) -> list[RoomEvent]:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return []
        match = " ".join(f'"{token}"' for token in tokens)
        where = ["events_fts MATCH ?"]
        params: list[Any] = [match]
        if room_id is not None:
            where.append("events_fts.room_id = ?")
            params.append(room_id)
        params.append(limit)
        rows = self._db().execute(
            f"""SELECT e.data FROM events_fts JOIN events e ON e.id = events_fts.event_id
                WHERE {" AND ".join(where)} ORDER BY events_fts.rank LIMIT ?""",  # nosec B608 — fragments are internal, values parameterised
            params,
        )
        return [_load_event(r[0]) for r in rows]

    # -- Binding operations ------------------------------------------------

    async def add_binding(self, binding: ChannelBinding) -> ChannelBinding:
        return await self._run(self._x_put_binding, binding)

    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        return await self._run(self._x_put_binding, binding)

    def _x_put_binding(self, binding: ChannelBinding) -> ChannelBinding:
        self._db().execute(
            """INSERT INTO bindings(room_id, channel_id, participant_id, channel_type, data)
               VALUES(?, ?, ?, ?, ?)
               ON CONFLICT(room_id, channel_id) DO UPDATE SET
                   participant_id=excluded.participant_id,
                   channel_type=excluded.channel_type, data=excluded.data""",
            (
                binding.room_id,
                binding.channel_id,
                binding.participant_id,
                str(binding.channel_type) if binding.channel_type else None,
                binding.model_dump_json(),
            ),
        )
        return binding

    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        return await self._run(self._x_get_binding, room_id, channel_id)

    def _x_get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        row = (
            self._db()
            .execute(
                "SELECT data FROM bindings WHERE room_id = ? AND channel_id = ?",
                (room_id, channel_id),
            )
            .fetchone()
        )
        return ChannelBinding.model_validate_json(row[0]) if row is not None else None

    async def binding_exists(self, room_id: str, channel_id: str) -> bool:
        return await self._run(self._x_binding_exists, room_id, channel_id)

    def _x_binding_exists(self, room_id: str, channel_id: str) -> bool:
        row = (
            self._db()
            .execute(
                "SELECT 1 FROM bindings WHERE room_id = ? AND channel_id = ?",
                (room_id, channel_id),
            )
            .fetchone()
        )
        return row is not None

    async def remove_binding(self, room_id: str, channel_id: str) -> bool:
        return await self._run(self._x_remove_binding, room_id, channel_id)

    def _x_remove_binding(self, room_id: str, channel_id: str) -> bool:
        cur = self._db().execute(
            "DELETE FROM bindings WHERE room_id = ? AND channel_id = ?", (room_id, channel_id)
        )
        return cur.rowcount > 0

    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        return await self._run(self._x_list_bindings, room_id)

    def _x_list_bindings(self, room_id: str) -> list[ChannelBinding]:
        rows = self._db().execute(
            "SELECT data FROM bindings WHERE room_id = ? ORDER BY rowid", (room_id,)
        )
        return [ChannelBinding.model_validate_json(r[0]) for r in rows]

    # -- Participant operations --------------------------------------------

    async def add_participant(self, participant: Participant) -> Participant:
        return await self._run(self._x_put_participant, participant)

    async def update_participant(self, participant: Participant) -> Participant:
        return await self._run(self._x_put_participant, participant)

    def _x_put_participant(self, participant: Participant) -> Participant:
        self._db().execute(
            """INSERT INTO participants(room_id, id, data) VALUES(?, ?, ?)
               ON CONFLICT(room_id, id) DO UPDATE SET data=excluded.data""",
            (participant.room_id, participant.id, participant.model_dump_json()),
        )
        return participant

    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        return await self._run(self._x_get_participant, room_id, participant_id)

    def _x_get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        row = (
            self._db()
            .execute(
                "SELECT data FROM participants WHERE room_id = ? AND id = ?",
                (room_id, participant_id),
            )
            .fetchone()
        )
        return Participant.model_validate_json(row[0]) if row is not None else None

    async def list_participants(self, room_id: str) -> list[Participant]:
        return await self._run(self._x_list_participants, room_id)

    def _x_list_participants(self, room_id: str) -> list[Participant]:
        rows = self._db().execute(
            "SELECT data FROM participants WHERE room_id = ? ORDER BY rowid", (room_id,)
        )
        return [Participant.model_validate_json(r[0]) for r in rows]

    # -- Read tracking -----------------------------------------------------

    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        await self._run(self._x_mark_read, room_id, channel_id, event_id)

    def _x_mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        self._db().execute(
            """INSERT INTO read_markers(room_id, channel_id, event_id) VALUES(?, ?, ?)
               ON CONFLICT(room_id, channel_id) DO UPDATE SET event_id=excluded.event_id""",
            (room_id, channel_id, event_id),
        )

    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        await self._run(self._x_mark_all_read, room_id, channel_id)

    def _x_mark_all_read(self, room_id: str, channel_id: str) -> None:
        row = (
            self._db()
            .execute(
                "SELECT id FROM events WHERE room_id = ? ORDER BY rowid DESC LIMIT 1", (room_id,)
            )
            .fetchone()
        )
        if row is not None:
            self._x_mark_read(room_id, channel_id, row[0])

    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        return await self._run(self._x_get_unread_count, room_id, channel_id)

    def _x_get_unread_count(self, room_id: str, channel_id: str) -> int:
        conn = self._db()
        total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE room_id = ?", (room_id,)
        ).fetchone()[0]
        marker = conn.execute(
            "SELECT event_id FROM read_markers WHERE room_id = ? AND channel_id = ?",
            (room_id, channel_id),
        ).fetchone()
        if marker is None:
            return total
        anchor = conn.execute(
            "SELECT rowid FROM events WHERE id = ? AND room_id = ?", (marker[0], room_id)
        ).fetchone()
        if anchor is None:
            return total
        return conn.execute(
            "SELECT COUNT(*) FROM events WHERE room_id = ? AND rowid > ?", (room_id, anchor[0])
        ).fetchone()[0]

    async def list_read_markers(self, room_id: str) -> dict[str, int]:
        return await self._run(self._x_list_read_markers, room_id)

    def _x_list_read_markers(self, room_id: str) -> dict[str, int]:
        conn = self._db()
        rows = conn.execute(
            """SELECT rm.channel_id,
                      (SELECT COUNT(*) FROM events prior
                       WHERE prior.room_id = rm.room_id AND prior.rowid < e.rowid)
               FROM read_markers rm JOIN events e ON e.id = rm.event_id AND e.room_id = rm.room_id
               WHERE rm.room_id = ?""",
            (room_id,),
        )
        return {channel_id: position for channel_id, position in rows}

    # -- Identity operations -----------------------------------------------

    async def create_identity(self, identity: Identity) -> Identity:
        return await self._run(self._x_create_identity, identity)

    def _x_create_identity(self, identity: Identity) -> Identity:
        conn = self._db()
        with _write_transaction(conn):
            conn.execute(
                "INSERT OR REPLACE INTO identities(id, data) VALUES(?, ?)",
                (identity.id, identity.model_dump_json()),
            )
            for channel_type, addresses in identity.channel_addresses.items():
                for address in addresses:
                    conn.execute(
                        "INSERT OR REPLACE INTO identity_addresses(channel_type, address,"
                        " identity_id) VALUES(?, ?, ?)",
                        (channel_type, address, identity.id),
                    )
        return identity

    async def get_identity(self, identity_id: str) -> Identity | None:
        return await self._run(self._x_get_identity, identity_id)

    def _x_get_identity(self, identity_id: str) -> Identity | None:
        row = (
            self._db()
            .execute("SELECT data FROM identities WHERE id = ?", (identity_id,))
            .fetchone()
        )
        return Identity.model_validate_json(row[0]) if row is not None else None

    async def resolve_identity(
        self, channel_type: str, address: str, organization_id: str | None = None
    ) -> Identity | None:
        return await self._run(
            self._x_resolve_identity, channel_type, address, organization_id or ""
        )

    def _x_resolve_identity(
        self, channel_type: str, address: str, organization_id: str
    ) -> Identity | None:
        row = (
            self._db()
            .execute(
                "SELECT identity_id FROM identity_addresses WHERE channel_type = ?"
                " AND address = ? AND organization_id = ?",
                (channel_type, address, organization_id),
            )
            .fetchone()
        )
        return self._x_get_identity(row[0]) if row is not None else None

    async def link_address(
        self,
        identity_id: str,
        channel_type: str,
        address: str,
        organization_id: str | None = None,
    ) -> None:
        await self._run(
            self._x_link_address, identity_id, channel_type, address, organization_id or ""
        )

    def _x_link_address(
        self, identity_id: str, channel_type: str, address: str, organization_id: str
    ) -> None:
        conn = self._db()
        conn.execute("BEGIN IMMEDIATE")
        try:
            identity = self._x_get_identity(identity_id)
            if identity is None:
                conn.execute("ROLLBACK")
                return
            current = identity.channel_addresses.get(channel_type, [])
            if address not in current:
                addresses = {**identity.channel_addresses, channel_type: [*current, address]}
                identity = identity.model_copy(update={"channel_addresses": addresses})
                conn.execute(
                    "UPDATE identities SET data = ? WHERE id = ?",
                    (identity.model_dump_json(), identity_id),
                )
            conn.execute(
                "INSERT OR REPLACE INTO identity_addresses(channel_type, address,"
                " identity_id, organization_id) VALUES(?, ?, ?, ?)",
                (channel_type, address, identity_id, organization_id),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # -- Task operations ---------------------------------------------------

    async def add_task(self, task: Task) -> Task:
        return await self._run(self._x_put_task, task)

    async def update_task(self, task: Task) -> Task:
        return await self._run(self._x_put_task, task)

    def _x_put_task(self, task: Task) -> Task:
        self._db().execute(
            """INSERT INTO tasks(id, room_id, status, data) VALUES(?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   room_id=excluded.room_id, status=excluded.status, data=excluded.data""",
            # warnings=False: callers assign plain strings to enum fields
            # (``task.status = "completed"``) and the in-memory store accepts
            # that — reads re-validate, so the value is coerced on the way out.
            (task.id, task.room_id, str(task.status), task.model_dump_json(warnings=False)),
        )
        return task

    async def get_task(self, task_id: str) -> Task | None:
        return await self._run(self._x_get_task, task_id)

    def _x_get_task(self, task_id: str) -> Task | None:
        row = self._db().execute("SELECT data FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return Task.model_validate_json(row[0]) if row is not None else None

    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        return await self._run(self._x_list_tasks, room_id, status)

    def _x_list_tasks(self, room_id: str, status: str | None) -> list[Task]:
        where = ["room_id = ?"]
        params: list[Any] = [room_id]
        if status is not None:
            where.append("status = ?")
            params.append(str(status))
        rows = self._db().execute(
            f"SELECT data FROM tasks WHERE {' AND '.join(where)} ORDER BY rowid",  # nosec B608 — fragments are internal, values parameterised
            params,
        )
        return [Task.model_validate_json(r[0]) for r in rows]

    # -- Observation operations --------------------------------------------

    async def add_observation(self, observation: Observation) -> Observation:
        return await self._run(self._x_add_observation, observation)

    def _x_add_observation(self, observation: Observation) -> Observation:
        self._db().execute(
            "INSERT OR REPLACE INTO observations(id, room_id, data) VALUES(?, ?, ?)",
            (observation.id, observation.room_id, observation.model_dump_json()),
        )
        return observation

    async def list_observations(self, room_id: str) -> list[Observation]:
        return await self._run(self._x_list_observations, room_id)

    def _x_list_observations(self, room_id: str) -> list[Observation]:
        rows = self._db().execute(
            "SELECT data FROM observations WHERE room_id = ? ORDER BY rowid", (room_id,)
        )
        return [Observation.model_validate_json(r[0]) for r in rows]
