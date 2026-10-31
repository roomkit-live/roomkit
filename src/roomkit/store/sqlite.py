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
import json
import re
import sqlite3
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any

from roomkit.core._changes import is_deleted
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import EventStatus, RoomStatus, VerificationStatus
from roomkit.models.event import RoomEvent, ThreadSummary
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter, RoomFilter, includes_blocked
from roomkit.models.task import Observation, Task
from roomkit.models.verification import Verification, VerificationRequest
from roomkit.models.voice_delivery import VoiceDeliveryRecord
from roomkit.store.base import (
    ConversationStore,
    address_linked,
    committed_state,
    deletes_edits,
    is_marked_deleted,
    key_taken,
    link_target,
    room_holds,
    room_taken,
    sender_names,
    target_gone,
    without_address,
)

_SCHEMA_VERSION = 12

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
    data TEXT NOT NULL,
    responds_to TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_room ON events(room_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_room_idx ON events(room_id, idx);
CREATE INDEX IF NOT EXISTS idx_events_room_parent ON events(room_id, parent_event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency_channel
    ON events(room_id, source_channel_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
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
CREATE INDEX IF NOT EXISTS idx_bindings_channel_participant
    ON bindings(channel_id, participant_id);
CREATE TABLE IF NOT EXISTS participants(
    room_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(room_id, id)
);
CREATE INDEX IF NOT EXISTS idx_participants_id ON participants(id);
CREATE INDEX IF NOT EXISTS idx_participants_identity
    ON participants(json_extract(data, '$.identity_id'));
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
CREATE TABLE IF NOT EXISTS verifications(
    identity_id TEXT NOT NULL,
    organization_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(identity_id, organization_id)
);
CREATE TABLE IF NOT EXISTS verification_requests(
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL,
    organization_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    data TEXT NOT NULL
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
    event_index INTEGER NOT NULL,
    PRIMARY KEY(room_id, channel_id)
);
CREATE TABLE IF NOT EXISTS voice_deliveries(
    room_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(room_id, key_hash)
);
CREATE TABLE IF NOT EXISTS idempotency(
    room_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    key TEXT NOT NULL,
    PRIMARY KEY(room_id, channel_id, key)
);
CREATE INDEX IF NOT EXISTS idx_idempotency_room_key ON idempotency(room_id, key);
CREATE TABLE IF NOT EXISTS private_state(
    room_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(room_id, key)
);
CREATE INDEX IF NOT EXISTS idx_events_room_idempotency
    ON events(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
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


def _key_taken(conn: sqlite3.Connection, event: RoomEvent) -> bool:
    """Whether *event*'s insert was refused on its key, which its source
    channel holds in the room for another event (RFC §13.4), and not on its
    id: an id the store holds is the database's own refusal."""
    if not event.idempotency_key:
        return False
    if conn.execute("SELECT 1 FROM events WHERE id = ?", (event.id,)).fetchone() is not None:
        return False
    held = conn.execute(
        "SELECT 1 FROM events WHERE room_id = ? AND source_channel_id = ? AND idempotency_key = ?",
        (event.room_id, event.source.channel_id, event.idempotency_key),
    ).fetchone()
    return held is not None


def _mark_edits_deleted(conn: sqlite3.Connection, room_id: str, target_id: str) -> None:
    """Mark every edit of *target_id* deleted, in the caller's transaction:
    an edit holds a version of the deleted text (RFC §10.3)."""
    conn.execute(
        "UPDATE events SET data = json_set(data, '$.metadata.deleted', json('true'))"
        " WHERE room_id = ? AND json_extract(data, '$.content.type') = 'edit'"
        " AND json_extract(data, '$.content.target_event_id') = ?",
        (room_id, target_id),
    )


@contextmanager
def _write_transaction(conn: sqlite3.Connection) -> Generator[None, None, None]:
    """Run one immediate SQLite transaction with reliable rollback."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _filter_rooms(where: list[str], params: list[Any], room_filter: RoomFilter | None) -> None:
    """Narrow a routing lookup joined on ``rooms r`` to the statuses and the
    organization *room_filter* names (RFC §10.4, §14.1, §17.2)."""
    if room_filter is None:
        return
    if room_filter.statuses is not None:
        where.append(f"r.status IN ({', '.join('?' * len(room_filter.statuses))})")
        params.extend(status.value for status in room_filter.statuses)
    if room_filter.organization_id is not None:
        where.append("r.organization_id = ?")
        params.append(room_filter.organization_id)


def _placeholders(values: list[str]) -> str:
    """``?, ?`` for *values*, the parameters of an ``IN`` list."""
    return ", ".join("?" * len(values))


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
        # latest_index survives the deletion of the room's highest event, so
        # the next index never reuses a deleted one.
        historical_next = room.latest_index + 1 if room.latest_index is not None else 0
        conn.execute(
            """INSERT INTO event_sequences(room_id, next_index) VALUES(?, ?)
               ON CONFLICT(room_id) DO UPDATE SET
                   next_index = MAX(event_sequences.next_index, excluded.next_index)""",
            (room_id, historical_next),
        )


_RESPONDS_TO_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_events_room_responds_to ON events(room_id, responds_to);"
)
"""Kept out of ``_SCHEMA``: the older migrations replay it on an events table that
has no ``responds_to`` column yet, which an index on it would fail."""


def _create_schema(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{_SCHEMA}\n{_RESPONDS_TO_INDEX}")
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
        conn.execute("PRAGMA user_version=2")
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
        conn.execute("PRAGMA user_version=3")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """An answer names the event it answers (RFC §8.5): ``events`` gains a
    ``responds_to`` column, indexed per room. No event written before v4 carries
    one, so there is nothing to backfill: old rows read ``NULL``."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        if "responds_to" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN responds_to TEXT")
        conn.execute(_RESPONDS_TO_INDEX)
        conn.execute("PRAGMA user_version=4")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Identity verification (RFC §11.7): two new tables, nothing to backfill."""
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{_SCHEMA}")
        conn.execute("PRAGMA user_version=5")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """A read marker holds the read event's ``index``, a high-water mark, not
    its id: an id's position shifted when an earlier event was deleted, and
    write order is not index order. The table is rebuilt; a marker whose
    event no longer exists is dropped, which reads as before (all unread)."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(read_markers)")}
        if "event_index" not in columns:
            conn.execute("ALTER TABLE read_markers RENAME TO read_markers_v5")
            conn.execute(
                """CREATE TABLE read_markers(
                    room_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    PRIMARY KEY(room_id, channel_id)
                )"""
            )
            conn.execute(
                """INSERT INTO read_markers(room_id, channel_id, event_index)
                   SELECT rm.room_id, rm.channel_id, e.idx FROM read_markers_v5 rm
                   JOIN events e ON e.id = rm.event_id AND e.room_id = rm.room_id"""
            )
            conn.execute("DROP TABLE read_markers_v5")
        conn.execute("PRAGMA user_version=6")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """An event's visibility is stamped at its commit (RFC §7.3): its own
    intersected with its source binding's. An event stored ``all`` before
    read its source binding's when history was rebuilt; it takes that
    binding's now, once, so the stored field is the whole answer and its
    readers stay the same. A message held for verification (§11.7.3)
    records the visibility its release restores; a recorded ``all`` takes
    the binding's too."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        # Another process may have migrated the file since it was opened.
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= 7:
            conn.execute("COMMIT")
            return
        conn.execute(
            """UPDATE events SET data = json_set(
                   data, '$.metadata."roomkit:verification_held".visibility', (
                       SELECT json_extract(b.data, '$.visibility') FROM bindings b
                       WHERE b.room_id = events.room_id
                         AND b.channel_id = events.source_channel_id))
               WHERE json_extract(
                       data, '$.metadata."roomkit:verification_held".visibility') = 'all'
                 AND EXISTS (
                     SELECT 1 FROM bindings b
                     WHERE b.room_id = events.room_id
                       AND b.channel_id = events.source_channel_id
                       AND COALESCE(json_extract(b.data, '$.visibility'), 'all') <> 'all')"""
        )
        conn.execute(
            """UPDATE events SET
                   visibility = (
                       SELECT json_extract(b.data, '$.visibility') FROM bindings b
                       WHERE b.room_id = events.room_id
                         AND b.channel_id = events.source_channel_id),
                   data = json_set(data, '$.visibility', (
                       SELECT json_extract(b.data, '$.visibility') FROM bindings b
                       WHERE b.room_id = events.room_id
                         AND b.channel_id = events.source_channel_id))
               WHERE visibility = 'all'
                 AND EXISTS (
                     SELECT 1 FROM bindings b
                     WHERE b.room_id = events.room_id
                       AND b.channel_id = events.source_channel_id
                       AND COALESCE(json_extract(b.data, '$.visibility'), 'all') <> 'all')"""
        )
        conn.execute("PRAGMA user_version=7")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    """An idempotency key is unique per room and per source channel (RFC
    §13.4): the key table and the events' unique index name the channel.
    Keys unique per room are unique per room and channel, so nothing
    conflicts."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= 8:
            conn.execute("COMMIT")
            return
        conn.execute("DROP INDEX IF EXISTS idx_events_idempotency")
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency_channel
               ON events(room_id, source_channel_id, idempotency_key)
               WHERE idempotency_key IS NOT NULL"""
        )
        conn.execute("DROP TABLE IF EXISTS idempotency")
        conn.execute(
            """CREATE TABLE idempotency(
                   room_id TEXT NOT NULL,
                   channel_id TEXT NOT NULL,
                   key TEXT NOT NULL,
                   PRIMARY KEY(room_id, channel_id, key)
               )"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO idempotency(room_id, channel_id, key)
               SELECT room_id, source_channel_id, idempotency_key FROM events
               WHERE idempotency_key IS NOT NULL AND source_channel_id IS NOT NULL"""
        )
        conn.execute("PRAGMA user_version=8")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v8_to_v9(conn: sqlite3.Connection) -> None:
    """A room records the instant it paused (RFC §5.1), which its close
    timer counts from. A room stored paused before it takes its last
    modification, which the pause was."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= 9:
            conn.execute("COMMIT")
            return
        conn.execute(
            """UPDATE rooms
               SET data = json_set(data, '$.paused_at', json_extract(data, '$.updated_at'))
               WHERE status = 'paused' AND json_extract(data, '$.paused_at') IS NULL"""
        )
        conn.execute("PRAGMA user_version=9")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v9_to_v10(conn: sqlite3.Connection) -> None:
    """A proactive delivery's key is looked up within its room, whatever
    channel committed it (RFC §22.1): its events and its keys are indexed by
    room and key."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= 10:
            conn.execute("COMMIT")
            return
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_idempotency_room_key ON idempotency(room_id, key)"
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_events_room_idempotency
               ON events(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL"""
        )
        conn.execute("PRAGMA user_version=10")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v10_to_v11(conn: sqlite3.Connection) -> None:
    """A soft-deleted message's text is withheld (RFC §10.3): no search
    finds it. A file from before kept it in the full-text index, which the
    delete's rewrite of the message filled again: its rows leave the index,
    the records keep their text. A sender's participant lookup names it by
    its identity too (RFC §10.4 step 1), through an index on the identity."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= 11:
            conn.execute("COMMIT")
            return
        conn.execute(
            """DELETE FROM events_fts WHERE event_id IN (
                   SELECT id FROM events
                   WHERE json_type(data, '$.metadata.deleted') = 'true')"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_participants_identity
               ON participants(json_extract(data, '$.identity_id'))"""
        )
        conn.execute("PRAGMA user_version=11")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _migrate_v11_to_v12(conn: sqlite3.Connection) -> None:
    """A room's private state (RFC §14.1) gets a table of its own, apart from
    its metadata, which clients may read, and the room's register of authors
    moves there (RFC §10.1 step 12): its digests name the room's senders to
    whoever tries every phone number or name. One that is not a map is
    dropped, as its readers read it absent."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= 12:
            conn.execute("COMMIT")
            return
        conn.execute(
            """CREATE TABLE IF NOT EXISTS private_state(
                   room_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                   PRIMARY KEY(room_id, key))"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO private_state(room_id, key, value)
               SELECT id, 'author_register', json_extract(data, '$.metadata.author_register')
               FROM rooms WHERE json_type(data, '$.metadata.author_register') = 'object'"""
        )
        conn.execute(
            """UPDATE rooms SET data = json_remove(data, '$.metadata.author_register')
               WHERE json_type(data, '$.metadata.author_register') IS NOT NULL"""
        )
        conn.execute("PRAGMA user_version=12")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,
    5: _migrate_v5_to_v6,
    6: _migrate_v6_to_v7,
    7: _migrate_v7_to_v8,
    8: _migrate_v8_to_v9,
    9: _migrate_v9_to_v10,
    10: _migrate_v10_to_v11,
    11: _migrate_v11_to_v12,
}
"""Each step takes a file from its version to the next."""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > _SCHEMA_VERSION:
        raise SQLiteSchemaError(
            f"SQLite schema version {version} is newer than supported version "
            f"{_SCHEMA_VERSION}; upgrade RoomKit before opening this file"
        )
    if version == 0:
        _create_schema(conn)
    elif version in _MIGRATIONS:
        for step in range(version, _SCHEMA_VERSION):
            _MIGRATIONS[step](conn)
    elif version == _SCHEMA_VERSION:
        # Additive repair for a partially initialized file. The version is
        # never rewritten, and all statements are idempotent.
        conn.executescript(f"{_SCHEMA}\n{_RESPONDS_TO_INDEX}")
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

    async def get_voice_delivery(self, room_id: str, key_hash: str) -> VoiceDeliveryRecord | None:
        def read() -> VoiceDeliveryRecord | None:
            row = (
                self._db()
                .execute(
                    "SELECT data FROM voice_deliveries WHERE room_id = ? AND key_hash = ?",
                    (room_id, key_hash),
                )
                .fetchone()
            )
            return VoiceDeliveryRecord.model_validate_json(row["data"]) if row else None

        return await self._run(read)

    async def claim_voice_delivery(self, record: VoiceDeliveryRecord) -> VoiceDeliveryRecord:
        if record.outcome is not None:
            raise ValueError("A new claim must have no outcome")

        def claim() -> VoiceDeliveryRecord:
            conn = self._db()
            with _write_transaction(conn):
                if (
                    conn.execute("SELECT 1 FROM rooms WHERE id = ?", (record.room_id,)).fetchone()
                    is None
                ):
                    raise ValueError("Room does not exist")
                row = conn.execute(
                    "SELECT data FROM voice_deliveries WHERE room_id = ? AND key_hash = ?",
                    (record.room_id, record.key_hash),
                ).fetchone()
                if row:
                    current = VoiceDeliveryRecord.model_validate_json(row["data"])
                    if not current.retryable or current.content_hash != record.content_hash:
                        return current
                conn.execute(
                    "INSERT INTO voice_deliveries(room_id, key_hash, data) VALUES (?, ?, ?) "
                    "ON CONFLICT(room_id, key_hash) DO UPDATE SET data = excluded.data",
                    (record.room_id, record.key_hash, record.model_dump_json()),
                )
                return record.model_copy(deep=True)

        return await self._run(claim)

    async def complete_voice_delivery(self, record: VoiceDeliveryRecord) -> bool:
        if record.outcome is None:
            raise ValueError("Completion requires an outcome")

        def complete() -> bool:
            conn = self._db()
            with _write_transaction(conn):
                row = conn.execute(
                    "SELECT data FROM voice_deliveries WHERE room_id = ? AND key_hash = ?",
                    (record.room_id, record.key_hash),
                ).fetchone()
                if row is None:
                    return False
                current = VoiceDeliveryRecord.model_validate_json(row["data"])
                if current.attempt_id != record.attempt_id:
                    return False
                if current.outcome is not None:
                    return current.outcome == record.outcome
                completed = current.model_copy(update={"outcome": record.outcome})
                conn.execute(
                    "UPDATE voice_deliveries SET data = ? WHERE room_id = ? AND key_hash = ?",
                    (completed.model_dump_json(), record.room_id, record.key_hash),
                )
                return True

        return await self._run(complete)

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
                           parent_event_id, idempotency_key, created_ts, data,
                           responds_to)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    # A rewrite of a stored row keeps its delivery results: only
    # record_delivery_results writes them (RFC §10.3, §14.3).
    _EVENT_UPSERT = """
        INSERT INTO events(id, room_id, idx, type, visibility, source_channel_id,
                           source_channel_type, participant_id, correlation_id,
                           parent_event_id, idempotency_key, created_ts, data,
                           responds_to)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            room_id=excluded.room_id, idx=excluded.idx, type=excluded.type,
            visibility=excluded.visibility,
            source_channel_id=excluded.source_channel_id,
            source_channel_type=excluded.source_channel_type,
            participant_id=excluded.participant_id,
            correlation_id=excluded.correlation_id,
            parent_event_id=excluded.parent_event_id,
            idempotency_key=excluded.idempotency_key,
            created_ts=excluded.created_ts,
            data=json_set(
                excluded.data, '$.delivery_results',
                json(COALESCE(json_extract(events.data, '$.delivery_results'), '{}'))
            ),
            responds_to=excluded.responds_to
    """

    def _write_event_row(
        self,
        conn: sqlite3.Connection,
        event: RoomEvent,
        *,
        replace: bool = False,
    ) -> None:
        source = event.source
        row = (
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
            event.responds_to,
        )
        try:
            conn.execute(self._EVENT_UPSERT if replace else self._EVENT_INSERT, row)
        except sqlite3.IntegrityError as exc:
            if not replace and _key_taken(conn, event):
                raise key_taken(event) from exc
            raise
        if event.idempotency_key:
            conn.execute(
                "INSERT OR IGNORE INTO idempotency(room_id, channel_id, key) VALUES(?, ?, ?)",
                (event.room_id, event.source.channel_id, event.idempotency_key),
            )
        conn.execute("DELETE FROM events_fts WHERE event_id = ?", (event.id,))
        # A soft-deleted event's text is withheld (RFC §10.3): no search finds it.
        text = "" if is_deleted(event) else _fts_text(event)
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
        with _write_transaction(conn):
            # Never REPLACE: the room an id names is created once (§14.3).
            created = conn.execute(
                "INSERT OR IGNORE INTO rooms(id, created_ts, organization_id, status,"
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
            if created.rowcount == 0:
                raise room_taken(room.id)
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
        conn = self._db()
        with _write_transaction(conn):
            return self._write_room_row(conn, room)

    async def update_room_if(
        self,
        room: Room,
        *,
        status: RoomStatus,
        paused_at: datetime | None,
        last_activity_at: datetime | None,
    ) -> Room | None:
        return await self._run(self._x_update_room_if, room, status, paused_at, last_activity_at)

    def _x_update_room_if(
        self,
        room: Room,
        status: RoomStatus,
        paused_at: datetime | None,
        last_activity_at: datetime | None,
    ) -> Room | None:
        """Atomic compare-and-set: the comparison and the write in one
        immediate transaction, which a commit's waits for (RFC §5.1)."""
        conn = self._db()
        with _write_transaction(conn):
            row = conn.execute("SELECT * FROM rooms WHERE id = ?", (room.id,)).fetchone()
            if row is None or not room_holds(
                self._row_to_room(row), status, paused_at, last_activity_at
            ):
                return None
            return self._write_room_row(conn, room)

    def _write_room_row(self, conn: sqlite3.Connection, room: Room) -> Room:
        """Write *room*'s row inside the caller's transaction.

        delivered_index, event_count and latest_index are store-managed
        (advance_delivered_index, commit_event): the room written carries
        the stored ones, so a stale copy cannot rewind them.
        """
        stored = conn.execute(
            "SELECT delivered_index, json_extract(data, '$.event_count'),"
            " json_extract(data, '$.latest_index') FROM rooms WHERE id=?",
            (room.id,),
        ).fetchone()
        if stored is None:
            raise RoomNotFoundError(room.id)
        room = room.model_copy(
            update={
                "delivered_index": stored[0],
                "event_count": stored[1],
                # A room stored before latest_index was nullable kept 0
                # with no event; that 0 names no event (RFC §5.1).
                "latest_index": stored[2] if stored[1] else None,
            }
        )
        conn.execute(
            "UPDATE rooms SET created_ts=?, organization_id=?, status=?, data=? WHERE id=?",
            (
                _ts(room.created_at),
                room.organization_id,
                str(room.status),
                room.model_dump_json(),
                room.id,
            ),
        )
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

    async def get_private_state(self, room_id: str, key: str) -> dict[str, Any] | None:
        return await self._run(self._x_get_private_state, room_id, key)

    def _x_get_private_state(self, room_id: str, key: str) -> dict[str, Any] | None:
        row = (
            self._db()
            .execute(
                "SELECT value FROM private_state WHERE room_id = ? AND key = ?", (room_id, key)
            )
            .fetchone()
        )
        return json.loads(row["value"]) if row is not None else None

    async def set_private_state(
        self, room_id: str, key: str, value: dict[str, Any] | None
    ) -> None:
        written = json.dumps(value) if value is not None else None
        await self._run(self._x_set_private_state, room_id, key, written)

    def _x_set_private_state(self, room_id: str, key: str, value: str | None) -> None:
        conn = self._db()
        with _write_transaction(conn):
            row = conn.execute("SELECT data FROM rooms WHERE id = ?", (room_id,)).fetchone()
            if row is None:
                return
            if value is None:
                conn.execute(
                    "DELETE FROM private_state WHERE room_id = ? AND key = ?", (room_id, key)
                )
            else:
                conn.execute(
                    """INSERT INTO private_state(room_id, key, value) VALUES(?, ?, ?)
                       ON CONFLICT(room_id, key) DO UPDATE SET value = excluded.value""",
                    (room_id, key, value),
                )
            room = json.loads(row["data"])
            if key in room.get("metadata", {}):
                del room["metadata"][key]
                conn.execute("UPDATE rooms SET data = ? WHERE id = ?", (json.dumps(room), room_id))

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
                "voice_deliveries",
                "event_sequences",
                "events_fts",
                "private_state",
            ):
                conn.execute(f"DELETE FROM {table} WHERE room_id = ?", (room_id,))  # nosec B608  # fragments are internal, values parameterised
        return existed

    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        return await self._run(self._x_list_rooms, offset, limit)

    def _x_list_rooms(self, offset: int, limit: int) -> list[Room]:
        rows = self._db().execute(
            "SELECT * FROM rooms ORDER BY created_ts, id LIMIT ? OFFSET ?", (limit, offset)
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
        query = f"SELECT * FROM rooms WHERE {' AND '.join(where)} ORDER BY created_ts, id"  # nosec B608  # fragments are internal, values parameterised
        if not metadata_filter:
            rows = self._db().execute(f"{query} LIMIT ? OFFSET ?", [*params, limit, offset])
            return [self._row_to_room(r) for r in rows]
        # Metadata values are arbitrary JSON — compare in Python rather than
        # fight json_extract over type affinities, reading no further than
        # the page.
        with closing(self._db().execute(query, params)) as rows:
            matching = (
                room
                for room in map(self._row_to_room, rows)
                if all(room.metadata.get(k) == v for k, v in metadata_filter.items())
            )
            return list(islice(matching, offset, offset + limit))

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
                f"SELECT * FROM rooms WHERE {' AND '.join(where)}"  # nosec B608  # fragments are internal, values parameterised
                " ORDER BY created_ts DESC, id DESC LIMIT 1",
                params,
            )
            .fetchone()
        )
        return self._row_to_room(row) if row is not None else None

    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        statuses = None if status is None else [RoomStatus(status)]
        matches = await self.find_room_ids_by_channel(channel_id, RoomFilter(statuses=statuses), 1)
        return matches[0] if matches else None

    async def find_room_id_by_binding(
        self,
        channel_id: str,
        participant_id: str,
        identity_id: str | None = None,
        room_filter: RoomFilter | None = None,
        *,
        recipient_key: str | None = None,
    ) -> str | None:
        return await self._run(
            self._x_find_room_id_by_binding,
            channel_id,
            participant_id,
            identity_id,
            room_filter,
            recipient_key,
        )

    def _x_find_room_id_by_binding(
        self,
        channel_id: str,
        participant_id: str,
        identity_id: str | None,
        room_filter: RoomFilter | None,
        recipient_key: str | None,
    ) -> str | None:
        names = sorted(sender_names(participant_id, identity_id))
        # Two branches, each read through idx_bindings_channel_participant:
        # the bindings naming one of the names, and the channel's bindings
        # naming no one (an OR of both would read every binding of the
        # channel).
        where = ["b.channel_id = ?", f"b.participant_id IN ({_placeholders(names)})"]
        params: list[Any] = [channel_id, *names]
        _filter_rooms(where, params, room_filter)
        branches = [where]
        if recipient_key is not None and participant_id.strip():
            # A binding naming no correspondent, not a group, names its
            # recipient (RFC §5.7): its value under the key, a string that
            # is not blank or an integer, read whatever characters the key
            # holds. The value is compared to the address, so a blank
            # address finds none.
            unnamed = [
                "b.channel_id = ?",
                "b.participant_id IS NULL",
                "NOT COALESCE(json_extract(b.data, '$.group'), 0)",
                "EXISTS (SELECT 1 FROM json_each(b.data, '$.metadata')"
                " WHERE key = ? AND type IN ('text', 'integer')"
                " AND CAST(value AS TEXT) = ?)",
            ]
            params += [channel_id, recipient_key, participant_id]
            _filter_rooms(unnamed, params, room_filter)
            branches.append(unnamed)
        found = " UNION ALL ".join(
            "SELECT r.id, r.created_ts FROM bindings b JOIN rooms r ON r.id = b.room_id"
            f" WHERE {' AND '.join(branch)}"  # nosec B608  # fragments are internal, values parameterised
            for branch in branches
        )
        row = (
            self._db()
            .execute(
                f"SELECT id FROM ({found}) ORDER BY created_ts DESC, id DESC LIMIT 1",  # nosec B608  # fragments are internal, values parameterised
                params,
            )
            .fetchone()
        )
        return row[0] if row is not None else None

    async def find_room_id_by_participant(
        self,
        channel_id: str,
        participant_id: str,
        identity_id: str | None = None,
        room_filter: RoomFilter | None = None,
    ) -> str | None:
        return await self._run(
            self._x_find_room_id_by_participant,
            channel_id,
            participant_id,
            identity_id,
            room_filter,
        )

    def _x_find_room_id_by_participant(
        self,
        channel_id: str,
        participant_id: str,
        identity_id: str | None,
        room_filter: RoomFilter | None,
    ) -> str | None:
        # Named by its id or its identity, and reached through the channel
        # whatever its status (RFC §5.5, §10.4 step 1). Two branches, one
        # read through idx_participants_id and one through
        # idx_participants_identity: an OR of both reads every participant.
        names = sorted(sender_names(participant_id, identity_id))
        params: list[Any] = []
        branches: list[list[str]] = []
        for named in ("p.id", "json_extract(p.data, '$.identity_id')"):
            where = [
                f"{named} IN ({_placeholders(names)})",
                "EXISTS (SELECT 1 FROM json_each(p.data, '$.connected_via') WHERE value = ?)",
            ]
            params += [*names, channel_id]
            _filter_rooms(where, params, room_filter)
            branches.append(where)
        found = " UNION ALL ".join(
            "SELECT r.id, r.created_ts FROM participants p JOIN rooms r ON r.id = p.room_id"
            f" WHERE {' AND '.join(branch)}"  # nosec B608  # fragments are internal, values parameterised
            for branch in branches
        )
        row = (
            self._db()
            .execute(
                f"SELECT id FROM ({found}) ORDER BY created_ts DESC, id DESC LIMIT 1",  # nosec B608  # fragments are internal, values parameterised
                params,
            )
            .fetchone()
        )
        return row[0] if row is not None else None

    async def find_room_ids_by_channel(
        self,
        channel_id: str,
        room_filter: RoomFilter | None = None,
        limit: int = 2,
        *,
        group: bool | None = None,
    ) -> list[str]:
        return await self._run(
            self._x_find_room_ids_by_channel, channel_id, room_filter, limit, group
        )

    def _x_find_room_ids_by_channel(
        self, channel_id: str, room_filter: RoomFilter | None, limit: int, group: bool | None
    ) -> list[str]:
        where = ["b.channel_id = ?"]
        params: list[Any] = [channel_id]
        _filter_rooms(where, params, room_filter)
        if group is not None:
            where.append("COALESCE(json_extract(b.data, '$.group'), 0) = ?")
            params.append(int(group))
        params.append(limit)
        rows = self._db().execute(
            f"""SELECT r.id FROM bindings b JOIN rooms r ON r.id = b.room_id
                WHERE {" AND ".join(where)}
                ORDER BY r.created_ts, r.id LIMIT ?""",  # nosec B608  # fragments are internal, values parameterised
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

    async def write_change_target(self, room_id: str, target: RoomEvent) -> None:
        await self._run(self._x_write_change_target, room_id, target)

    def _x_write_change_target(self, room_id: str, target: RoomEvent) -> None:
        """The target's row, and its edits' marks when it is now deleted, in
        one SQLite transaction; no row of the change (RFC §10.3)."""
        conn = self._db()
        with _write_transaction(conn):
            held = conn.execute(
                "SELECT 1 FROM events WHERE id = ? AND room_id = ?", (target.id, room_id)
            ).fetchone()
            if held is None:
                raise target_gone(target)
            self._write_event_row(conn, target, replace=True)
            if is_marked_deleted(target):
                _mark_edits_deleted(conn, room_id, target.id)

    async def record_delivery_results(
        self, room_id: str, event_id: str, results: dict[str, Any]
    ) -> None:
        await self._run(self._x_record_delivery_results, room_id, event_id, results)

    def _x_record_delivery_results(
        self, room_id: str, event_id: str, results: dict[str, Any]
    ) -> None:
        """The results alone, written into the stored row's JSON in place: the
        rest of it, a change committed since included, stays as stored."""
        conn = self._db()
        with _write_transaction(conn):
            conn.execute(
                "UPDATE events SET data = json_set(data, '$.delivery_results', json(?))"
                " WHERE id = ? AND room_id = ?",
                (json.dumps(results), event_id, room_id),
            )

    async def delete_event(
        self, room_id: str, event_id: str, *, cascade_replies: bool = True
    ) -> list[RoomEvent]:
        return await self._run(self._x_delete_event, room_id, event_id, cascade_replies)

    def _x_delete_event(
        self, room_id: str, event_id: str, cascade_replies: bool
    ) -> list[RoomEvent]:
        conn = self._db()
        with _write_transaction(conn):
            root = conn.execute(
                "SELECT data FROM events WHERE id = ? AND room_id = ?", (event_id, room_id)
            ).fetchone()
            if root is None:
                return []
            removed = [_load_event(root[0])]
            if cascade_replies:
                rows = conn.execute(
                    "SELECT data FROM events WHERE room_id = ? AND parent_event_id = ?"
                    " AND id != ? ORDER BY idx",
                    (room_id, event_id, event_id),
                )
                removed.extend(_load_event(row[0]) for row in rows)
            # The edits and deletes of every event removed (RFC §10.3).
            gone = [event.id for event in removed]
            marks = ",".join("?" for _ in gone)
            rows = conn.execute(
                "SELECT data FROM events WHERE room_id = ?"
                " AND json_extract(data, '$.content.type') IN ('edit', 'delete')"
                f" AND json_extract(data, '$.content.target_event_id') IN ({marks})"  # nosec B608  # fragments are internal, values parameterised
                f" AND id NOT IN ({marks}) ORDER BY idx",
                [room_id, *gone, *gone],
            )
            removed.extend(_load_event(row[0]) for row in rows)
            deleted = [event.id for event in removed]
            marks = ",".join("?" for _ in deleted)
            keys = conn.execute(
                f"SELECT source_channel_id, idempotency_key FROM events WHERE id IN ({marks})"  # nosec B608  # fragments are internal, values parameterised
                " AND idempotency_key IS NOT NULL",
                deleted,
            ).fetchall()
            for channel_id, key in keys:
                conn.execute(
                    "DELETE FROM idempotency WHERE room_id = ? AND channel_id = ? AND key = ?",
                    (room_id, channel_id, key),
                )
            conn.execute(f"DELETE FROM events WHERE id IN ({marks})", deleted)  # nosec B608  # fragments are internal, values parameterised
            conn.execute(f"DELETE FROM events_fts WHERE event_id IN ({marks})", deleted)  # nosec B608  # fragments are internal, values parameterised
        return removed

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
        self._apply_event_filter_sql(ef, where, params, visibility=visibility)

        base = f"SELECT data FROM events WHERE {' AND '.join(where)}"  # nosec B608  # fragments are internal, values parameterised
        if before_index is not None or (
            before_index is None and after_index is None and newest_first
        ):
            # Tail page: newest ``limit`` rows (offset counted from the newest
            # end), returned in chronological order. Rendered by ``idx``
            # whatever the write order (RFC §14.1); ``rowid`` breaks ties
            # between rows stored without an index.
            tail_offset = 0 if before_index is not None else offset
            rows = self._db().execute(
                f"{base} ORDER BY idx DESC, rowid DESC LIMIT ? OFFSET ?",
                [*params, limit, tail_offset],
            )
            return [_load_event(r[0]) for r in rows][::-1]
        head_offset = 0 if after_index is not None else offset
        rows = self._db().execute(
            f"{base} ORDER BY idx, rowid LIMIT ? OFFSET ?", [*params, limit, head_offset]
        )
        return [_load_event(r[0]) for r in rows]

    @staticmethod
    def _apply_event_filter_sql(
        ef: EventFilter | None,
        where: list[str],
        params: list[Any],
        *,
        visibility: str | None = None,
    ) -> None:
        """Append the row conditions of a page, or of the count that stands for
        it, under ``ef``.

        The one constructor behind :meth:`list_events` and
        :meth:`get_event_count`, so the two never diverge on a criterion: the
        visibility the caller resolved, then the filter's own fields, then the
        received-rows default (RFC §14.1) unless ``include_blocked`` lifts it.
        ``status`` lives in the JSON document; it is a SQL condition all the
        same, so a page is cut after the refused rows are gone, not before.
        """
        if visibility is not None:
            where.append("visibility = ?")
            params.append(visibility)
        if ef is None:
            where.append("json_extract(data, '$.status') IS NOT ?")
            params.append(EventStatus.BLOCKED.value)
            return
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
        if ef.responds_to is not None:
            where.append("responds_to = ?")
            params.append(ef.responds_to)
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
        if not includes_blocked(ef):
            where.append("json_extract(data, '$.status') IS NOT ?")
            params.append(EventStatus.BLOCKED.value)

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
                  AND json_extract(data, '$.status') IS NOT ?
                GROUP BY parent_event_id""",  # nosec B608  # fragments are internal, values parameterised
            [room_id, *root_event_ids, EventStatus.BLOCKED.value],
        )
        return {
            root: ThreadSummary(
                root_event_id=root,
                reply_count=count,
                last_reply_at=datetime.fromtimestamp(last_ts, UTC),
            )
            for root, count, last_ts in rows
        }

    async def check_idempotency(self, room_id: str, key: str, *, channel_id: str | None) -> bool:
        return await self._run(self._x_check_idempotency, room_id, channel_id, key)

    def _x_check_idempotency(self, room_id: str, channel_id: str | None, key: str) -> bool:
        # No channel: any channel of the room, a proactive key's scope (RFC
        # §22.1), read through idx_idempotency_room_key.
        if channel_id is None:
            query = "SELECT 1 FROM idempotency WHERE room_id = ? AND key = ? LIMIT 1"
            params: tuple[str, ...] = (room_id, key)
        else:
            query = "SELECT 1 FROM idempotency WHERE room_id = ? AND channel_id = ? AND key = ?"
            params = (room_id, channel_id, key)
        return self._db().execute(query, params).fetchone() is not None

    async def get_event_by_idempotency_key(
        self, room_id: str, key: str, *, channel_id: str | None
    ) -> RoomEvent | None:
        return await self._run(self._x_get_event_by_idempotency_key, room_id, channel_id, key)

    def _x_get_event_by_idempotency_key(
        self, room_id: str, channel_id: str | None, key: str
    ) -> RoomEvent | None:
        # No channel: any channel of the room, the lowest index first, read
        # through idx_events_room_idempotency.
        if channel_id is None:
            # "+idx": sorting the few rows of the key, not reading the room in
            # index order for them.
            query = (
                "SELECT data FROM events WHERE room_id = ? AND idempotency_key = ?"
                " ORDER BY +idx LIMIT 1"
            )
            params: tuple[str, ...] = (room_id, key)
        else:
            query = (
                "SELECT data FROM events WHERE room_id = ? AND source_channel_id = ?"
                " AND idempotency_key = ?"
            )
            params = (room_id, channel_id, key)
        row = self._db().execute(query, params).fetchone()
        return _load_event(row[0]) if row is not None else None

    async def get_event_count(self, room_id: str, event_filter: EventFilter | None = None) -> int:
        return await self._run(self._x_get_event_count, room_id, event_filter)

    def _x_get_event_count(self, room_id: str, ef: EventFilter | None = None) -> int:
        where = ["room_id = ?"]
        params: list[Any] = [room_id]
        if ef is not None:
            # The conditions of a ``list_events`` page under this filter, and
            # no page (RFC §14.1).
            self._apply_event_filter_sql(ef, where, params, visibility=ef.visibility)
        query = f"SELECT COUNT(*) FROM events WHERE {' AND '.join(where)}"  # nosec B608  # fragments are internal, values parameterised
        return self._db().execute(query, params).fetchone()[0]

    async def add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        return await self._run(self._x_add_event_auto_index, room_id, event)

    def _x_add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        conn = self._db()
        with _write_transaction(conn):
            index = self._reserve_event_index(conn, room_id)
            indexed = event.model_copy(update={"index": index})
            self._write_event_row(conn, indexed)
        return indexed

    async def commit_event(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        target: RoomEvent | None = None,
        activity: bool = True,
    ) -> RoomEvent:
        return await self._run(self._x_commit_event, room_id, event, target, activity)

    def _x_commit_event(
        self,
        room_id: str,
        event: RoomEvent,
        target: RoomEvent | None = None,
        activity: bool = True,
    ) -> RoomEvent:
        """Atomic commit (RFC §10.1 step 12): index, insert and counter bump
        in one SQLite transaction, so the timeline and the room counters can
        never be observed diverged; a change's *target* is written in it, or
        the whole transaction rolls back when the room no longer holds it
        (§14.3). Only an *activity* commit moves ``last_activity_at`` and
        resumes a PAUSED room (RFC §5.1)."""
        conn = self._db()
        with _write_transaction(conn):
            if target is not None:
                held = conn.execute(
                    "SELECT 1 FROM events WHERE id = ? AND room_id = ?", (target.id, room_id)
                ).fetchone()
                if held is None:
                    raise target_gone(target)
                self._write_event_row(conn, target, replace=True)
                if deletes_edits(event, target):
                    _mark_edits_deleted(conn, room_id, target.id)
            index = self._reserve_event_index(conn, room_id)
            indexed = event.model_copy(update={"index": index})
            self._write_event_row(conn, indexed)
            row = conn.execute("SELECT * FROM rooms WHERE id = ?", (room_id,)).fetchone()
            if row is not None:
                room = self._row_to_room(row)
                room = room.model_copy(
                    update={
                        # The tally, incremented — never a recount (see base).
                        "event_count": room.event_count + 1,
                        "latest_index": indexed.index,
                        **committed_state(room, activity),
                    }
                )
                conn.execute(
                    "UPDATE rooms SET status=?, data=? WHERE id=?",
                    (str(room.status), room.model_dump_json(), room_id),
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
        of terms, so user input can be passed through verbatim. Rows the room
        refused (stored ``BLOCKED``) are not returned, as a default
        :meth:`list_events` read does not serve them (RFC §14.1); the audit
        read is :meth:`list_events` with ``include_blocked=True``. A message an
        accepted delete marked is not indexed either (RFC §10.3): its record
        keeps its text, which no search finds.
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
        # The received-rows rule reaches the index too (RFC §14.1): a body a
        # hook refused is out of the timeline by default, and not findable.
        where.append("json_extract(e.data, '$.status') IS NOT ?")
        params.append(EventStatus.BLOCKED.value)
        params.append(limit)
        rows = self._db().execute(
            f"""SELECT e.data FROM events_fts JOIN events e ON e.id = events_fts.event_id
                WHERE {" AND ".join(where)} ORDER BY events_fts.rank LIMIT ?""",  # nosec B608  # fragments are internal, values parameterised
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

    async def delete_participant(self, room_id: str, participant_id: str) -> None:
        await self._run(self._x_delete_participant, room_id, participant_id)

    def _x_delete_participant(self, room_id: str, participant_id: str) -> None:
        self._db().execute(
            "DELETE FROM participants WHERE room_id = ? AND id = ?", (room_id, participant_id)
        )

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
        # An event of another room, or none, marks nothing; a marker is a
        # high-water mark, it never moves back.
        self._db().execute(
            """INSERT INTO read_markers(room_id, channel_id, event_index)
               SELECT ?, ?, idx FROM events WHERE id = ? AND room_id = ?
               ON CONFLICT(room_id, channel_id) DO UPDATE
               SET event_index = MAX(read_markers.event_index, excluded.event_index)""",
            (room_id, channel_id, event_id, room_id),
        )

    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        await self._run(self._x_mark_all_read, room_id, channel_id)

    def _x_mark_all_read(self, room_id: str, channel_id: str) -> None:
        # An empty room gets no marker: it makes no group, so no row. (HAVING
        # without GROUP BY needs SQLite 3.39, newer than some systems ship.)
        self._db().execute(
            """INSERT INTO read_markers(room_id, channel_id, event_index)
               SELECT ?, ?, MAX(idx) FROM events WHERE room_id = ? GROUP BY room_id
               ON CONFLICT(room_id, channel_id) DO UPDATE
               SET event_index = MAX(read_markers.event_index, excluded.event_index)""",
            (room_id, channel_id, room_id),
        )

    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        return await self._run(self._x_get_unread_count, room_id, channel_id)

    def _x_get_unread_count(self, room_id: str, channel_id: str) -> int:
        conn = self._db()
        total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE room_id = ?", (room_id,)
        ).fetchone()[0]
        marker = conn.execute(
            "SELECT event_index FROM read_markers WHERE room_id = ? AND channel_id = ?",
            (room_id, channel_id),
        ).fetchone()
        if marker is None:
            return total
        return conn.execute(
            "SELECT COUNT(*) FROM events WHERE room_id = ? AND idx > ?", (room_id, marker[0])
        ).fetchone()[0]

    async def list_read_markers(self, room_id: str) -> dict[str, int]:
        return await self._run(self._x_list_read_markers, room_id)

    def _x_list_read_markers(self, room_id: str) -> dict[str, int]:
        conn = self._db()
        rows = conn.execute(
            "SELECT channel_id, event_index FROM read_markers WHERE room_id = ?", (room_id,)
        )
        return {channel_id: index for channel_id, index in rows}

    # -- Identity operations -----------------------------------------------

    async def create_identity(self, identity: Identity) -> Identity:
        return await self._run(self._x_create_identity, identity)

    def _x_create_identity(self, identity: Identity) -> Identity:
        conn = self._db()
        scope = identity.organization_id or ""
        with _write_transaction(conn):
            for channel_type, addresses in identity.channel_addresses.items():
                for address in addresses:
                    self._x_refuse_held_address(conn, channel_type, address, scope, identity.id)
            conn.execute(
                "INSERT OR REPLACE INTO identities(id, data) VALUES(?, ?)",
                (identity.id, identity.model_dump_json()),
            )
            for channel_type, addresses in identity.channel_addresses.items():
                for address in addresses:
                    conn.execute(
                        "INSERT OR REPLACE INTO identity_addresses(channel_type, address,"
                        " identity_id, organization_id) VALUES(?, ?, ?, ?)",
                        # The identity's own organization: an address declared
                        # on the identity belongs to the tenant that owns it.
                        # Omitting it files the address under the empty
                        # organization, where the owning tenant cannot resolve
                        # it and an unscoped caller can.
                        (channel_type, address, identity.id, scope),
                    )
        return identity

    @staticmethod
    def _x_refuse_held_address(
        conn: sqlite3.Connection,
        channel_type: str,
        address: str,
        organization_id: str,
        identity_id: str,
    ) -> None:
        """Refuse an address another identity of the organization holds (RFC §5.6)."""
        row = conn.execute(
            "SELECT identity_id FROM identity_addresses WHERE channel_type = ?"
            " AND address = ? AND organization_id = ?",
            (channel_type, address, organization_id),
        ).fetchone()
        if row is not None and row[0] != identity_id:
            raise address_linked(channel_type)

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
        with _write_transaction(conn):
            identity = link_target(self._x_get_identity(identity_id), identity_id, organization_id)
            self._x_refuse_held_address(conn, channel_type, address, organization_id, identity_id)
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

    async def unlink_address(
        self,
        identity_id: str,
        channel_type: str,
        address: str,
        organization_id: str | None = None,
    ) -> None:
        await self._run(
            self._x_unlink_address, identity_id, channel_type, address, organization_id or ""
        )

    def _x_unlink_address(
        self, identity_id: str, channel_type: str, address: str, organization_id: str
    ) -> None:
        conn = self._db()
        with _write_transaction(conn):
            identity = link_target(self._x_get_identity(identity_id), identity_id, organization_id)
            # Its own row only: an address another identity holds stays theirs.
            conn.execute(
                "DELETE FROM identity_addresses WHERE channel_type = ? AND address = ?"
                " AND organization_id = ? AND identity_id = ?",
                (channel_type, address, organization_id, identity_id),
            )
            addresses = without_address(identity.channel_addresses, channel_type, address)
            if addresses != identity.channel_addresses:
                identity = identity.model_copy(update={"channel_addresses": addresses})
                conn.execute(
                    "UPDATE identities SET data = ? WHERE id = ?",
                    (identity.model_dump_json(), identity_id),
                )

    # -- Verification operations (RFC §11.7) -------------------------------

    async def get_verification(
        self, identity_id: str, organization_id: str | None = None
    ) -> Verification | None:
        return await self._run(self._x_get_verification, identity_id, organization_id or "")

    def _x_get_verification(self, identity_id: str, organization_id: str) -> Verification | None:
        row = (
            self._db()
            .execute(
                "SELECT data FROM verifications WHERE identity_id = ? AND organization_id = ?",
                (identity_id, organization_id),
            )
            .fetchone()
        )
        return Verification.model_validate_json(row[0]) if row is not None else None

    async def put_verification(self, verification: Verification) -> Verification:
        return await self._run(self._x_put_verification, verification)

    def _x_put_verification(self, verification: Verification) -> Verification:
        conn = self._db()
        with _write_transaction(conn):
            conn.execute(
                "INSERT OR REPLACE INTO verifications(identity_id, organization_id, status, data)"
                " VALUES(?, ?, ?, ?)",
                (
                    verification.identity_id,
                    verification.organization_id or "",
                    str(verification.status),
                    verification.model_dump_json(),
                ),
            )
        return verification

    async def list_verifications_in_force(self) -> list[Verification]:
        return await self._run(self._x_list_verifications_in_force)

    def _x_list_verifications_in_force(self) -> list[Verification]:
        rows = (
            self._db()
            .execute(
                "SELECT data FROM verifications WHERE status = ?",
                (str(VerificationStatus.VERIFIED),),
            )
            .fetchall()
        )
        verifications = [Verification.model_validate_json(row[0]) for row in rows]
        return [v for v in verifications if v.expires_at is not None]

    async def create_verification_request(
        self, request: VerificationRequest
    ) -> VerificationRequest:
        return await self._run(self._x_save_verification_request, request, True)

    async def update_verification_request(
        self, request: VerificationRequest
    ) -> VerificationRequest:
        return await self._run(self._x_save_verification_request, request, False)

    def _x_save_verification_request(
        self, request: VerificationRequest, create: bool
    ) -> VerificationRequest:
        conn = self._db()
        row = (
            request.id,
            request.identity_id,
            request.organization_id or "",
            str(request.status),
            request.model_dump_json(),
        )
        with _write_transaction(conn):
            if create:
                conn.execute(
                    "INSERT INTO verification_requests(id, identity_id, organization_id,"
                    " status, data) VALUES(?, ?, ?, ?, ?)",
                    row,
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO verification_requests(id, identity_id,"
                    " organization_id, status, data) VALUES(?, ?, ?, ?, ?)",
                    row,
                )
        return request

    async def get_verification_request(self, request_id: str) -> VerificationRequest | None:
        return await self._run(self._x_get_verification_request, request_id)

    def _x_get_verification_request(self, request_id: str) -> VerificationRequest | None:
        row = (
            self._db()
            .execute("SELECT data FROM verification_requests WHERE id = ?", (request_id,))
            .fetchone()
        )
        return VerificationRequest.model_validate_json(row[0]) if row is not None else None

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
            f"SELECT data FROM tasks WHERE {' AND '.join(where)} ORDER BY rowid",  # nosec B608  # fragments are internal, values parameterised
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
