"""PostgreSQL implementation of ConversationStore using asyncpg.

Uses a proper relational schema with indexed columns for efficient
querying. All RoomKit models map to dedicated columns — no JSONB blob
storage for core fields.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.event import ChannelData, EventSource, RoomEvent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room, RoomTimers
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Observation, Task
from roomkit.store.base import ConversationStore

_SCHEMA = """\
-- rooms
CREATE TABLE IF NOT EXISTS rooms (
    id              TEXT PRIMARY KEY,
    organization_id TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    event_count     INTEGER NOT NULL DEFAULT 0,
    latest_index    INTEGER NOT NULL DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{}',
    timers          JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_rooms_org ON rooms(organization_id);
CREATE INDEX IF NOT EXISTS idx_rooms_status ON rooms(status);
CREATE INDEX IF NOT EXISTS idx_rooms_updated ON rooms(updated_at DESC);

-- events
CREATE TABLE IF NOT EXISTS events (
    id                  TEXT PRIMARY KEY,
    room_id             TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    type                TEXT NOT NULL DEFAULT 'message',
    content             JSONB NOT NULL,
    source_channel_id   TEXT NOT NULL,
    source_channel_type TEXT NOT NULL,
    source_direction    TEXT NOT NULL DEFAULT 'inbound',
    source_participant_id TEXT,
    source_provider     TEXT,
    source_extra        JSONB NOT NULL DEFAULT '{}',
    status              TEXT NOT NULL DEFAULT 'pending',
    visibility          TEXT NOT NULL DEFAULT 'all',
    response_visibility TEXT,
    index               INTEGER NOT NULL DEFAULT 0,
    chain_depth         INTEGER NOT NULL DEFAULT 0,
    correlation_id      TEXT,
    parent_event_id     TEXT,
    idempotency_key     TEXT,
    blocked_by          TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    channel_data        JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_room_index ON events(room_id, index);
CREATE INDEX IF NOT EXISTS idx_events_room_created ON events(room_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_room_type ON events(room_id, type);
CREATE INDEX IF NOT EXISTS idx_events_correlation
    ON events(correlation_id) WHERE correlation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_channel_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency
    ON events(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL;

-- bindings
CREATE TABLE IF NOT EXISTS bindings (
    channel_id    TEXT NOT NULL,
    room_id       TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_type  TEXT NOT NULL,
    category      TEXT NOT NULL DEFAULT 'transport',
    direction     TEXT NOT NULL DEFAULT 'bidirectional',
    access        TEXT NOT NULL DEFAULT 'read_write',
    muted         BOOLEAN NOT NULL DEFAULT FALSE,
    output_muted  BOOLEAN NOT NULL DEFAULT FALSE,
    visibility    TEXT DEFAULT 'all',
    participant_id TEXT,
    last_read_index INTEGER,
    attached_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    capabilities  JSONB NOT NULL DEFAULT '{}',
    metadata      JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (room_id, channel_id)
);
CREATE INDEX IF NOT EXISTS idx_bindings_channel_id ON bindings(channel_id);

-- participants
CREATE TABLE IF NOT EXISTS participants (
    id             TEXT NOT NULL,
    room_id        TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_id     TEXT NOT NULL,
    display_name   TEXT,
    role           TEXT NOT NULL DEFAULT 'member',
    status         TEXT NOT NULL DEFAULT 'active',
    identification TEXT NOT NULL DEFAULT 'pending',
    identity_id    TEXT,
    external_id    TEXT,
    joined_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at    TIMESTAMPTZ,
    resolved_by    TEXT,
    metadata       JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (room_id, id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_channel
    ON participants(room_id, channel_id);

-- identities
CREATE TABLE IF NOT EXISTS identities (
    id                TEXT PRIMARY KEY,
    organization_id   TEXT,
    display_name      TEXT,
    email             TEXT,
    channel_addresses JSONB NOT NULL DEFAULT '{}',
    metadata          JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS identity_addresses (
    channel_type TEXT NOT NULL,
    address      TEXT NOT NULL,
    identity_id  TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    PRIMARY KEY (channel_type, address)
);

-- tasks
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    room_id     TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    description TEXT,
    assigned_to TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tasks_room_id ON tasks(room_id);

-- observations
CREATE TABLE IF NOT EXISTS observations (
    id          TEXT PRIMARY KEY,
    room_id     TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_id  TEXT,
    content     TEXT NOT NULL,
    category    TEXT,
    confidence  REAL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_observations_room_id ON observations(room_id);

-- read tracking
CREATE TABLE IF NOT EXISTS read_markers (
    room_id    TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_id TEXT NOT NULL,
    event_index INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (room_id, channel_id)
);

-- schema version
CREATE TABLE IF NOT EXISTS schema_version (
    version INT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO schema_version (version)
    SELECT 2 WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version >= 2);
"""


# ── Row ↔ Model helpers ────────────────────────────────────────


def _row_to_room(row: Any) -> Room:
    """Convert a database row to a Room model."""
    return Room(
        id=row["id"],
        organization_id=row["organization_id"],
        status=row["status"],
        event_count=row["event_count"],
        latest_index=row["latest_index"],
        metadata=(
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        ),
        timers=RoomTimers.model_validate(
            json.loads(row["timers"]) if isinstance(row["timers"], str) else row["timers"]
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        closed_at=row["closed_at"],
    )


def _row_to_event(row: Any) -> RoomEvent:
    """Convert a database row to a RoomEvent model."""
    content_raw = row["content"]
    content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
    source_extra = row["source_extra"]
    if isinstance(source_extra, str):
        source_extra = json.loads(source_extra)
    channel_data_raw = row["channel_data"]
    if isinstance(channel_data_raw, str):
        channel_data_raw = json.loads(channel_data_raw)
    metadata_raw = row["metadata"]
    if isinstance(metadata_raw, str):
        metadata_raw = json.loads(metadata_raw)
    return RoomEvent(
        id=row["id"],
        room_id=row["room_id"],
        type=row["type"],
        content=content,
        source=EventSource(
            channel_id=row["source_channel_id"],
            channel_type=row["source_channel_type"],
            direction=row["source_direction"],
            participant_id=row["source_participant_id"],
            provider=row["source_provider"],
            raw_payload=source_extra.get("raw_payload", {}),
            external_id=source_extra.get("external_id"),
            provider_message_id=source_extra.get("provider_message_id"),
        ),
        status=row["status"],
        visibility=row["visibility"],
        response_visibility=row["response_visibility"],
        index=row["index"],
        chain_depth=row["chain_depth"],
        correlation_id=row["correlation_id"],
        parent_event_id=row["parent_event_id"],
        idempotency_key=row["idempotency_key"],
        blocked_by=row["blocked_by"],
        metadata=metadata_raw,
        channel_data=ChannelData.model_validate(channel_data_raw),
        created_at=row["created_at"],
    )


def _row_to_binding(row: Any) -> ChannelBinding:
    """Convert a database row to a ChannelBinding model."""
    caps = row["capabilities"]
    if isinstance(caps, str):
        caps = json.loads(caps)
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return ChannelBinding(
        channel_id=row["channel_id"],
        room_id=row["room_id"],
        channel_type=row["channel_type"],
        category=row["category"],
        direction=row["direction"],
        access=row["access"],
        muted=row["muted"],
        output_muted=row["output_muted"],
        visibility=row["visibility"],
        participant_id=row["participant_id"],
        last_read_index=row["last_read_index"],
        attached_at=row["attached_at"],
        capabilities=caps,
        metadata=meta,
    )


def _row_to_participant(row: Any) -> Participant:
    """Convert a database row to a Participant model."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Participant(
        id=row["id"],
        room_id=row["room_id"],
        channel_id=row["channel_id"],
        display_name=row["display_name"],
        role=row["role"],
        status=row["status"],
        identification=row["identification"],
        identity_id=row["identity_id"],
        external_id=row["external_id"],
        joined_at=row["joined_at"],
        resolved_at=row["resolved_at"],
        resolved_by=row["resolved_by"],
        metadata=meta,
    )


def _row_to_task(row: Any) -> Task:
    """Convert a database row to a Task model."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Task(
        id=row["id"],
        room_id=row["room_id"],
        title=row["title"],
        description=row["description"],
        assigned_to=row["assigned_to"],
        status=row["status"],
        created_at=row["created_at"],
        metadata=meta,
    )


def _row_to_observation(row: Any) -> Observation:
    """Convert a database row to an Observation model."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Observation(
        id=row["id"],
        room_id=row["room_id"],
        channel_id=row["channel_id"],
        content=row["content"],
        category=row["category"],
        confidence=row["confidence"],
        created_at=row["created_at"],
        metadata=meta,
    )


def _source_extra(source: EventSource) -> dict[str, Any]:
    """Extract non-column source fields into a JSONB dict."""
    extra: dict[str, Any] = {}
    if source.raw_payload:
        extra["raw_payload"] = source.raw_payload
    if source.external_id:
        extra["external_id"] = source.external_id
    if source.provider_message_id:
        extra["provider_message_id"] = source.provider_message_id
    return extra


class PostgresStore(ConversationStore):
    """PostgreSQL-backed conversation store using asyncpg.

    Uses a fully relational schema with indexed columns for all
    frequently queried fields. JSONB is used only for flexible/extensible
    data (metadata, content, capabilities).
    """

    def __init__(
        self,
        dsn: str | None = None,
        pool: Any = None,
    ) -> None:
        try:
            import asyncpg as _asyncpg
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PostgresStore. "
                "Install it with: pip install roomkit[postgres]"
            ) from exc
        self._asyncpg = _asyncpg
        self._dsn = dsn
        self._pool = pool
        self._owns_pool = pool is None

    _acquire_timeout: float = 5.0

    def _ensure_pool(self) -> Any:
        """Return the pool or raise if not initialized."""
        if self._pool is None:
            raise RuntimeError("PostgresStore.init() must be called before use")
        return self._pool

    def _acquire(self) -> Any:
        """Acquire a connection from the pool with a timeout."""
        return self._ensure_pool().acquire(timeout=self._acquire_timeout)

    @contextmanager
    def _query_span(self, operation: str, table: str) -> Generator[str, None, None]:
        """Context manager for STORE_QUERY telemetry spans."""
        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.context import get_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        span_id = telemetry.start_span(
            SpanKind.STORE_QUERY,
            f"store.{operation}",
            parent_id=get_current_span(),
            attributes={
                Attr.STORE_OPERATION: operation,
                Attr.STORE_TABLE: table,
            },
        )
        try:
            yield span_id
            telemetry.end_span(span_id)
        except Exception as exc:
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise

    async def init(self, min_size: int = 2, max_size: int = 10) -> None:
        """Create the connection pool (if needed) and ensure schema exists."""
        if self._pool is None:
            self._pool = await self._asyncpg.create_pool(
                self._dsn,
                min_size=min_size,
                max_size=max_size,
            )
        async with self._acquire() as conn:
            await conn.execute(_SCHEMA)

    async def close(self) -> None:
        """Release the connection pool if we own it."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> PostgresStore:
        await self.init()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Room operations ──────────────────────────────────────────

    async def create_room(self, room: Room) -> Room:
        with self._query_span("create_room", "rooms"):
            async with self._acquire() as conn:
                await conn.execute(
                    "INSERT INTO rooms "
                    "(id, organization_id, status, event_count, latest_index,"
                    " metadata, timers, created_at, updated_at, closed_at)"
                    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
                    room.id,
                    room.organization_id,
                    room.status.value,
                    room.event_count,
                    room.latest_index,
                    room.metadata,
                    room.timers.model_dump(),
                    room.created_at,
                    room.updated_at,
                    room.closed_at,
                )
        return room

    async def get_room(self, room_id: str) -> Room | None:
        with self._query_span("get_room", "rooms"):
            async with self._acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM rooms WHERE id = $1", room_id)
        if row is None:
            return None
        return _row_to_room(row)

    async def update_room(self, room: Room) -> Room:
        with self._query_span("update_room", "rooms"):
            async with self._acquire() as conn:
                await conn.execute(
                    "UPDATE rooms SET organization_id=$2, status=$3, event_count=$4,"
                    " latest_index=$5, metadata=$6, timers=$7, updated_at=$8, closed_at=$9"
                    " WHERE id=$1",
                    room.id,
                    room.organization_id,
                    room.status.value,
                    room.event_count,
                    room.latest_index,
                    room.metadata,
                    room.timers.model_dump(),
                    room.updated_at,
                    room.closed_at,
                )
        return room

    async def delete_room(self, room_id: str) -> bool:
        with self._query_span("delete_room", "rooms"):
            async with self._acquire() as conn:
                tag = await conn.execute("DELETE FROM rooms WHERE id = $1", room_id)
        return bool(tag == "DELETE 1")

    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        with self._query_span("list_rooms", "rooms"):
            async with self._acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM rooms ORDER BY created_at LIMIT $1 OFFSET $2",
                    limit,
                    offset,
                )
        return [_row_to_room(r) for r in rows]

    async def find_rooms(
        self,
        organization_id: str | None = None,
        status: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Room]:
        clauses: list[str] = []
        params: list[Any] = []
        idx = 1

        if organization_id is not None:
            clauses.append(f"organization_id = ${idx}")
            params.append(organization_id)
            idx += 1
        if status is not None:
            status_val = status.value if hasattr(status, "value") else status
            clauses.append(f"status = ${idx}")
            params.append(status_val)
            idx += 1
        if metadata_filter:
            clauses.append(f"metadata @> ${idx}::jsonb")
            params.append(json.dumps(metadata_filter))
            idx += 1

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        query = (
            f"SELECT * FROM rooms {where}"  # nosec B608
            f" ORDER BY created_at LIMIT ${idx} OFFSET ${idx + 1}"
        )
        params.extend([limit, offset])
        with self._query_span("find_rooms", "rooms"):
            async with self._acquire() as conn:
                rows = await conn.fetch(query, *params)
        return [_row_to_room(r) for r in rows]

    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        params: list[Any] = [participant_id]
        idx = 2

        base = (
            "SELECT r.* FROM rooms r WHERE ("
            "  EXISTS (SELECT 1 FROM participants p WHERE p.room_id = r.id AND p.id = $1)"
            "  OR EXISTS ("
            "    SELECT 1 FROM bindings b WHERE b.room_id = r.id AND b.participant_id = $1"
        )
        if channel_type is not None:
            ct_val = channel_type.value if hasattr(channel_type, "value") else channel_type
            base += f" AND b.channel_type = ${idx}"
            params.append(ct_val)
            idx += 1
        base += "  ))"

        if status is not None:
            status_val = status.value if hasattr(status, "value") else status
            base += f" AND r.status = ${idx}"
            params.append(status_val)
            idx += 1

        base += " ORDER BY r.created_at DESC LIMIT 1"

        async with self._acquire() as conn:
            row = await conn.fetchrow(base, *params)
        if row is None:
            return None
        return _row_to_room(row)

    async def find_room_id_by_channel(
        self,
        channel_id: str,
        status: str | None = None,
    ) -> str | None:
        if status is not None:
            status_val = status.value if hasattr(status, "value") else status
            query = (
                "SELECT b.room_id FROM bindings b "
                "JOIN rooms r ON r.id = b.room_id "
                "WHERE b.channel_id = $1 AND r.status = $2 LIMIT 1"
            )
            async with self._acquire() as conn:
                row = await conn.fetchrow(query, channel_id, status_val)
        else:
            async with self._acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT room_id FROM bindings WHERE channel_id = $1 LIMIT 1",
                    channel_id,
                )
        return row["room_id"] if row else None

    # ── Event operations ─────────────────────────────────────────

    async def add_event(self, event: RoomEvent) -> RoomEvent:
        with self._query_span("add_event", "events"):
            async with self._acquire() as conn:
                await conn.execute(
                    "INSERT INTO events"
                    " (id, room_id, type, content, source_channel_id, source_channel_type,"
                    "  source_direction, source_participant_id, source_provider, source_extra,"
                    "  status, visibility, response_visibility, index, chain_depth,"
                    "  correlation_id, parent_event_id, idempotency_key, blocked_by,"
                    "  metadata, channel_data, created_at)"
                    " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,"
                    "         $16,$17,$18,$19,$20,$21,$22)",
                    event.id,
                    event.room_id,
                    event.type.value,
                    event.content.model_dump(),
                    event.source.channel_id,
                    event.source.channel_type.value,
                    event.source.direction.value,
                    event.source.participant_id,
                    event.source.provider,
                    _source_extra(event.source),
                    event.status.value,
                    event.visibility,
                    event.response_visibility,
                    event.index,
                    event.chain_depth,
                    event.correlation_id,
                    event.parent_event_id,
                    event.idempotency_key,
                    event.blocked_by,
                    event.metadata,
                    event.channel_data.model_dump(),
                    event.created_at,
                )
        return event

    async def get_event(self, event_id: str) -> RoomEvent | None:
        with self._query_span("get_event", "events"):
            async with self._acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM events WHERE id = $1", event_id)
        if row is None:
            return None
        return _row_to_event(row)

    async def update_event(self, event: RoomEvent) -> RoomEvent:
        with self._query_span("update_event", "events"):
            async with self._acquire() as conn:
                await conn.execute(
                    "UPDATE events SET content=$2, status=$3, visibility=$4,"
                    " metadata=$5, idempotency_key=$6"
                    " WHERE id=$1",
                    event.id,
                    event.content.model_dump(),
                    event.status.value,
                    event.visibility,
                    event.metadata,
                    event.idempotency_key,
                )
        return event

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
    ) -> list[RoomEvent]:
        if after_index is not None and before_index is not None:
            raise ValueError("after_index and before_index are mutually exclusive")

        use_cursor = after_index is not None or before_index is not None

        conditions = ["room_id = $1"]
        params: list[object] = [room_id]
        idx = 2

        # Visibility: event_filter.visibility takes precedence
        effective_visibility = (
            event_filter.visibility
            if event_filter is not None and event_filter.visibility is not None
            else visibility_filter
        )
        if effective_visibility is not None:
            conditions.append(f"visibility = ${idx}")
            params.append(effective_visibility)
            idx += 1

        if after_index is not None:
            conditions.append(f"index > ${idx}")
            params.append(after_index)
            idx += 1
        elif before_index is not None:
            conditions.append(f"index < ${idx}")
            params.append(before_index)
            idx += 1

        if event_filter is not None:
            idx = self._apply_event_filter_sql(event_filter, conditions, params, idx)

        where = " AND ".join(conditions)

        order_cols = {
            "before": "index DESC",
            "after": "index",
            "default": "created_at",
        }
        if before_index is not None:
            order_col = order_cols["before"]
        elif after_index is not None:
            order_col = order_cols["after"]
        else:
            order_col = order_cols["default"]

        query = f"SELECT * FROM events WHERE {where} ORDER BY {order_col}"  # nosec B608

        query += f" LIMIT ${idx}"
        params.append(limit)
        idx += 1

        if not use_cursor:
            query += f" OFFSET ${idx}"
            params.append(offset)

        with self._query_span("list_events", "events"):
            async with self._acquire() as conn:
                rows = await conn.fetch(query, *params)

        events = [_row_to_event(r) for r in rows]
        if before_index is not None:
            events.reverse()
        return events

    @staticmethod
    def _apply_event_filter_sql(
        ef: EventFilter,
        conditions: list[str],
        params: list[object],
        idx: int,
    ) -> int:
        """Append SQL conditions for EventFilter fields. Returns next param index."""
        if ef.event_types is not None:
            conditions.append(f"type = ANY(${idx})")
            params.append([t.value for t in ef.event_types])
            idx += 1
        if ef.exclude_types is not None:
            conditions.append(f"type != ALL(${idx})")
            params.append([t.value for t in ef.exclude_types])
            idx += 1
        if ef.source_channel_id is not None:
            conditions.append(f"source_channel_id = ${idx}")
            params.append(ef.source_channel_id)
            idx += 1
        if ef.source_channel_type is not None:
            conditions.append(f"source_channel_type = ${idx}")
            params.append(ef.source_channel_type.value)
            idx += 1
        if ef.correlation_id is not None:
            conditions.append(f"correlation_id = ${idx}")
            params.append(ef.correlation_id)
            idx += 1
        if ef.participant_id is not None:
            conditions.append(f"source_participant_id = ${idx}")
            params.append(ef.participant_id)
            idx += 1
        if ef.after_time is not None:
            conditions.append(f"created_at > ${idx}")
            params.append(ef.after_time)
            idx += 1
        if ef.before_time is not None:
            conditions.append(f"created_at < ${idx}")
            params.append(ef.before_time)
            idx += 1
        return idx

    async def check_idempotency(self, room_id: str, key: str) -> bool:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM events WHERE room_id = $1 AND idempotency_key = $2",
                room_id,
                key,
            )
        return row is not None

    async def get_event_count(self, room_id: str) -> int:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                "SELECT count(*) AS cnt FROM events WHERE room_id = $1",
                room_id,
            )
        return row["cnt"] if row else 0

    async def add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Atomically assign the next index and store the event in one transaction.

        Uses ``SELECT ... FOR UPDATE`` on the rooms table to serialise
        concurrent index assignments for the same room.
        """
        with self._query_span("add_event_auto_index", "events"):
            async with self._acquire() as conn, conn.transaction():
                await conn.fetchrow(
                    "SELECT id FROM rooms WHERE id = $1 FOR UPDATE",
                    room_id,
                )
                row = await conn.fetchrow(
                    "SELECT COALESCE(MAX(index), -1) + 1 AS next_idx"
                    " FROM events WHERE room_id = $1",
                    room_id,
                )
                next_idx = row["next_idx"]
                indexed = event.model_copy(update={"index": next_idx})
                await conn.execute(
                    "INSERT INTO events"
                    " (id, room_id, type, content, source_channel_id, source_channel_type,"
                    "  source_direction, source_participant_id, source_provider, source_extra,"
                    "  status, visibility, response_visibility, index, chain_depth,"
                    "  correlation_id, parent_event_id, idempotency_key, blocked_by,"
                    "  metadata, channel_data, created_at)"
                    " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,"
                    "         $16,$17,$18,$19,$20,$21,$22)",
                    indexed.id,
                    indexed.room_id,
                    indexed.type.value,
                    indexed.content.model_dump(),
                    indexed.source.channel_id,
                    indexed.source.channel_type.value,
                    indexed.source.direction.value,
                    indexed.source.participant_id,
                    indexed.source.provider,
                    _source_extra(indexed.source),
                    indexed.status.value,
                    indexed.visibility,
                    indexed.response_visibility,
                    indexed.index,
                    indexed.chain_depth,
                    indexed.correlation_id,
                    indexed.parent_event_id,
                    indexed.idempotency_key,
                    indexed.blocked_by,
                    indexed.metadata,
                    indexed.channel_data.model_dump(),
                    indexed.created_at,
                )
        return indexed

    # ── Binding operations ───────────────────────────────────────

    async def add_binding(self, binding: ChannelBinding) -> ChannelBinding:
        with self._query_span("add_binding", "bindings"):
            async with self._acquire() as conn:
                await conn.execute(
                    "INSERT INTO bindings"
                    " (channel_id, room_id, channel_type, category, direction,"
                    "  access, muted, output_muted, visibility, participant_id,"
                    "  last_read_index, attached_at, capabilities, metadata)"
                    " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)"
                    " ON CONFLICT (room_id, channel_id) DO UPDATE SET"
                    "  channel_type=$3, category=$4, direction=$5, access=$6,"
                    "  muted=$7, output_muted=$8, visibility=$9, participant_id=$10,"
                    "  last_read_index=$11, capabilities=$13, metadata=$14",
                    binding.channel_id,
                    binding.room_id,
                    binding.channel_type.value,
                    binding.category.value,
                    binding.direction.value,
                    binding.access.value,
                    binding.muted,
                    binding.output_muted,
                    binding.visibility,
                    binding.participant_id,
                    binding.last_read_index,
                    binding.attached_at,
                    binding.capabilities.model_dump(),
                    binding.metadata,
                )
        return binding

    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        with self._query_span("get_binding", "bindings"):
            async with self._acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM bindings WHERE room_id = $1 AND channel_id = $2",
                    room_id,
                    channel_id,
                )
        if row is None:
            return None
        return _row_to_binding(row)

    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        with self._query_span("update_binding", "bindings"):
            async with self._acquire() as conn:
                await conn.execute(
                    "UPDATE bindings SET channel_type=$3, category=$4, direction=$5,"
                    " access=$6, muted=$7, output_muted=$8, visibility=$9,"
                    " participant_id=$10, last_read_index=$11, capabilities=$12, metadata=$13"
                    " WHERE room_id=$1 AND channel_id=$2",
                    binding.room_id,
                    binding.channel_id,
                    binding.channel_type.value,
                    binding.category.value,
                    binding.direction.value,
                    binding.access.value,
                    binding.muted,
                    binding.output_muted,
                    binding.visibility,
                    binding.participant_id,
                    binding.last_read_index,
                    binding.capabilities.model_dump(),
                    binding.metadata,
                )
        return binding

    async def remove_binding(self, room_id: str, channel_id: str) -> bool:
        with self._query_span("remove_binding", "bindings"):
            async with self._acquire() as conn:
                tag = await conn.execute(
                    "DELETE FROM bindings WHERE room_id = $1 AND channel_id = $2",
                    room_id,
                    channel_id,
                )
        return bool(tag == "DELETE 1")

    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        with self._query_span("list_bindings", "bindings"):
            async with self._acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM bindings WHERE room_id = $1",
                    room_id,
                )
        return [_row_to_binding(r) for r in rows]

    # ── Participant operations ───────────────────────────────────

    async def add_participant(self, participant: Participant) -> Participant:
        with self._query_span("add_participant", "participants"):
            async with self._acquire() as conn:
                await conn.execute(
                    "INSERT INTO participants"
                    " (id, room_id, channel_id, display_name, role, status,"
                    "  identification, identity_id, external_id, joined_at,"
                    "  resolved_at, resolved_by, metadata)"
                    " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)"
                    " ON CONFLICT (room_id, id) DO UPDATE SET"
                    "  channel_id=$3, display_name=$4, role=$5, status=$6,"
                    "  identification=$7, identity_id=$8, external_id=$9,"
                    "  resolved_at=$11, resolved_by=$12, metadata=$13",
                    participant.id,
                    participant.room_id,
                    participant.channel_id,
                    participant.display_name,
                    participant.role.value,
                    participant.status.value,
                    participant.identification.value,
                    participant.identity_id,
                    participant.external_id,
                    participant.joined_at,
                    participant.resolved_at,
                    participant.resolved_by,
                    participant.metadata,
                )
        return participant

    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        with self._query_span("get_participant", "participants"):
            async with self._acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM participants WHERE room_id = $1 AND id = $2",
                    room_id,
                    participant_id,
                )
        if row is None:
            return None
        return _row_to_participant(row)

    async def update_participant(self, participant: Participant) -> Participant:
        with self._query_span("update_participant", "participants"):
            async with self._acquire() as conn:
                await conn.execute(
                    "UPDATE participants SET channel_id=$3, display_name=$4, role=$5,"
                    " status=$6, identification=$7, identity_id=$8, external_id=$9,"
                    " resolved_at=$10, resolved_by=$11, metadata=$12"
                    " WHERE room_id=$1 AND id=$2",
                    participant.room_id,
                    participant.id,
                    participant.channel_id,
                    participant.display_name,
                    participant.role.value,
                    participant.status.value,
                    participant.identification.value,
                    participant.identity_id,
                    participant.external_id,
                    participant.resolved_at,
                    participant.resolved_by,
                    participant.metadata,
                )
        return participant

    async def list_participants(self, room_id: str) -> list[Participant]:
        with self._query_span("list_participants", "participants"):
            async with self._acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM participants WHERE room_id = $1",
                    room_id,
                )
        return [_row_to_participant(r) for r in rows]

    # ── Read tracking ────────────────────────────────────────────

    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                "SELECT index FROM events WHERE id = $1",
                event_id,
            )
            if row is not None:
                await conn.execute(
                    "INSERT INTO read_markers (room_id, channel_id, event_index)"
                    " VALUES ($1, $2, $3)"
                    " ON CONFLICT (room_id, channel_id)"
                    " DO UPDATE SET event_index = GREATEST(read_markers.event_index, $3)",
                    room_id,
                    channel_id,
                    row["index"],
                )

    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        async with self._acquire() as conn:
            await conn.execute(
                "INSERT INTO read_markers (room_id, channel_id, event_index)"
                " VALUES ($1, $2, COALESCE("
                "   (SELECT MAX(index) FROM events WHERE room_id = $1), 0))"
                " ON CONFLICT (room_id, channel_id)"
                " DO UPDATE SET event_index = COALESCE("
                "   (SELECT MAX(index) FROM events WHERE room_id = $1), 0)",
                room_id,
                channel_id,
            )

    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        async with self._acquire() as conn:
            marker = await conn.fetchrow(
                "SELECT event_index FROM read_markers WHERE room_id = $1 AND channel_id = $2",
                room_id,
                channel_id,
            )
            if marker is None:
                row = await conn.fetchrow(
                    "SELECT count(*) AS cnt FROM events WHERE room_id = $1",
                    room_id,
                )
                return row["cnt"] if row else 0

            row = await conn.fetchrow(
                "SELECT count(*) AS cnt FROM events WHERE room_id = $1 AND index > $2",
                room_id,
                marker["event_index"],
            )
            return row["cnt"] if row else 0

    # ── Identity operations ──────────────────────────────────────

    async def create_identity(self, identity: Identity) -> Identity:
        async with self._acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO identities"
                " (id, organization_id, display_name, email, channel_addresses, metadata)"
                " VALUES ($1, $2, $3, $4, $5, $6)",
                identity.id,
                identity.organization_id,
                identity.display_name,
                identity.email,
                identity.channel_addresses,
                identity.metadata,
            )
            for ch_type, addresses in identity.channel_addresses.items():
                for addr in addresses:
                    await conn.execute(
                        "INSERT INTO identity_addresses (channel_type, address, identity_id)"
                        " VALUES ($1, $2, $3)"
                        " ON CONFLICT (channel_type, address) DO UPDATE SET identity_id = $3",
                        ch_type,
                        addr,
                        identity.id,
                    )
        return identity

    async def get_identity(self, identity_id: str) -> Identity | None:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM identities WHERE id = $1",
                identity_id,
            )
        if row is None:
            return None
        ch_addr = row["channel_addresses"]
        if isinstance(ch_addr, str):
            ch_addr = json.loads(ch_addr)
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return Identity(
            id=row["id"],
            organization_id=row["organization_id"],
            display_name=row["display_name"],
            email=row["email"],
            channel_addresses=ch_addr,
            metadata=meta,
        )

    async def resolve_identity(self, channel_type: str, address: str) -> Identity | None:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                "SELECT i.* FROM identity_addresses ia"
                " JOIN identities i ON i.id = ia.identity_id"
                " WHERE ia.channel_type = $1 AND ia.address = $2",
                channel_type,
                address,
            )
        if row is None:
            return None
        ch_addr = row["channel_addresses"]
        if isinstance(ch_addr, str):
            ch_addr = json.loads(ch_addr)
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return Identity(
            id=row["id"],
            organization_id=row["organization_id"],
            display_name=row["display_name"],
            email=row["email"],
            channel_addresses=ch_addr,
            metadata=meta,
        )

    async def link_address(self, identity_id: str, channel_type: str, address: str) -> None:
        async with self._acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM identities WHERE id = $1 FOR UPDATE",
                identity_id,
            )
            if row is None:
                return
            ch_addr = row["channel_addresses"]
            if isinstance(ch_addr, str):
                ch_addr = json.loads(ch_addr)
            current = ch_addr.get(channel_type, [])
            if address not in current:
                ch_addr = {**ch_addr, channel_type: [*current, address]}
                await conn.execute(
                    "UPDATE identities SET channel_addresses = $2 WHERE id = $1",
                    identity_id,
                    ch_addr,
                )
            await conn.execute(
                "INSERT INTO identity_addresses (channel_type, address, identity_id)"
                " VALUES ($1, $2, $3)"
                " ON CONFLICT (channel_type, address) DO UPDATE SET identity_id = $3",
                channel_type,
                address,
                identity_id,
            )

    # ── Task operations ──────────────────────────────────────────

    async def add_task(self, task: Task) -> Task:
        async with self._acquire() as conn:
            await conn.execute(
                "INSERT INTO tasks"
                " (id, room_id, title, description, assigned_to, status, created_at, metadata)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                task.id,
                task.room_id,
                task.title,
                task.description,
                task.assigned_to,
                task.status.value if hasattr(task.status, "value") else task.status,
                task.created_at,
                task.metadata,
            )
        return task

    async def get_task(self, task_id: str) -> Task | None:
        async with self._acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
        if row is None:
            return None
        return _row_to_task(row)

    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        if status is not None:
            async with self._acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM tasks WHERE room_id = $1 AND status = $2",
                    room_id,
                    status,
                )
        else:
            async with self._acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM tasks WHERE room_id = $1",
                    room_id,
                )
        return [_row_to_task(r) for r in rows]

    async def update_task(self, task: Task) -> Task:
        async with self._acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET title=$2, description=$3, assigned_to=$4,"
                " status=$5, metadata=$6 WHERE id=$1",
                task.id,
                task.title,
                task.description,
                task.assigned_to,
                task.status.value if hasattr(task.status, "value") else task.status,
                task.metadata,
            )
        return task

    # ── Observation operations ───────────────────────────────────

    async def add_observation(self, observation: Observation) -> Observation:
        async with self._acquire() as conn:
            await conn.execute(
                "INSERT INTO observations"
                " (id, room_id, channel_id, content, category, confidence, created_at, metadata)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                observation.id,
                observation.room_id,
                observation.channel_id,
                observation.content,
                observation.category,
                observation.confidence,
                observation.created_at,
                observation.metadata,
            )
        return observation

    async def list_observations(self, room_id: str) -> list[Observation]:
        async with self._acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM observations WHERE room_id = $1",
                room_id,
            )
        return [_row_to_observation(r) for r in rows]
