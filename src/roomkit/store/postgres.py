"""PostgreSQL implementation of ConversationStore using asyncpg."""

from __future__ import annotations

import json
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.event import RoomEvent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task
from roomkit.store.base import ConversationStore

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS rooms (
    id TEXT PRIMARY KEY,
    organization_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    data JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    idempotency_key TEXT,
    visibility TEXT NOT NULL DEFAULT 'all',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_room_id ON events(room_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency
    ON events(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS bindings (
    channel_id TEXT NOT NULL,
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_type TEXT NOT NULL,
    participant_id TEXT,
    data JSONB NOT NULL,
    PRIMARY KEY (room_id, channel_id)
);

CREATE TABLE IF NOT EXISTS participants (
    id TEXT NOT NULL,
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    data JSONB NOT NULL,
    PRIMARY KEY (room_id, id)
);

CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS identity_addresses (
    channel_type TEXT NOT NULL,
    address TEXT NOT NULL,
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    PRIMARY KEY (channel_type, address)
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_room_id ON tasks(room_id);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_room_id ON observations(room_id);

CREATE TABLE IF NOT EXISTS read_markers (
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    PRIMARY KEY (room_id, channel_id)
);

CREATE INDEX IF NOT EXISTS idx_rooms_org ON rooms(organization_id);
CREATE INDEX IF NOT EXISTS idx_rooms_status ON rooms(status);
CREATE INDEX IF NOT EXISTS idx_rooms_metadata ON rooms USING GIN (data jsonb_path_ops);
"""


def _dump(model: Any) -> str:
    return str(model.model_dump_json())


class PostgresStore(ConversationStore):
    """PostgreSQL-backed conversation store using asyncpg."""

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

    async def init(self, min_size: int = 2, max_size: int = 10) -> None:
        """Create the connection pool (if needed) and ensure schema exists."""
        if self._pool is None:
            self._pool = await self._asyncpg.create_pool(
                self._dsn,
                min_size=min_size,
                max_size=max_size,
            )
        async with self._pool.acquire() as conn:
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
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO rooms (id, organization_id, status, created_at, data) "
                "VALUES ($1, $2, $3, $4, $5)",
                room.id,
                room.organization_id,
                room.status.value,
                room.created_at,
                _dump(room),
            )
        return room

    async def get_room(self, room_id: str) -> Room | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT data FROM rooms WHERE id = $1", room_id)
        if row is None:
            return None
        return Room.model_validate_json(row["data"])

    async def update_room(self, room: Room) -> Room:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE rooms SET organization_id = $2, status = $3, data = $4 WHERE id = $1",
                room.id,
                room.organization_id,
                room.status.value,
                _dump(room),
            )
        return room

    async def delete_room(self, room_id: str) -> bool:
        async with self._pool.acquire() as conn:
            tag = await conn.execute("DELETE FROM rooms WHERE id = $1", room_id)
        return bool(tag == "DELETE 1")

    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM rooms ORDER BY created_at LIMIT $1 OFFSET $2",
                limit,
                offset,
            )
        return [Room.model_validate_json(r["data"]) for r in rows]

    async def find_rooms(
        self,
        organization_id: str | None = None,
        status: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
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
            clauses.append(f"data->'metadata' @> ${idx}::jsonb")
            params.append(json.dumps(metadata_filter))
            idx += 1

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT data FROM rooms {where}", *params)
        return [Room.model_validate_json(r["data"]) for r in rows]

    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        params: list[Any] = [participant_id]
        idx = 2

        base = (
            "SELECT r.data FROM rooms r WHERE ("
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

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(base, *params)
        if row is None:
            return None
        return Room.model_validate_json(row["data"])

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
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, channel_id, status_val)
        else:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT room_id FROM bindings WHERE channel_id = $1 LIMIT 1",
                    channel_id,
                )
        return row["room_id"] if row else None

    # ── Event operations ─────────────────────────────────────────

    async def add_event(self, event: RoomEvent) -> RoomEvent:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO events (id, room_id, idempotency_key, visibility, created_at, data) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                event.id,
                event.room_id,
                event.idempotency_key,
                event.visibility,
                event.created_at,
                _dump(event),
            )
        return event

    async def get_event(self, event_id: str) -> RoomEvent | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT data FROM events WHERE id = $1", event_id)
        if row is None:
            return None
        return RoomEvent.model_validate_json(row["data"])

    async def update_event(self, event: RoomEvent) -> RoomEvent:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE events SET data = $2 WHERE id = $1",
                event.id,
                _dump(event),
            )
        return event

    async def list_events(
        self,
        room_id: str,
        offset: int = 0,
        limit: int = 50,
        visibility_filter: str | None = None,
    ) -> list[RoomEvent]:
        if visibility_filter is not None:
            query = (
                "SELECT data FROM events WHERE room_id = $1 AND visibility = $2 "
                "ORDER BY created_at LIMIT $3 OFFSET $4"
            )
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, room_id, visibility_filter, limit, offset)
        else:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT data FROM events WHERE room_id = $1 "
                    "ORDER BY created_at LIMIT $2 OFFSET $3",
                    room_id,
                    limit,
                    offset,
                )
        return [RoomEvent.model_validate_json(r["data"]) for r in rows]

    async def check_idempotency(self, room_id: str, key: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM events WHERE room_id = $1 AND idempotency_key = $2",
                room_id,
                key,
            )
        return row is not None

    async def get_event_count(self, room_id: str) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT count(*) AS cnt FROM events WHERE room_id = $1",
                room_id,
            )
        return row["cnt"] if row else 0

    # ── Binding operations ───────────────────────────────────────

    async def add_binding(self, binding: ChannelBinding) -> ChannelBinding:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO bindings (channel_id, room_id, channel_type, participant_id, data) "
                "VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (room_id, channel_id) DO UPDATE SET data = $5, "
                "channel_type = $3, participant_id = $4",
                binding.channel_id,
                binding.room_id,
                binding.channel_type.value,
                binding.participant_id,
                _dump(binding),
            )
        return binding

    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM bindings WHERE room_id = $1 AND channel_id = $2",
                room_id,
                channel_id,
            )
        if row is None:
            return None
        return ChannelBinding.model_validate_json(row["data"])

    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE bindings SET channel_type = $3, participant_id = $4, data = $5 "
                "WHERE room_id = $1 AND channel_id = $2",
                binding.room_id,
                binding.channel_id,
                binding.channel_type.value,
                binding.participant_id,
                _dump(binding),
            )
        return binding

    async def remove_binding(self, room_id: str, channel_id: str) -> bool:
        async with self._pool.acquire() as conn:
            tag = await conn.execute(
                "DELETE FROM bindings WHERE room_id = $1 AND channel_id = $2",
                room_id,
                channel_id,
            )
        return bool(tag == "DELETE 1")

    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM bindings WHERE room_id = $1",
                room_id,
            )
        return [ChannelBinding.model_validate_json(r["data"]) for r in rows]

    # ── Participant operations ───────────────────────────────────

    async def add_participant(self, participant: Participant) -> Participant:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO participants (id, room_id, data) VALUES ($1, $2, $3) "
                "ON CONFLICT (room_id, id) DO UPDATE SET data = $3",
                participant.id,
                participant.room_id,
                _dump(participant),
            )
        return participant

    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM participants WHERE room_id = $1 AND id = $2",
                room_id,
                participant_id,
            )
        if row is None:
            return None
        return Participant.model_validate_json(row["data"])

    async def update_participant(self, participant: Participant) -> Participant:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE participants SET data = $3 WHERE room_id = $1 AND id = $2",
                participant.room_id,
                participant.id,
                _dump(participant),
            )
        return participant

    async def list_participants(self, room_id: str) -> list[Participant]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM participants WHERE room_id = $1",
                room_id,
            )
        return [Participant.model_validate_json(r["data"]) for r in rows]

    # ── Read tracking ────────────────────────────────────────────

    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO read_markers (room_id, channel_id, event_id) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (room_id, channel_id) DO UPDATE SET event_id = $3",
                room_id,
                channel_id,
                event_id,
            )

    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM events WHERE room_id = $1 ORDER BY created_at DESC LIMIT 1",
                room_id,
            )
        if row is not None:
            await self.mark_read(room_id, channel_id, row["id"])

    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        async with self._pool.acquire() as conn:
            marker = await conn.fetchrow(
                "SELECT event_id FROM read_markers WHERE room_id = $1 AND channel_id = $2",
                room_id,
                channel_id,
            )
            if marker is None:
                row = await conn.fetchrow(
                    "SELECT count(*) AS cnt FROM events WHERE room_id = $1",
                    room_id,
                )
                return row["cnt"] if row else 0

            last_read_event = await conn.fetchrow(
                "SELECT created_at FROM events WHERE id = $1",
                marker["event_id"],
            )
            if last_read_event is None:
                row = await conn.fetchrow(
                    "SELECT count(*) AS cnt FROM events WHERE room_id = $1",
                    room_id,
                )
                return row["cnt"] if row else 0

            row = await conn.fetchrow(
                "SELECT count(*) AS cnt FROM events WHERE room_id = $1 AND created_at > $2",
                room_id,
                last_read_event["created_at"],
            )
            return row["cnt"] if row else 0

    # ── Identity operations ──────────────────────────────────────

    async def create_identity(self, identity: Identity) -> Identity:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO identities (id, data) VALUES ($1, $2)",
                identity.id,
                _dump(identity),
            )
            for ch_type, addresses in identity.channel_addresses.items():
                for addr in addresses:
                    await conn.execute(
                        "INSERT INTO identity_addresses (channel_type, address, identity_id) "
                        "VALUES ($1, $2, $3) "
                        "ON CONFLICT (channel_type, address) DO UPDATE SET identity_id = $3",
                        ch_type,
                        addr,
                        identity.id,
                    )
        return identity

    async def get_identity(self, identity_id: str) -> Identity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM identities WHERE id = $1",
                identity_id,
            )
        if row is None:
            return None
        return Identity.model_validate_json(row["data"])

    async def resolve_identity(self, channel_type: str, address: str) -> Identity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT i.data FROM identity_addresses ia "
                "JOIN identities i ON i.id = ia.identity_id "
                "WHERE ia.channel_type = $1 AND ia.address = $2",
                channel_type,
                address,
            )
        if row is None:
            return None
        return Identity.model_validate_json(row["data"])

    async def link_address(self, identity_id: str, channel_type: str, address: str) -> None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM identities WHERE id = $1",
                identity_id,
            )
            if row is None:
                return
            identity = Identity.model_validate_json(row["data"])
            current = identity.channel_addresses.get(channel_type, [])
            if address not in current:
                new_addresses = {
                    **identity.channel_addresses,
                    channel_type: [*current, address],
                }
                updated = identity.model_copy(update={"channel_addresses": new_addresses})
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE identities SET data = $2 WHERE id = $1",
                        identity_id,
                        _dump(updated),
                    )
                    await conn.execute(
                        "INSERT INTO identity_addresses (channel_type, address, identity_id) "
                        "VALUES ($1, $2, $3) "
                        "ON CONFLICT (channel_type, address) DO UPDATE SET identity_id = $3",
                        channel_type,
                        address,
                        identity_id,
                    )
            else:
                await conn.execute(
                    "INSERT INTO identity_addresses (channel_type, address, identity_id) "
                    "VALUES ($1, $2, $3) "
                    "ON CONFLICT (channel_type, address) DO UPDATE SET identity_id = $3",
                    channel_type,
                    address,
                    identity_id,
                )

    # ── Task operations ──────────────────────────────────────────

    async def add_task(self, task: Task) -> Task:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tasks (id, room_id, status, data) VALUES ($1, $2, $3, $4)",
                task.id,
                task.room_id,
                task.status,
                _dump(task),
            )
        return task

    async def get_task(self, task_id: str) -> Task | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT data FROM tasks WHERE id = $1", task_id)
        if row is None:
            return None
        return Task.model_validate_json(row["data"])

    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        if status is not None:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT data FROM tasks WHERE room_id = $1 AND status = $2",
                    room_id,
                    status,
                )
        else:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT data FROM tasks WHERE room_id = $1",
                    room_id,
                )
        return [Task.model_validate_json(r["data"]) for r in rows]

    async def update_task(self, task: Task) -> Task:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET status = $2, data = $3 WHERE id = $1",
                task.id,
                task.status,
                _dump(task),
            )
        return task

    # ── Observation operations ───────────────────────────────────

    async def add_observation(self, observation: Observation) -> Observation:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO observations (id, room_id, data) VALUES ($1, $2, $3)",
                observation.id,
                observation.room_id,
                _dump(observation),
            )
        return observation

    async def list_observations(self, room_id: str) -> list[Observation]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM observations WHERE room_id = $1",
                room_id,
            )
        return [Observation.model_validate_json(r["data"]) for r in rows]
