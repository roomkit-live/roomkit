"""PostgreSQL DDL schema for the relational ConversationStore.

Owns the DDL used by ``PostgresStore``. Two concerns are kept strictly
separate:

* ``SCHEMA`` — **additive, idempotent** DDL (``CREATE TABLE IF NOT EXISTS``
  and index touch-ups). Safe to run on every connect; never drops a table.
  This is the only DDL ``PostgresStore.init()`` executes.
* ``V1_TO_V2_DROP`` — the **destructive** v1→v2 migration. It drops every
  table (data loss) and is *never* run automatically. It is applied only by
  the explicit, opt-in ``PostgresStore.migrate()`` after the caller confirms
  a backup.

Keeping the destructive DDL out of the connect path means a routine
``init()`` — e.g. after a library upgrade — can never delete a user's data.
"""

from __future__ import annotations

SCHEMA_VERSION = 2

# Tables owned by RoomKit, in dependency order (children before parents) so a
# CASCADE-free reading of the list still makes sense. Used to build the
# destructive migration DDL and to report what a migration would drop.
V1_TABLES = [
    "read_markers",
    "observations",
    "tasks",
    "identity_addresses",
    "identities",
    "participants",
    "bindings",
    "events",
    "rooms",
    "schema_version",
]

# Detects the legacy v1 (JSONB-blob) schema by the presence of the ``data``
# column on the ``rooms`` table. Returns a single boolean.
V1_DETECT = """\
SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'rooms' AND column_name = 'data'
)
"""

# DESTRUCTIVE: drops all v1 tables. Applied only by PostgresStore.migrate()
# with explicit confirmation — never by init(). ``schema_version`` is dropped
# last so a partial failure still reads as "not yet migrated".
V1_TO_V2_DROP = "\n".join(f"DROP TABLE IF EXISTS {t} CASCADE;" for t in V1_TABLES)

# Additive, idempotent schema. Run on every connect. Contains no DROP TABLE.
SCHEMA = """\
-- Migration: idx_participants_channel must NOT be unique.
-- Multiple participants can share the same channel in group rooms.
-- Dropping/recreating an INDEX is non-destructive to data.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_participants_channel'
        AND indexdef LIKE '%UNIQUE%'
    ) THEN
        DROP INDEX idx_participants_channel;
        CREATE INDEX idx_participants_channel ON participants(room_id, channel_id);
        RAISE NOTICE 'Converted idx_participants_channel from UNIQUE to regular index';
    END IF;
END $$;

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
CREATE INDEX IF NOT EXISTS idx_events_parent
    ON events(parent_event_id) WHERE parent_event_id IS NOT NULL;
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
CREATE INDEX IF NOT EXISTS idx_participants_channel
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
