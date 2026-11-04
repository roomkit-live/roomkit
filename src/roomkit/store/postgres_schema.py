"""PostgreSQL DDL schema for the relational ConversationStore.

Owns the DDL used by ``PostgresStore``. Two concerns are kept strictly
separate:

* ``SCHEMA`` — **additive, idempotent** schema setup (``CREATE TABLE IF NOT
  EXISTS``, index touch-ups, bounded repairs for newly added columns, and
  one-time rewrites a rule change needs, each recorded in ``schema_version``
  so it runs once). Safe to run on every connect; never drops a table or
  discards user data. This is the only schema batch ``PostgresStore.init()``
  executes.
* ``V1_TO_V2_DROP`` — the **destructive** v1→v2 migration. It drops every
  table (data loss) and is *never* run automatically. It is applied only by
  the explicit, opt-in ``PostgresStore.migrate()`` after the caller confirms
  a backup.

Keeping the destructive DDL out of the connect path means a routine
``init()`` — e.g. after a library upgrade — can never delete a user's data.
"""

from __future__ import annotations

SCHEMA_VERSION = 2
"""The table layout's generation, which ``migrate()`` reports. The rows of
``schema_version`` above 2 record one-time data rewrites, not a new layout."""

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
    "event_sequences",
    "events",
    "rooms",
    "schema_version",
]

# Detects the v1 (JSONB-blob) schema by the presence of the ``data``
# column on the ``rooms`` table of the schema this session writes to: a
# namesake in another schema is another deployment's. Returns a single
# boolean.
V1_DETECT = """\
SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'rooms' AND column_name = 'data'
)
"""

# DESTRUCTIVE: drops all v1 tables. Applied only by PostgresStore.migrate()
# with explicit confirmation — never by init(). ``schema_version`` is dropped
# last so a partial failure still reads as "not yet migrated". Each table is
# the one of the schema this session writes to: through search_path, a table
# the v1 schema lacks would be another deployment's.
V1_TO_V2_DROP = "\n".join(
    "DO $$ BEGIN EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE', current_schema(),"
    f" '{t}'); END $$;"
    for t in V1_TABLES
)


def _column_check(table: str, name: str, definition: str) -> str:
    """The check adding column *name* to *table* when the table of the
    schema this session writes to lacks it."""
    return (
        "    IF NOT EXISTS (SELECT 1 FROM information_schema.columns"  # nosec B608  # this module's names
        f" WHERE table_schema = current_schema() AND table_name = '{table}'"
        f" AND column_name = '{name}') THEN\n"
        f"        ALTER TABLE {table} ADD COLUMN {name} {definition};\n"
        "    END IF;\n"
    )


def _add_missing_columns(table: str, columns: dict[str, str]) -> str:
    """A block adding each of *columns* (name: definition) to *table*, only
    when the table of the schema this session writes to lacks it. ``ALTER
    TABLE`` takes ACCESS EXCLUSIVE on the table even when ``IF NOT EXISTS``
    finds the column, which would queue every boot behind any open
    transaction on the table, and all the traffic behind it.
    ``information_schema.columns`` spans every schema the role can see, so
    the lookup names ``current_schema()``."""
    checks = (_column_check(table, name, definition) for name, definition in columns.items())
    return "".join(("DO $$\nBEGIN\n", *checks, "END $$;\n"))


def _index_check(table: str, name: str, definition: str, unique: bool) -> str:
    """The check creating index *name* on *table* when the schema this
    session writes to lacks it."""
    kind = "UNIQUE INDEX" if unique else "INDEX"
    return (
        f"    IF to_regclass(format('%I.%I', current_schema(), '{name}')) IS NULL THEN\n"
        f"        CREATE {kind} {name} ON {table} {definition};\n"
        "    END IF;\n"
    )


def _create_missing_indexes(table: str, indexes: dict[str, str], *, unique: bool = False) -> str:
    """A block creating each of *indexes* (name: columns and predicate) on
    *table*, only when the schema this session writes to lacks it. ``CREATE
    INDEX`` takes SHARE on the table even when ``IF NOT EXISTS`` finds the
    index, which waits for every open writer of the table, and every writer
    then waits behind it."""
    checks = (_index_check(table, name, cols, unique) for name, cols in indexes.items())
    return "".join(("DO $$\nBEGIN\n", *checks, "END $$;\n"))


# Additive, idempotent schema. Run on every connect. Contains no DROP TABLE.
SCHEMA = "".join(
    (
        """\
-- Every name this batch looks up in the catalogs, or drops, is the one of the
-- schema this session writes to (current_schema()): two deployments may share
-- a database, one per schema, and through search_path a namesake of the
-- other's would be read for ours, or dropped.
--
-- Migration: idx_participants_channel must NOT be unique.
-- Multiple participants can share the same channel in group rooms.
-- Dropping/recreating an INDEX is non-destructive to data.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
        AND indexname = 'idx_participants_channel'
        AND indexdef LIKE '%UNIQUE%'
    ) THEN
        EXECUTE format('DROP INDEX %I.%I', current_schema(), 'idx_participants_channel');
        EXECUTE format(
            'CREATE INDEX %I ON %I.%I (room_id, channel_id)',
            'idx_participants_channel', current_schema(), 'participants'
        );
        RAISE NOTICE 'Converted idx_participants_channel from UNIQUE to regular index';
    END IF;
END $$;

-- Migration: upgrade idx_events_room_index to UNIQUE(room_id, index) so a
-- duplicate index (e.g. from two processes assigning concurrently) is rejected
-- rather than silently persisted (RFC §8.1 / §14.3). Best-effort in init(): if
-- the table already holds duplicate (room_id, index) rows (from a prior
-- release's races), the upgrade is skipped with a warning so startup still
-- succeeds — deduplicate, then recreate the index UNIQUE. The inner block's
-- rollback on failure preserves the existing non-unique index.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
        AND indexname = 'idx_events_room_index'
        AND indexdef NOT LIKE '%UNIQUE%'
    ) THEN
        BEGIN
            EXECUTE format('DROP INDEX %I.%I', current_schema(), 'idx_events_room_index');
            EXECUTE format(
                'CREATE UNIQUE INDEX %I ON %I.%I (room_id, index)',
                'idx_events_room_index', current_schema(), 'events'
            );
            RAISE NOTICE 'Upgraded idx_events_room_index to UNIQUE(room_id, index)';
        EXCEPTION WHEN unique_violation THEN
            RAISE WARNING 'idx_events_room_index NOT upgraded to UNIQUE: duplicate '
                '(room_id, index) rows exist. Deduplicate the events table, then '
                'recreate the index UNIQUE. Multi-process index safety is NOT '
                'enforced until then.';
        END;
    END IF;
END $$;

-- rooms
CREATE TABLE IF NOT EXISTS rooms (
    id              TEXT PRIMARY KEY,
    organization_id TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    event_count     INTEGER NOT NULL DEFAULT 0,
    latest_index    INTEGER,
    delivered_index INTEGER NOT NULL DEFAULT -1,
    metadata        JSONB NOT NULL DEFAULT '{}',
    timers          JSONB NOT NULL DEFAULT '{}',
    -- What an agent's own output solicits here (RFC §19.3.1).
    agent_response_policy TEXT NOT NULL DEFAULT 'agent_chain',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- When the room was paused (RFC §5.1); NULL unless paused.
    paused_at       TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ
);

-- Additive migration for pre-existing deployments: the delivery-lane
-- cursor (RFC §10.1 step 14). -1 = no delivery set executed yet. The
-- backfill runs ONLY when the column is first created: a pre-lane
-- deployment delivered under the room lock, so every stored event is
-- delivered and the cursor starts at latest_index — without it, a busy
-- room's first laned delivery would wait out a delivery_gap_timeout
-- against a purely historical hole. Guarded (not re-run per boot):
-- re-applying it against a live multi-worker deployment would declare
-- an in-flight first delivery lost. Empty rooms keep -1.
--
-- The lookup is scoped to the schema this session actually writes to:
-- information_schema.columns spans every schema the role can see, so an
-- unqualified match would read a namesake `rooms` table from another
-- schema and skip the migration our tables need. The guard is safe against
-- a concurrent boot because PostgresStore.init() runs this DDL under the
-- migration advisory lock.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'rooms'
          AND column_name = 'delivered_index'
    ) THEN
        ALTER TABLE rooms ADD COLUMN delivered_index INTEGER NOT NULL DEFAULT -1;
        UPDATE rooms SET delivered_index = latest_index WHERE event_count > 0;
    END IF;
END $$;
-- latest_index is NULL while a room has no event (RFC §5.1): an empty room
-- stored 0, which reads as the first event's index. Guarded on the column's
-- nullability, so it runs once and no later boot scans the table.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'rooms'
          AND column_name = 'latest_index'
          AND is_nullable = 'NO'
    ) THEN
        ALTER TABLE rooms ALTER COLUMN latest_index DROP NOT NULL;
        ALTER TABLE rooms ALTER COLUMN latest_index DROP DEFAULT;
        UPDATE rooms SET latest_index = NULL WHERE event_count = 0;
    END IF;
END $$;
-- The instant a room paused (RFC §5.1), which its close timer counts from.
-- Added only when missing, as for delivered_index: a room stored paused
-- before it takes its last modification, which the pause was, once; no later
-- boot scans the table.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'rooms'
          AND column_name = 'paused_at'
    ) THEN
        ALTER TABLE rooms ADD COLUMN paused_at TIMESTAMPTZ;
        UPDATE rooms SET paused_at = updated_at WHERE status = 'paused';
    END IF;
END $$;
-- Additive: the default IS the behaviour every existing room ran under, so
-- there is nothing to backfill.
""",
        _add_missing_columns(
            "rooms", {"agent_response_policy": "TEXT NOT NULL DEFAULT 'agent_chain'"}
        ),
        _create_missing_indexes(
            "rooms",
            {
                "idx_rooms_org": "(organization_id)",
                "idx_rooms_status": "(status)",
                "idx_rooms_updated": "(updated_at DESC)",
            },
        ),
        """\

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
    -- Intelligence channels asked to act (RFC §19.3). NULL = unaddressed,
    -- which is not the same as the empty array (addressed to nobody), so the
    -- column is nullable rather than defaulting to '{}'.
    addressed_to        TEXT[],
    response_visibility TEXT,
    index               INTEGER NOT NULL DEFAULT 0,
    chain_depth         INTEGER NOT NULL DEFAULT 0,
    correlation_id      TEXT,
    parent_event_id     TEXT,
    -- The event this one answers (RFC §8.5); NULL for an event that answers none.
    responds_to         TEXT,
    idempotency_key     TEXT,
    blocked_by          TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    channel_data        JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- The outcome of the event's delivery set, written once it has run.
    delivery_results    JSONB NOT NULL DEFAULT '{}'
);
-- Columns that arrived after events existed, each additive, with nothing to
-- backfill: addressing (an existing row is unaddressed, NULL, the behaviour it
-- was stored under), the event an answer answers (RFC §8.5: no event stored
-- before it named one), a delivery set's outcome (RFC §10.3, §14.3: an
-- existing row recorded none, '{}').
""",
        _add_missing_columns(
            "events",
            {
                "addressed_to": "TEXT[]",
                "responds_to": "TEXT",
                "delivery_results": "JSONB NOT NULL DEFAULT '{}'",
            },
        ),
        _create_missing_indexes(
            "events", {"idx_events_room_index": "(room_id, index)"}, unique=True
        ),
        """\
-- Composite idx_events_parent_index on (parent_event_id, index): thread-reply
-- pagination filters by parent_event_id then reads forward ORDER BY index, so
-- the second column lets Postgres return the page pre-sorted instead of
-- sorting the whole thread. The leading column still serves plain
-- parent_event_id lookups. A distinct name (not the older single-column
-- idx_events_parent) so init() creates it on databases that predate the
-- composite; PostgresStore.drop_legacy_parent_index() removes the old one.
-- idx_events_room_responds_to finds a request's answers: the events of a
-- room that answer a given event.
""",
        _create_missing_indexes(
            "events",
            {
                "idx_events_room_created": "(room_id, created_at)",
                "idx_events_room_type": "(room_id, type)",
                "idx_events_correlation": "(correlation_id) WHERE correlation_id IS NOT NULL",
                "idx_events_parent_index": (
                    "(parent_event_id, index) WHERE parent_event_id IS NOT NULL"
                ),
                "idx_events_source": "(source_channel_id)",
                "idx_events_room_responds_to": (
                    "(room_id, responds_to) WHERE responds_to IS NOT NULL"
                ),
            },
        ),
        """\
-- An idempotency key is unique per room and per source channel (RFC §13.4):
-- the per-room index gives way to one that names the channel. Keys unique
-- per room are unique per room and channel, so the new index always builds.
-- The index names below are looked up in the schema this session writes to.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema() AND indexname = 'idx_events_idempotency'
    ) THEN
        EXECUTE format('DROP INDEX %I.%I', current_schema(), 'idx_events_idempotency');
    END IF;
END $$;
""",
        _create_missing_indexes(
            "events",
            {
                "idx_events_idempotency_channel": (
                    "(room_id, source_channel_id, idempotency_key)"
                    " WHERE idempotency_key IS NOT NULL"
                )
            },
            unique=True,
        ),
        """\
-- A proactive delivery's key is looked up within its room, whatever channel
-- committed it (RFC §22.1).
""",
        _create_missing_indexes(
            "events",
            {
                "idx_events_room_idempotency": (
                    "(room_id, idempotency_key) WHERE idempotency_key IS NOT NULL"
                )
            },
        ),
        """\

-- Per-room high-water marks make indexes monotonic even when the latest
-- event is deleted. Every write keeps them since the table exists (an insert
-- observes its index, a commit reserves the next), so the backfill from the
-- rooms and events of a deployment that predates it runs in the boot that
-- creates it, and only then: run at every boot it wrote every row, waiting on
-- any commit holding one, and every commit then waited behind it.
DO $$
BEGIN
    IF to_regclass(format('%I.%I', current_schema(), 'event_sequences')) IS NULL THEN
        CREATE TABLE event_sequences (
            room_id    TEXT PRIMARY KEY REFERENCES rooms(id) ON DELETE CASCADE,
            next_index INTEGER NOT NULL DEFAULT 0 CHECK (next_index >= 0)
        );
        INSERT INTO event_sequences(room_id, next_index)
        SELECT r.id,
               GREATEST(
                   CASE WHEN r.event_count > 0 THEN r.latest_index + 1 ELSE 0 END,
                   COALESCE(MAX(e.index) + 1, 0)
               )
        FROM rooms r
        LEFT JOIN events e ON e.room_id = r.id
        GROUP BY r.id, r.event_count, r.latest_index;
    END IF;
END $$;

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

-- Nullable policies preserve defaults for existing deployments, and retain
-- existing binding rows. A group binding admits every sender (RFC §10.4);
-- existing rows are not one. Not named "group": a reserved word in SQL.
""",
        _add_missing_columns(
            "bindings",
            {
                "rate_limit": "JSONB",
                "retry_policy": "JSONB",
                "is_group": "BOOLEAN NOT NULL DEFAULT FALSE",
            },
        ),
        """\
-- Routing reads the binding of one channel naming one sender (RFC §10.4): on
-- a number shared by many correspondents every binding has the same channel.
""",
        _create_missing_indexes(
            "bindings",
            {
                "idx_bindings_channel_id": "(channel_id)",
                "idx_bindings_channel_participant": "(channel_id, participant_id)",
            },
        ),
        """\

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
-- The channels a participant has been reached through (RFC 5.5), primary
-- included and first. The UPDATE is an idempotent data repair for rows created
-- before the column existed, and for any old writer that stored an empty or
-- out-of-order list. It never discards a channel already recorded.
""",
        _add_missing_columns(
            "participants",
            {
                "connected_via": "TEXT[] NOT NULL DEFAULT '{}'",
                # The identities a resolver found an ambiguous sender could be
                # (RFC §11.2). NULL is not the empty list: none were offered.
                "candidates": "TEXT[]",
            },
        ),
        """\
UPDATE participants
SET connected_via = ARRAY[channel_id] || array_remove(connected_via, channel_id)
WHERE connected_via[1] IS DISTINCT FROM channel_id;
-- Routing reads the rooms a sender is a participant of, and names a
-- participant by its identity too (RFC §10.4 step 1).
""",
        _create_missing_indexes(
            "participants",
            {
                "idx_participants_channel": "(room_id, channel_id)",
                "idx_participants_id": "(id)",
                "idx_participants_identity": "(identity_id)",
            },
        ),
        """\

-- identities
CREATE TABLE IF NOT EXISTS identities (
    id                TEXT PRIMARY KEY,
    organization_id   TEXT,
    display_name      TEXT,
    email             TEXT,
    channel_addresses JSONB NOT NULL DEFAULT '{}',
    metadata          JSONB NOT NULL DEFAULT '{}'
);

-- What an identity holds beside its addresses (RFC §5.6). Additive; an existing
-- row was stored without them, and reads as having none.
""",
        _add_missing_columns(
            "identities",
            {
                "phone": "TEXT",
                "external_id": "TEXT",
                "external_ids": "JSONB NOT NULL DEFAULT '{}'",
            },
        ),
        """\

CREATE TABLE IF NOT EXISTS identity_addresses (
    channel_type    TEXT NOT NULL,
    address         TEXT NOT NULL,
    identity_id     TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    organization_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (channel_type, address, organization_id)
);

-- An address is unique WITHIN an organization, not globally (RFC §17.2): the
-- same number belongs to one person at one tenant and someone else at another.
-- Existing rows carry the empty string, the unscoped tenant, which preserves
-- what they resolved to before. The key is rebuilt only when it is still the
-- two-column one, so init() stays idempotent.
""",
        _add_missing_columns(
            "identity_addresses", {"organization_id": "TEXT NOT NULL DEFAULT ''"}
        ),
        """\

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_index i
        WHERE i.indrelid = to_regclass(format('%I.%I', current_schema(), 'identity_addresses'))
          AND i.indisprimary
          AND array_length(i.indkey::int[], 1) = 2
    ) THEN
        ALTER TABLE identity_addresses DROP CONSTRAINT identity_addresses_pkey;
        ALTER TABLE identity_addresses
            ADD PRIMARY KEY (channel_type, address, organization_id);
    END IF;
END $$;

-- Identity verification (RFC §11.7). Additive: new tables only. The record
-- itself is a JSONB document; the columns are what lookups filter on.
CREATE TABLE IF NOT EXISTS verifications (
    identity_id     TEXT NOT NULL,
    organization_id TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL,
    data            JSONB NOT NULL,
    PRIMARY KEY (identity_id, organization_id)
);

CREATE TABLE IF NOT EXISTS verification_requests (
    id              TEXT PRIMARY KEY,
    identity_id     TEXT NOT NULL,
    organization_id TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL,
    data            JSONB NOT NULL
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
""",
        _create_missing_indexes("tasks", {"idx_tasks_room_id": "(room_id)"}),
        """\

-- observations
CREATE TABLE IF NOT EXISTS observations (
    id          TEXT PRIMARY KEY,
    room_id     TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_id  TEXT,
    content     TEXT NOT NULL,
    category    TEXT,
    confidence  DOUBLE PRECISION,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB NOT NULL DEFAULT '{}'
);
-- A confidence reads as written: a REAL read 0.42 back as 0.41999998688697815.
-- Through its text, the shortest digits that name the single, so that a value
-- stored before reads as written too. Guarded on the column's type, so it runs
-- once and no later boot rewrites the table.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'observations'
          AND column_name = 'confidence'
          AND data_type = 'real'
    ) THEN
        ALTER TABLE observations ALTER COLUMN confidence TYPE DOUBLE PRECISION
            USING confidence::text::double precision;
    END IF;
END $$;
""",
        _create_missing_indexes("observations", {"idx_observations_room_id": "(room_id)"}),
        """\

-- read tracking
CREATE TABLE IF NOT EXISTS read_markers (
    room_id    TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    channel_id TEXT NOT NULL,
    event_index INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (room_id, channel_id)
);

-- Proactive voice injection reservations, independent of queue acknowledgement.
CREATE TABLE IF NOT EXISTS voice_deliveries (
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    outcome JSONB,
    PRIMARY KEY (room_id, key_hash)
);

-- A room's private state (RFC §14.1), apart from its metadata, which clients
-- may read: the keys the runtime keeps for itself, the register of authors
-- among them (§10.1 step 12), whose digests name the room's senders to
-- whoever tries every phone number or name.
CREATE TABLE IF NOT EXISTS private_state (
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    key     TEXT NOT NULL,
    value   JSONB NOT NULL,
    PRIMARY KEY (room_id, key)
);

-- schema version
CREATE TABLE IF NOT EXISTS schema_version (
    version INT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO schema_version (version)
    SELECT 2 WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version >= 2);

-- An event's visibility is stamped at its commit (RFC §7.3): its own
-- intersected with its source binding's. An event stored 'all' before read its
-- source binding's when history was rebuilt; it takes that binding's now,
-- once, so the stored field is the whole answer and its readers stay the
-- same. A message held for verification records the visibility its release
-- restores; a recorded 'all' takes the binding's too. Version 3 records the
-- rewrite as done.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_version WHERE version >= 3) THEN
        UPDATE events e SET visibility = b.visibility
        FROM bindings b
        WHERE b.room_id = e.room_id
          AND b.channel_id = e.source_channel_id
          AND e.visibility = 'all'
          AND COALESCE(b.visibility, 'all') <> 'all';
        UPDATE events e SET metadata = jsonb_set(
            e.metadata, '{roomkit:verification_held,visibility}', to_jsonb(b.visibility))
        FROM bindings b
        WHERE b.room_id = e.room_id
          AND b.channel_id = e.source_channel_id
          AND e.metadata -> 'roomkit:verification_held' ->> 'visibility' = 'all'
          AND COALESCE(b.visibility, 'all') <> 'all';
        INSERT INTO schema_version (version) VALUES (3);
    END IF;
END $$;

-- A room's register of authors leaves its metadata for its private state, once;
-- one that is not an object is dropped, as its readers read it absent. The
-- metadata loses only the copy the table now holds: one a process from
-- before writes meanwhile, or wrote since a move, stays to be merged at the
-- room's next turn that names its sender (a register only adds), never
-- dropped with the ranks it gave. Version 4 records the move as done.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_version WHERE version >= 4) THEN
        INSERT INTO private_state (room_id, key, value)
            SELECT id, 'author_register', metadata -> 'author_register' FROM rooms
            WHERE jsonb_typeof(metadata -> 'author_register') = 'object'
            ON CONFLICT (room_id, key) DO NOTHING;
        UPDATE rooms r SET metadata = r.metadata - 'author_register'
            FROM private_state p
            WHERE p.room_id = r.id AND p.key = 'author_register'
              AND r.metadata -> 'author_register' = p.value;
        UPDATE rooms SET metadata = metadata - 'author_register'
            WHERE jsonb_typeof(metadata -> 'author_register') <> 'object';
        INSERT INTO schema_version (version) VALUES (4);
    END IF;
END $$;

-- An address the move to per-organization addresses left unscoped ('') for an
-- identity of an organization moves to that organization, once: a read serves an
-- identity in its own organization only (RFC §17.2), and its sender stays
-- identified there. One whose organization holds the address already yields to
-- that row, whose identity keeps it. A row of another organization is left: no API
-- writes one, and no read serves it. Version 5 records the rewrite as done.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_version WHERE version >= 5) THEN
        DELETE FROM identity_addresses ia USING identities i, identity_addresses held
            WHERE i.id = ia.identity_id
              AND ia.organization_id = ''
              AND COALESCE(i.organization_id, '') <> ''
              AND held.channel_type = ia.channel_type
              AND held.address = ia.address
              AND held.organization_id = i.organization_id;
        UPDATE identity_addresses ia SET organization_id = i.organization_id
            FROM identities i
            WHERE i.id = ia.identity_id
              AND ia.organization_id = ''
              AND COALESCE(i.organization_id, '') <> '';
        INSERT INTO schema_version (version) VALUES (5);
    END IF;
END $$;
""",
    )
)
