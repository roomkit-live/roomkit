# Storage

RoomKit uses the `ConversationStore` ABC for persistence. Choose
`InMemoryStore` for ephemeral development state, `SQLiteStore` for an embedded
single-process deployment, or `PostgresStore` when several processes share the
same conversations.

## InMemoryStore (Default)

```python
from roomkit import RoomKit

kit = RoomKit()  # Uses InMemoryStore automatically
```

Data lives in Python dicts — fast for development, lost on restart.

What a store returns is its snapshot (RFC §14.4): a room, a binding, a
participant or an event read from it, the room, bindings and participants of a
hook's or a channel's `RoomContext` included, is treated as frozen, and changed by a copy
and a write. `InMemoryStore` copies a record once as it is written and shares
the stored object with every reader, so changing a read object in place
changes the store behind the write interface's back (a scratch copy is
`model_copy(deep=True)`: a plain `model_copy()` shares its dicts); SQL stores
return a new object per read, and the same code holds on all of them:

```python
from roomkit import RoomStatus

room = await kit.store.get_room("persistent")
await kit.store.update_room(room.model_copy(update={"status": RoomStatus.PAUSED}))
```

## SQLiteStore

`SQLiteStore` is included in the core package and uses only Python's stdlib
`sqlite3`. It is suited to desktop applications, edge deployments, and small
bots running in one process.

```python
from roomkit import RoomKit, SQLiteStore

kit = RoomKit(store=SQLiteStore("roomkit.db"))
```

RoomKit closes the database when `await kit.close()` runs. SQLite history is
stored as Pydantic JSON plus indexed query columns, and message text can be
searched with `await kit.store.search_events("invoice", room_id="support")`.

Do not share one SQLite file between RoomKit worker processes: SQLite
serializes database writes, but it cannot serialize the complete inbound
pipeline and its idempotency decisions. Use PostgreSQL with a distributed lock
manager for horizontal scaling.

## PostgresStore

Install: `pip install roomkit[postgres]`

```python
from roomkit import RoomKit
from roomkit.store import PostgresStore

store = PostgresStore("postgresql://user:pass@localhost/roomkit")
await store.init()  # Creates connection pool and tables

kit = RoomKit(store=store)
```

### Connection Pooling

```python
store = PostgresStore("postgresql://user:pass@localhost/roomkit")
await store.init(min_size=5, max_size=20)  # Pool sizing via init()
```

### Schema

PostgresStore creates these tables, among others:

| Table | Purpose |
|-------|---------|
| `rooms` | Room records with status, metadata, timers |
| `private_state` | What each room keeps for the runtime, apart from its metadata (the register of authors) |
| `events` | Event timeline with sequential indexing |
| `event_sequences` | Per-room monotonic event-index high-water marks |
| `participants` | Room participants with roles and status |
| `bindings` | Channel-to-room bindings with config |
| `identities` | Known identity records |
| `tasks` | AI-extracted tasks |
| `observations` | AI-extracted observations |
| `identity_addresses` | Multi-channel identity address lookup |
| `read_markers` | Per-channel read positions |
| `schema_version` | Schema migration tracking |

The tables live in the session's schema, `current_schema()` (the first schema
of `search_path` that exists). Two deployments may share a database, one
schema each: `init()`, `migrate()`, `dedupe_event_indices()` and
`drop_legacy_parent_index()` read, create and drop the tables, columns and
indexes of their own schema only, never a namesake another schema holds.
`init()` adds a column, or an index, only when its schema lacks it, and
seeds the per-room index marks (`event_sequences`) only in the boot that
creates that table. A boot on an up-to-date schema runs no `ALTER TABLE`, no
`CREATE INDEX` and no write of a row: it waits on no open reader or writer of
the tables, and none waits on it.

### Operations

```python
# Room operations
room = await kit.create_room(room_id="persistent", metadata={"topic": "billing"})

# Event storage and retrieval
events = await kit.store.list_events("persistent", offset=0, limit=50)

# Timeline query with filters
timeline = await kit.get_timeline("persistent", offset=0, limit=50)

# Participant management
participants = await kit.store.list_participants("persistent")

# Binding management
bindings = await kit.store.list_bindings("persistent")
```

### Full Example

```python
from __future__ import annotations

import asyncio
import os

from roomkit import InboundMessage, RoomKit, TextContent, WebSocketChannel
from roomkit.store import PostgresStore


async def main() -> None:
    store = PostgresStore(os.environ["DATABASE_URL"])
    await store.init()

    kit = RoomKit(store=store)

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)

    room = await kit.create_room(room_id="support", metadata={"topic": "billing"})
    await kit.attach_channel("support", "ws-user")

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="I need help with my invoice"),
        )
    )

    # Data persists across restarts
    events = await kit.store.list_events("support")
    print(f"Stored {len(events)} events")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Custom Store

Implement `ConversationStore` for other backends (Redis, DynamoDB, etc.):

```python
from roomkit import ConversationStore, Room, RoomAlreadyExistsError

class RedisStore(ConversationStore):
    async def create_room(self, room: Room) -> Room:
        # A room is created once (RFC §14.3): never replace the one an id names.
        if not await self.redis.set(f"room:{room.id}", room.model_dump_json(), nx=True):
            raise RoomAlreadyExistsError(f"Room {room.id} already exists")
        return room

    # Implement all abstract methods...

kit = RoomKit(store=RedisStore())
```

What every store keeps alike (RFC §10.3, §14.1, §14.3):

- `create_room()` refuses an id it holds with `RoomAlreadyExistsError`, nothing
  written: replacing the room would re-home its conversation, another
  organization's included.
- `update_event()` of an event the store does not hold raises
  `EventNotFoundError`, and `add_participant()` into a room it does not hold
  `RoomNotFoundError`, nothing written. `add_participant()` under an id the
  room has raises `ParticipantExistsError` and keeps the record:
  `update_participant()` rewrites one, and raises `ParticipantNotFoundError`
  for a record the room does not hold.
- `get_private_state(room_id, key)` and `set_private_state(room_id, key,
  value)` keep what a room holds for the runtime alone (RFC §14.1), apart
  from its metadata, which clients may read: a JSON object per key. The room's
  register of authors (RFC §6.4, §10.1 step 12) is the key `author_register`:
  its digests hash the room's id with each sender's id and name, and trying
  every phone number or name finds the senders. `get` returns the caller's
  own copy, `None` for an unknown room or key; `set` writes nothing for a
  room the store lacks, one deleted while it writes included, removes the
  key when `value` is `None`, and removes
  the copy the metadata may still hold under the key, in one write; the
  room's deletion takes its private state. The keys are the
  implementation's, and each reserves the same key in the room's metadata:
  the base class keeps the value there, and `set` removes the copy there, so
  only the runtime writes one there: the kit's `create_room()` and
  `update_room_metadata()` drop `author_register` or `_speak_queue_results`
  with a warning, as any `roomkit:` key, and write the rest. The store's own
  `update_room()` and `patch_room_metadata()` take any key: they sit below
  the kit's API, where migrations and the runtime write these keys. A register the
  metadata holds (a room stored before, or a process from an older RoomKit
  writing there) is merged into the store's at the room's next turn that
  names its sender (a BLOCKED turn does not): a register only adds, so the
  merge gives none of its ranks again. The base class keeps each key in the
  room's metadata, as before, and warns once: implement both to keep it
  private.
- `commit_event(room_id, event, *, target=None, activity=True)` stores the
  event and bumps the room's counters in one transaction; a store implements
  it, and `advance_delivered_index()`, the delivery cursor's conditional
  write, itself: the base class has no default for either. An activity commit
  (an event the pipeline stores DELIVERED) also moves `last_activity_at` and
  resumes a PAUSED room, in the same transaction (RFC §5.1); the framework
  passes `activity=False` for a lifecycle record, a BLOCKED record and a
  `commit_event()` record, which leave both. For a DELIVERED edit or delete the
  framework passes `target`, the target as the change writes it, which the same
  transaction writes; a target the room no longer holds fails the whole commit
  with `EventNotFoundError`. A delete's commit also marks every edit of its
  target `metadata.deleted`, in the same transaction: an edit holds a version
  of the deleted text. A host that soft-deletes a message itself, writing
  `metadata.deleted` with `kit.update_event()`, marks no edit: an edit the
  history holds without its message stays readable. Delete through the
  pipeline (a DELETE event, `send_event()`) or `kit.delete_event()`, or mark
  the message's edits too. An event whose idempotency key its source channel
  holds in the room for another event raises `IdempotencyKeyTakenError`,
  nothing written, its target included (RFC §13.4); the framework checks the
  key under the room lock first, so only a direct caller meets it. An event
  under an id the store holds is the database's own error
  (`asyncpg.UniqueViolationError`, `sqlite3.IntegrityError`).
- `update_room_if(room, *, status, paused_at, last_activity_at)` writes the
  room as `update_room()` does only if the stored room still holds that
  status, `paused_at` and `last_activity_at`, else writes nothing and returns
  `None`: a timer transition's compare-and-set, so a streamed row or a greeting
  committed off the room lock is never overwritten by a stale close or pause.
  The shipped stores hold the room's row from the comparison to the write;
  the default reads, compares and writes, and is not atomic.
- `write_change_target(room_id, target)` writes the target of an edit or a
  delete the persistence policy keeps out of the timeline: the target's row
  alone, no row of the change, and when the target is now deleted every edit
  of it marked deleted, in one transaction. A target the room no longer holds
  raises `EventNotFoundError`. Its default writes through `update_event()` and
  is not atomic.
- `delete_event(room_id, event_id, cascade_replies=True)` is a hard delete: the
  event, its thread replies unless told not to, and the edits and deletes of
  every event it removes, in one transaction, their idempotency keys released.
  It returns the removed events as stored: the root, its replies, then the
  changes, each in index order.
- The routing lookups (RFC §10.4, §14.1) answer the newest room, the most
  recently created, the greatest id on a tie, among the rooms a
  `RoomFilter(statuses, organization_id)` reads:
  `find_room_id_by_binding(channel_id, participant_id, identity_id,
  room_filter, *, recipient_key=None)` finds a binding naming the address or
  the identity, or naming no one and not `group` with the address as its
  recipient under `recipient_key` (SQLite reads an integer recipient beyond
  64 bits as a real number and does not find it; the other stores do);
  `find_room_id_by_participant(channel_id,
  participant_id, identity_id, room_filter)` a participant reached through the
  channel whose id or identity is one of the names;
  `find_room_ids_by_channel(channel_id, room_filter, limit, *, group=None)`
  the rooms bound to the channel, oldest first, `group` keeping a group
  binding's rooms or the others. The base class falls back to scans.
- `check_idempotency(room_id, key, *, channel_id)` and
  `get_event_by_idempotency_key(room_id, key, *, channel_id)` look a key up
  within its source channel, or with `channel_id=None` within the whole room
  (the lowest index first), as a proactive delivery's key is.
- `record_delivery_results(room_id, event_id, results)` writes a delivery set's
  outcome on the event as stored, replacing the map, that field alone, so an
  edit or a delete committed while the delivery ran stands. An event is
  committed with the `delivery_results` it brings; once stored, this is the
  one write of `delivery_results`: `update_event()` and a change's target
  (`commit_event(target=)`, `write_change_target()`) keep the stored results,
  whatever the copy they write carries. Its default reads the event and
  writes it back through `update_event()` (the framework calls it under the
  room lock), which works for a store whose `update_event()` writes the row
  as given; a store that keeps the results there overrides
  `record_delivery_results()`, and the default, reading the event back,
  raises `NotImplementedError` saying so rather than record nothing (the
  framework logs it as a delivery record it could not write).
