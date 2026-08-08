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
from roomkit.store.postgres import PostgresStore

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

PostgresStore creates 11 tables:

| Table | Purpose |
|-------|---------|
| `rooms` | Room records with status, metadata, timers |
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
from roomkit.store.postgres import PostgresStore


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
from roomkit.store.base import ConversationStore
from roomkit.models.room import Room

class RedisStore(ConversationStore):
    async def create_room(self, room: Room) -> Room:
        await self.redis.set(f"room:{room.id}", room.model_dump_json())
        return room

    # Implement all abstract methods...

kit = RoomKit(store=RedisStore())
```
