"""Real-Postgres tests proving the multi-process event-index safety (H1b).

These exercise behaviour that mocks cannot: the UNIQUE(room_id, index)
constraint and PostgreSQL advisory locks under genuine cross-connection
concurrency. Require a running PostgreSQL:

    POSTGRES_DSN=postgresql://user:pass@localhost/roomkit_test \
    pytest tests/test_postgres_multiprocess.py -v
"""

from __future__ import annotations

import asyncio
import os

import asyncpg
import pytest

from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.store.postgres import PostgresStore
from roomkit.store.postgres_lock import PostgresAdvisoryLockManager

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set"),
    pytest.mark.asyncio,
]


@pytest.fixture
async def store():
    s = PostgresStore(dsn=POSTGRES_DSN)
    await s.init(min_size=1, max_size=5)
    async with s._pool.acquire() as conn:
        await conn.execute("TRUNCATE rooms, events CASCADE")
    yield s
    await s.close()


def _event(room_id: str, eid: str, index: int) -> RoomEvent:
    return RoomEvent(
        id=eid,
        room_id=room_id,
        index=index,
        source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
        content=TextContent(body="hi"),
    )


async def test_unique_index_constraint_rejects_duplicate(store: PostgresStore) -> None:
    """The DB rejects a second event at the same (room_id, index) — the backstop
    that turns a multi-process race into a loud failure instead of corruption."""
    await store.create_room(Room(id="r1"))
    await store.add_event(_event("r1", "e1", 0))
    with pytest.raises(asyncpg.UniqueViolationError):
        await store.add_event(_event("r1", "e2", 0))


async def test_advisory_lock_serializes_concurrent_assignment(store: PostgresStore) -> None:
    """Two lock managers (separate pools = two processes) sharing one DB must
    serialize index assignment on the same room via the server-global advisory
    lock, producing unique sequential indices with no collision."""
    await store.create_room(Room(id="r1"))
    mgr_a = PostgresAdvisoryLockManager(dsn=POSTGRES_DSN)
    mgr_b = PostgresAdvisoryLockManager(dsn=POSTGRES_DSN)
    await mgr_a.init()
    await mgr_b.init()

    n = 16

    async def worker(mgr: PostgresAdvisoryLockManager, i: int) -> None:
        # The racy count-then-insert pattern; correctness comes from the lock.
        async with mgr.locked("r1"):
            index = await store.get_event_count("r1")
            await store.add_event(_event("r1", f"e{i}", index))

    try:
        await asyncio.gather(*(worker(mgr_a if i % 2 == 0 else mgr_b, i) for i in range(n)))
    finally:
        await mgr_a.close()
        await mgr_b.close()

    events = await store.list_events("r1", limit=100)
    indices = sorted(e.index for e in events)
    assert indices == list(range(n))  # unique + sequential — no duplicates


async def test_without_cross_process_lock_the_constraint_catches_races(
    store: PostgresStore,
) -> None:
    """Sanity: without a shared lock, concurrent count-then-insert races on the
    same index; the UNIQUE constraint MUST surface at least one violation rather
    than let a duplicate through (no silent corruption)."""
    await store.create_room(Room(id="r1"))

    async def unlocked_worker(i: int) -> bool:
        try:
            index = await store.get_event_count("r1")
            await asyncio.sleep(0)  # widen the race window
            await store.add_event(_event("r1", f"e{i}", index))
            return True
        except asyncpg.UniqueViolationError:
            return False

    results = await asyncio.gather(*(unlocked_worker(i) for i in range(10)))
    # Whatever the interleaving: every stored index is unique (constraint holds),
    # and losers failed loudly rather than duplicating.
    events = await store.list_events("r1", limit=100)
    indices = [e.index for e in events]
    assert len(indices) == len(set(indices))  # no duplicate ever persisted
    assert sum(results) == len(events)  # successes == rows stored
