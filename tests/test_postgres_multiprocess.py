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

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType
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


async def _seed_duplicates(store: PostgresStore, room_id: str = "r1", n: int = 3) -> None:
    """Degrade to a pre-fix DB: non-unique index + ``n`` events all at index 0."""
    async with store._pool.acquire() as conn:
        await conn.execute(
            "DROP INDEX IF EXISTS idx_events_room_index; "
            "CREATE INDEX idx_events_room_index ON events(room_id, index)"
        )
    await store.create_room(Room(id=room_id))
    for i in range(n):
        await store.add_event(_event(room_id, f"d{i}", 0))


async def test_dedupe_dry_run_reports_without_changing(store: PostgresStore) -> None:
    await _seed_duplicates(store, n=3)
    report = await store.dedupe_event_indices()  # dry_run=True
    assert report["action"] == "dry_run"
    assert report["duplicate_rows"] == 2  # 3 rows at index 0 → 2 extra
    assert report["affected_rooms"] == 1
    assert report["now_unique"] is False
    events = await store.list_events("r1", limit=100)
    assert sorted(e.index for e in events) == [0, 0, 0]  # unchanged


async def test_dedupe_repairs_and_enforces_unique(store: PostgresStore) -> None:
    await _seed_duplicates(store, n=3)
    report = await store.dedupe_event_indices(dry_run=False)
    assert report["action"] == "repaired"
    assert report["now_unique"] is True
    events = await store.list_events("r1", limit=100)
    assert sorted(e.index for e in events) == [0, 1, 2]  # unique + sequential
    with pytest.raises(asyncpg.UniqueViolationError):  # constraint now enforced
        await store.add_event(_event("r1", "x", 0))
    room = await store.get_room("r1")
    assert room is not None
    assert room.event_count == 3
    assert room.latest_index == 2  # counters reconciled


async def test_dedupe_noop_on_clean(store: PostgresStore) -> None:
    await store.create_room(Room(id="r1"))
    await store.add_event_auto_index("r1", _event("r1", "e1", 0))
    report = await store.dedupe_event_indices(dry_run=False)
    assert report["action"] == "noop"
    assert report["now_unique"] is True


# ── End-to-end: the real inbound pipeline commits atomically (RFC §14.3) ──────
#
# The tests above drive the store directly. These drive two *RoomKit instances*
# sharing one database through ``process_inbound`` — the path the reviewer noted
# was previously bypassed — proving that index assignment, event insertion and
# the room-counter bump land as one atomic store transaction (``commit_event``),
# so the timeline and ``rooms.event_count`` / ``latest_index`` never diverge.


class _Transport(Channel):
    """Minimal transport: turns an inbound message into a RoomEvent."""

    channel_type = ChannelType.WEBSOCKET

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def _make_kit(dsn: str, lock_manager: object) -> tuple[RoomKit, PostgresStore]:
    """A RoomKit backed by its own Postgres pool (a distinct 'process')."""
    store = PostgresStore(dsn=dsn)
    await store.init(min_size=1, max_size=5)
    kit = RoomKit(store=store, lock_manager=lock_manager)  # ty: ignore[invalid-argument-type]
    kit.register_channel(_Transport("src"))
    return kit, store


async def _assert_dense_and_consistent(
    store: PostgresStore, room_id: str, n_messages: int
) -> None:
    """The §14.3 invariant, through the real pipeline: indices are dense from 0
    (unique + sequential — no duplicate, no gap), all ``n_messages`` inbound
    messages landed (none lost to a race), and the room counters match the
    timeline exactly (an attach system event precedes the messages)."""
    events = await store.list_events(room_id, limit=n_messages + 10)
    indices = sorted(e.index for e in events)
    assert indices == list(range(len(events))), indices  # dense, unique, no gap
    assert sum(e.type == EventType.MESSAGE for e in events) == n_messages  # nothing lost
    room = await store.get_room(room_id)
    assert room is not None
    assert room.event_count == len(events)  # counters reflect the timeline
    assert room.latest_index == len(events) - 1


async def test_inbound_pipeline_commits_atomically_with_advisory_lock(
    store: PostgresStore,
) -> None:
    """Two RoomKit instances (separate advisory-lock pools = two processes)
    sharing one DB, driving concurrent inbound through the real pipeline: every
    index is unique + sequential and the room counters stay consistent."""
    await store.create_room(Room(id="r1"))

    mgr_a = PostgresAdvisoryLockManager(dsn=POSTGRES_DSN)
    mgr_b = PostgresAdvisoryLockManager(dsn=POSTGRES_DSN)
    await mgr_a.init()
    await mgr_b.init()
    kit_a, store_a = await _make_kit(POSTGRES_DSN, mgr_a)
    kit_b, store_b = await _make_kit(POSTGRES_DSN, mgr_b)
    await kit_a.attach_channel("r1", "src")

    n = 16

    async def send(kit: RoomKit, i: int) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id="src", sender_id="u1", content=TextContent(body=f"m{i}")),
            room_id="r1",
        )

    try:
        await asyncio.gather(*(send(kit_a if i % 2 == 0 else kit_b, i) for i in range(n)))
        await _assert_dense_and_consistent(store, "r1", n)
    finally:
        await kit_a.close()
        await kit_b.close()
        await store_a.close()
        await store_b.close()
        await mgr_a.close()
        await mgr_b.close()


async def test_inbound_pipeline_atomic_commit_serializes_without_advisory_lock(
    store: PostgresStore,
) -> None:
    """Even with per-process (in-memory) locks only, ``commit_event``'s
    ``SELECT ... FOR UPDATE`` on the room row serializes concurrent writers at
    the STORAGE layer (RFC §8.1) — so two instances sharing one DB still produce
    unique sequential indices and consistent counters, no duplicate, no loss."""
    await store.create_room(Room(id="r1"))

    kit_a, store_a = await _make_kit(POSTGRES_DSN, InMemoryLockManager())
    kit_b, store_b = await _make_kit(POSTGRES_DSN, InMemoryLockManager())
    await kit_a.attach_channel("r1", "src")

    n = 16

    async def send(kit: RoomKit, i: int) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id="src", sender_id="u1", content=TextContent(body=f"m{i}")),
            room_id="r1",
        )

    try:
        await asyncio.gather(*(send(kit_a if i % 2 == 0 else kit_b, i) for i in range(n)))
        await _assert_dense_and_consistent(store, "r1", n)
    finally:
        await kit_a.close()
        await kit_b.close()
        await store_a.close()
        await store_b.close()


async def test_ai_reentry_commits_delivered_with_consistent_counters(store: PostgresStore) -> None:
    """An AI response (a reentry) commits DELIVERED with room counters that match
    the timeline through the REAL Postgres store — the atomic reentry commit of
    RFC §10.1 step 13, exercised end-to-end via the pipeline."""
    from roomkit.channels.ai import AIChannel
    from roomkit.models.enums import ChannelCategory, EventStatus
    from roomkit.providers.ai.mock import MockAIProvider

    await store.create_room(Room(id="r1"))
    kit = RoomKit(store=store)  # ty: ignore[invalid-argument-type]
    kit.register_channel(_Transport("ws"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["hi there"])))
    await kit.attach_channel("r1", "ws")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hello")),
            room_id="r1",
        )
        events = await store.list_events("r1", limit=100)
        ai_events = [e for e in events if e.source.channel_id == "ai"]
        assert ai_events, "the AI response must be persisted"
        assert all(e.status is EventStatus.DELIVERED for e in ai_events)
        room = await store.get_room("r1")
        assert room is not None
        assert room.event_count == len(events)
        assert room.latest_index == max(e.index for e in events)
    finally:
        await kit.close()
