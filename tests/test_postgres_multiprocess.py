"""Real-Postgres tests proving the multi-process event-index safety (H1b).

These exercise behaviour that mocks cannot: the UNIQUE(room_id, index)
constraint and PostgreSQL advisory locks under genuine cross-connection
concurrency. Most stand for several processes with several pools in this
one; the last two kill a real process holding a room's lock or its delivery
claim. Require a running PostgreSQL:

    POSTGRES_DSN=postgresql://user:pass@localhost/roomkit_test \
    pytest tests/test_postgres_multiprocess.py -v
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncpg
import pytest

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.core.lanes import delivery_claim_key
from roomkit.core.locks import _OTHER_KIND, InMemoryLockManager
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.store.postgres import _MIGRATION_LOCK_KEY, PostgresStore
from roomkit.store.postgres_lock import PostgresAdvisoryLockManager, _advisory_key

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


class _Recording(_Transport):
    """A transport that records the index of each event it is handed."""

    def __init__(self, channel_id: str, delivered: list[int]) -> None:
        super().__init__(channel_id)
        self._delivered = delivered

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self._delivered.append(event.index)
        return ChannelOutput.empty()


async def test_an_event_added_beside_the_kit_holds_its_next_delivery_and_loses_none(
    store: PostgresStore,
) -> None:
    """``add_event_auto_index`` beside a kit's commits on one room (RMK-1394, point 7):
    each index it takes is a hole of the delivery cursor that no lane runs, which the
    kit's lane skips (RFC §13.5). Every message the kit took is delivered once, in index
    order, and only added indexes are skipped: no delivery is lost. Green before the
    lot's fix too: it checks the point, it does not prove the fix."""
    await store.create_room(Room(id="r1"))
    delivered: list[int] = []
    kit = RoomKit(store=store, delivery_gap_timeout=0.02)  # ty: ignore[invalid-argument-type]
    kit.register_channel(_Transport("src"))
    kit.register_channel(_Recording("out", delivered))
    await kit.attach_channel("r1", "src")
    await kit.attach_channel("r1", "out")
    skipped: list[range] = []

    @kit.on("delivery_skipped")
    async def record(event: Any) -> None:
        skipped.append(range(event.data["from_index"], event.data["to_index"] + 1))

    async def send(i: int) -> int:
        result = await kit.process_inbound(
            InboundMessage(channel_id="src", sender_id="u1", content=TextContent(body=f"m{i}")),
            room_id="r1",
        )
        assert result.event is not None
        return result.event.index

    try:
        outcomes = await asyncio.gather(
            *(send(i) for i in range(8)),
            *(store.add_event_auto_index("r1", _event("r1", f"a{i}", 0)) for i in range(8)),
        )
    finally:
        await kit.close()
    sent = sorted(outcomes[:8])
    added = {event.index for event in outcomes[8:]}
    assert delivered == sent
    assert skipped
    assert {index for hole in skipped for index in hole} <= added


# ── A process killed holding a room's lock or its delivery claim (RFC §13.5) ──
#
# The kits above stand for processes inside this one. Here the holder is a real
# process (tests/postgres_lock_holder.py), killed by SIGKILL: it closes nothing,
# so only the server, ending the sessions of the dead process, lets its locks
# go (RK-13.5-003, RK-13.5-005). Each test writes to a schema of its own and
# names a room of its own: advisory locks are the database's, whatever the
# schema.

HOLDER = Path(__file__).with_name("postgres_lock_holder.py")
# The application name of the holder's sessions, by which pg_locks tells its locks.
_HOLDER_APP = "roomkit_killed_holder"


def _with(dsn: str, parameter: str) -> str:
    """*dsn* with *parameter* (``name=value``) in its query."""
    separator = "&" if "?" in dsn else "?"
    return f"{dsn}{separator}{parameter}"


def _in_schema(dsn: str, schema: str) -> str:
    """*dsn* with its sessions writing to *schema*."""
    return _with(dsn, f"options=-csearch_path%3D{schema}")


@pytest.fixture
async def schema_dsn() -> AsyncIterator[str]:
    """The connection string of a schema of the test's own, created before the
    test and dropped after it."""
    assert POSTGRES_DSN is not None
    name = f"rk_death_{uuid4().hex}"
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        await conn.execute(f"CREATE SCHEMA {name}")
    finally:
        await conn.close()
    try:
        yield _in_schema(POSTGRES_DSN, name)
    finally:
        conn = await asyncpg.connect(POSTGRES_DSN)
        try:
            # Under the lock PostgresStore.init() takes first: init reads
            # catalogs a concurrent DROP makes it fail on.
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK_KEY)
                await conn.execute(f"DROP SCHEMA {name} CASCADE")
        finally:
            await conn.close()


class _HolderProcess:
    """tests/postgres_lock_holder.py, sending one message to a room. Its
    standard input is a pipe from this process: it ends with it, killed or
    not."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def start(cls, dsn: str, room_id: str, hold: str) -> _HolderProcess:
        # The DSN in the environment, where ps does not show its password.
        holder_dsn = _with(dsn, f"application_name={_HOLDER_APP}")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(HOLDER),
            room_id,
            hold,
            env={**os.environ, "ROOMKIT_TEST_HOLDER_DSN": holder_dsn},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        return cls(process)

    async def wait_for(self, prefix: str) -> str:
        """The first line the process prints that starts with *prefix*."""
        assert self.process.stdout is not None
        while True:
            got = await asyncio.wait_for(self.process.stdout.readline(), 20)
            assert got, f"the holder process ended before printing {prefix!r}"
            line = got.decode().strip()
            if line.startswith(prefix):
                return line

    async def kill(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
        await self.process.wait()


async def _sessions_on(key: int) -> list[asyncpg.Record]:
    """pg_locks' rows on the advisory lock *key* of this database: each
    session's pid, whether it holds the lock or waits on it, and its
    application name."""
    unsigned = key & 0xFFFF_FFFF_FFFF_FFFF
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        return await conn.fetch(
            "SELECT l.pid, l.granted, a.application_name FROM pg_locks l"
            " JOIN pg_stat_activity a ON a.pid = l.pid"
            " WHERE l.locktype = 'advisory' AND l.objsubid = 1"
            " AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())"
            " AND l.classid = $1 AND l.objid = $2",
            unsigned >> 32,
            unsigned & 0xFFFF_FFFF,
        )
    finally:
        await conn.close()


async def _until[T](check: Callable[[], Awaitable[T]], what: str, within: float = 10.0) -> T:
    """*check*'s first truthy answer, polled: the server's views can't be awaited."""
    deadline = time.monotonic() + within
    while not (found := await check()):
        assert time.monotonic() < deadline, f"not seen within {within} s: {what}"
        await asyncio.sleep(0.01)
    return found


async def _waiting_on(key: int) -> list[asyncpg.Record]:
    """pg_locks' rows on the advisory lock *key*, the holder's first, once a
    session waits on it; none before."""
    rows = sorted(await _sessions_on(key), key=lambda r: not r["granted"])
    return rows if any(not r["granted"] for r in rows) else []


async def _gone(pid: int) -> bool:
    """Whether the server ended the session *pid*: it lists neither the
    session nor any lock of it."""
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        return not await conn.fetch(
            "SELECT pid FROM pg_locks WHERE pid = $1"
            " UNION SELECT pid FROM pg_stat_activity WHERE pid = $1",
            pid,
        )
    finally:
        await conn.close()


async def _held_by_the_holder(key: int) -> int:
    """The pid of the holder's session granted the advisory lock *key*, the
    only session listed on it."""
    rows = await _sessions_on(key)
    assert [(r["granted"], r["application_name"]) for r in rows] == [(True, _HOLDER_APP)]
    return rows[0]["pid"]


async def _own_kit(
    dsn: str, out: Channel, **kit_kwargs: Any
) -> tuple[RoomKit, PostgresStore, PostgresAdvisoryLockManager]:
    """A kit with its own store and advisory lock manager on *dsn*, as another
    process of the deployment opens them."""
    store = PostgresStore(dsn=dsn)
    await store.init(min_size=1, max_size=3)
    locks = PostgresAdvisoryLockManager(dsn, max_size=4)
    await locks.init()
    kit = RoomKit(store=store, lock_manager=locks, **kit_kwargs)
    kit.register_channel(_Transport("src"))
    kit.register_channel(out)
    return kit, store, locks


def _said(body: str) -> InboundMessage:
    return InboundMessage(channel_id="src", sender_id="u1", content=TextContent(body=body))


async def test_a_room_lock_a_killed_process_held_is_released_and_the_room_goes_on(
    schema_dsn: str,
) -> None:
    """A process holds a room's lock in a hook before its commit, and this
    process's kit waits on it on the server. The holder killed, the server
    ends its sessions and drops its lock: the kit takes the lock and commits,
    and the killed process's message was never stored (RK-13.5-003)."""
    room_id = f"room-{uuid4().hex}"
    kit, store, locks = await _own_kit(schema_dsn, _Transport("out"))
    holder: _HolderProcess | None = None
    try:
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "src")
        await kit.attach_channel(room_id, "out")
        holder = await _HolderProcess.start(schema_dsn, room_id, "room-lock")
        await holder.wait_for("holding")
        key = _advisory_key(room_id)
        holder_pid = await _held_by_the_holder(key)
        sent = asyncio.ensure_future(kit.process_inbound(_said("after the kill"), room_id=room_id))
        waiting = await _until(lambda: _waiting_on(key), "this kit waiting on the room's lock")
        # The holder's session holds it, one of this kit's waits on it.
        assert [(r["pid"] == holder_pid, r["granted"]) for r in waiting] == [
            (True, True),
            (False, False),
        ]
        assert waiting[1]["application_name"] != _HOLDER_APP
        assert not sent.done()

        await holder.kill()
        result = await asyncio.wait_for(sent, 10)
        await _until(lambda: _gone(holder_pid), "the killed session ended")
        events = await store.list_events(room_id, limit=50)
    finally:
        if holder is not None:
            await holder.kill()
        await kit.close()
        await store.close()
        await locks.close()

    assert holder.process.returncode == -signal.SIGKILL
    assert result.blocked is False and result.event is not None
    assert [e.content.body for e in events if e.type == EventType.MESSAGE] == ["after the kill"]
    assert await _sessions_on(key) == []


async def test_a_delivery_claim_a_killed_process_held_leaves_its_index_skipped(
    schema_dsn: str,
) -> None:
    """A process committed index N and holds the room's delivery claim,
    delivering N. This process's next message, N+1, waits on the hole at N
    for ``delivery_gap_timeout``, then on the claim on the server: a live
    owner's delivery is never skipped. The holder killed, the server drops
    the claim with its sessions; the lane skips N with ``delivery_skipped``,
    and N+1 is delivered once, the cursor at N+1 (RK-13.5-005)."""
    room_id = f"room-{uuid4().hex}"
    gap = 0.3
    delivered: list[int] = []
    kit, store, locks = await _own_kit(
        schema_dsn, _Recording("out", delivered), delivery_gap_timeout=gap
    )
    skipped: list[dict[str, Any]] = []

    @kit.on("delivery_skipped")
    async def record(event: Any) -> None:
        skipped.append(event.data)

    holder: _HolderProcess | None = None
    try:
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "src")
        await kit.attach_channel(room_id, "out")
        holder = await _HolderProcess.start(schema_dsn, room_id, "delivery-claim")
        lost = int((await holder.wait_for("delivering ")).split()[1])
        claim = _advisory_key(f"{_OTHER_KIND}{delivery_claim_key(room_id)}")
        holder_pid = await _held_by_the_holder(claim)
        before = await store.get_room(room_id)
        started = time.monotonic()
        sent = asyncio.ensure_future(kit.process_inbound(_said("after the kill"), room_id=room_id))
        waiting = await _until(lambda: _waiting_on(claim), "this kit waiting on the claim")
        # The holder's session holds it, one of this kit's waits on it.
        assert [(r["pid"] == holder_pid, r["granted"]) for r in waiting] == [
            (True, True),
            (False, False),
        ]
        assert waiting[1]["application_name"] != _HOLDER_APP
        # Its gap timeout past, the lane skipped nothing of the live owner's.
        assert not sent.done()
        assert skipped == []
        held = await store.get_room(room_id)
        assert held is not None and held.delivered_index == lost - 1

        await holder.kill()
        result = await asyncio.wait_for(sent, 10)
        waited = time.monotonic() - started
        await _until(lambda: _gone(holder_pid), "the killed session ended")
        after = await store.get_room(room_id)
    finally:
        if holder is not None:
            await holder.kill()
        await kit.close()
        await store.close()
        await locks.close()

    assert holder.process.returncode == -signal.SIGKILL
    assert before is not None and (before.latest_index, before.delivered_index) == (lost, lost - 1)
    assert result.event is not None and result.event.index == lost + 1
    assert waited >= gap
    assert skipped == [{"from_index": lost, "to_index": lost}]
    assert delivered == [lost + 1]
    assert after is not None
    assert (after.latest_index, after.delivered_index) == (lost + 1, lost + 1)
    assert await _sessions_on(claim) == []
