"""A discussion's pilot killed while it holds the lease (RFC §19.7.5 rule 16),
on a real PostgreSQL: set POSTGRES_DSN.

The pilot is a process of its own (tests/postgres_discussion_pilot.py), killed
by SIGKILL: it renews nothing, gives nothing back and records no end. This
test's process is the deployment's other process, with a PostgresStore and a
PostgresAdvisoryLockManager of its own on the same schema: the lease and the
queue are stored with the room, and a store both processes read and a lock
manager both take are what several processes need; an in-memory lock manager
coordinates no other process (RFC §13.5). Each test writes to a schema of its
own and names a room of its own, advisory locks being the database's whatever
the schema, so the tests can run in parallel.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from roomkit import AIChannel, Discussion, EventType, InboundMessage, RoomKit, TextContent
from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.orchestration.strategies.discussion import _driver, _shared
from roomkit.orchestration.strategies.discussion._queue import SpeakQueueState
from roomkit.providers.ai import AIContext, AIResponse, MockAIProvider
from roomkit.store import PostgresAdvisoryLockManager, PostgresStore
from roomkit.store.postgres import _MIGRATION_LOCK_KEY
from tests.postgres_discussion_pilot import AGENTS, LEASE, POLL
from tests.test_framework import SimpleChannel

try:
    import asyncpg
except ImportError:  # the `postgres` extra; without it POSTGRES_DSN is unset too
    asyncpg = None

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

pytestmark = pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set")

PILOT = Path(__file__).parents[1] / "postgres_discussion_pilot.py"


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll and lease on the pilot's time scale."""
    monkeypatch.setattr(_driver, "POLL_SECONDS", POLL)
    monkeypatch.setattr(_shared, "LEASE_SECONDS", LEASE)
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", LEASE * 2 / 3)


@pytest.fixture
async def schema_dsn() -> AsyncIterator[str]:
    """The connection string of a schema of the test's own, created before the
    test and dropped after it."""
    assert asyncpg is not None and POSTGRES_DSN is not None
    name = f"rk_pilot_{uuid4().hex}"
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        await conn.execute(f"CREATE SCHEMA {name}")
    finally:
        await conn.close()
    separator = "&" if "?" in POSTGRES_DSN else "?"
    try:
        yield f"{POSTGRES_DSN}{separator}options=-csearch_path%3D{name}"
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


class _Pilot:
    """tests/postgres_discussion_pilot.py: the room's discussion, installed in
    a process of its own."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def start(cls, dsn: str, room_id: str) -> _Pilot:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(PILOT), dsn, room_id, stdout=asyncio.subprocess.PIPE
        )
        return cls(process)

    async def line_starting(self, prefix: str) -> str:
        """The next line the process prints that starts with *prefix*."""
        assert self.process.stdout is not None
        while True:
            got = await asyncio.wait_for(self.process.stdout.readline(), 20)
            assert got, f"the pilot process ended before printing {prefix!r}"
            line = got.decode().strip()
            if line.startswith(prefix):
                return line

    async def kill(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
        await self.process.wait()


class _Answering(MockAIProvider):
    """Answers at once, noting in *log* when each generation started."""

    def __init__(self, name: str, log: list[tuple[str, float]]) -> None:
        super().__init__(ai_responses=[AIResponse(content=f"{name} answers")] * 5)
        self.who, self.log = name, log

    async def generate(self, context: AIContext) -> AIResponse:
        self.log.append((self.who, time.time()))
        return await super().generate(context)


async def _stored(store: PostgresStore, room_id: str) -> SpeakQueueState:
    """The speak queue and the lease, as the room stores them."""
    room = await store.get_room(room_id)
    assert room is not None
    return SpeakQueueState.model_validate(room.metadata[_shared.STATE_KEY])


async def _until[T](check: Callable[[], Awaitable[T]], what: str, within: float = 10.0) -> T:
    """*check*'s first truthy answer, polled: a store read can't be awaited."""
    deadline = time.monotonic() + within
    while not (found := await check()):
        assert time.monotonic() < deadline, f"not seen within {within} s: {what}"
        await asyncio.sleep(POLL)
    return found


async def _say(kit: RoomKit, room_id: str, body: str) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body)),
        room_id=room_id,
    )


@dataclass
class _Room:
    """A room whose discussion the pilot gives the turns of, and this
    process's kit, which installed the discussion too."""

    room_id: str
    store: PostgresStore
    kit: RoomKit
    pilot: _Pilot
    holder: str
    """The pilot's process, as the lease names it."""
    generated: list[tuple[str, float]] = field(default_factory=list)
    """The turns this process generated: agent, when it started."""


@asynccontextmanager
async def _piloted(dsn: str) -> AsyncIterator[_Room]:
    """The room, investigator's turn running in the pilot's process."""
    room_id = f"r-{uuid4().hex}"
    generated: list[tuple[str, float]] = []
    store = PostgresStore(dsn=dsn)
    await store.init(min_size=1, max_size=3)
    locks = PostgresAdvisoryLockManager(dsn, max_size=8)
    await locks.init()
    kit = RoomKit(store=store, lock_manager=locks)
    kit.register_channel(SimpleChannel("ops"))
    pilot: _Pilot | None = None
    try:
        pilot = await _Pilot.start(dsn, room_id)
        holder = (await pilot.line_starting("holding ")).removeprefix("holding ")
        assert (await _stored(store, room_id)).lease_holder == holder
        agents = [AIChannel(name, provider=_Answering(name, generated)) for name in AGENTS]
        await kit.install_strategy(room_id, Discussion(agents))
        await _say(kit, room_id, "@investigator what do the logs say?")
        await pilot.line_starting("generating investigator")
        # Compiled once per process, about a second: the bounds after the
        # kill are the lease's.
        await compile_mark_patterns()
        yield _Room(room_id, store, kit, pilot, holder, generated)
    finally:
        if pilot is not None:
            await pilot.kill()
        await kit.close()
        await locks.close()
        await store.close()


async def _killed(room: _Room) -> SpeakQueueState:
    """Kill the pilot: the room as it stores the queue and the lease then,
    the lease the pilot's and not expired when it died."""
    await room.pilot.kill()
    killed_at = time.time()
    state = await _stored(room.store, room.room_id)
    assert state.lease_holder == room.holder
    assert state.lease_expires > killed_at
    return state


@pytest.mark.wallclock
async def test_a_killed_pilot_s_queued_turn_is_given_once_its_lease_expires(
    schema_dsn: str,
) -> None:
    """The pilot is killed during its turn, another agent queued behind it.
    Once the lease it stored has expired, and not before, this process takes
    it and gives the queued turn, from the queue and the state the room
    stores (rule 16); the pilot's turn is given to no one again (rule 6)."""
    async with _piloted(schema_dsn) as room:
        await _say(room.kit, room.room_id, "@sre and the metrics?")

        async def sre_queued() -> SpeakQueueState | None:
            state = await _stored(room.store, room.room_id)
            return state if [e.agent for e in state.entries] == ["sre"] else None

        held = await _until(sre_queued, "sre queued in the stored queue")
        assert (held.speaking, held.turns_given) == ("investigator", 1)
        expires = (await _killed(room)).lease_expires

        async def at_rest() -> bool:
            queue = room.kit.speak_queue(room.room_id)
            idle = queue is not None and not queue.queue and not queue.speaking
            return bool(room.generated) and idle

        await _until(at_rest, "the queued turn given and ended", within=3 * LEASE + 10)
        after = await _stored(room.store, room.room_id)
        answers = [
            e.source.channel_id
            for e in await room.store.list_events(room.room_id, limit=100)
            if e.type == EventType.MESSAGE and e.source.channel_id in AGENTS
        ]

    [(agent, started)] = room.generated
    assert agent == "sre"
    assert expires <= started < expires + 3 * LEASE
    assert answers == ["sre"]
    assert (after.lease_holder, after.turns_given, after.entries) == (room.kit._instance_id, 2, [])


@pytest.mark.wallclock
async def test_a_killed_pilot_s_running_turn_has_ended_once_its_lease_is_taken(
    schema_dsn: str,
) -> None:
    """The pilot is killed during its turn, nothing queued behind it. Once
    the lease expired, this process takes it, and the room stores no running
    turn: the pilot's has ended (rules 6 and 16), its agent neither queued
    again nor given a turn here."""
    async with _piloted(schema_dsn) as room:
        held = await _stored(room.store, room.room_id)
        assert (held.speaking, held.turns_given, held.entries) == ("investigator", 1, [])
        await _killed(room)

        async def taken() -> SpeakQueueState | None:
            state = await _stored(room.store, room.room_id)
            return state if state.lease_holder == room.kit._instance_id else None

        after = await _until(taken, "this process holding the lease", within=3 * LEASE + 10)

    assert (after.speaking, after.turns_given, after.entries) == (None, 1, [])
    assert room.generated == []
