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
the schema, so the tests run in parallel.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from roomkit import AIChannel, Discussion, RoomKit
from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.discussion import _driver, _shared
from roomkit.orchestration.strategies.discussion._config import DiscussionConfig
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


@pytest.mark.wallclock
async def test_a_killed_pilot_s_discussion_is_taken_over_once_its_lease_expires(
    schema_dsn: str,
) -> None:
    """The pilot is killed during its turn, another agent queued behind it.
    Once the lease it stored has expired, this process takes it (rule 16):
    the pilot's turn has ended (rule 6) and is given to no one again, and this
    process gives the queued turn, from the queue and the state the room
    stores."""
    room_id = f"r-{uuid4().hex}"
    generated: list[tuple[str, float]] = []
    store = PostgresStore(dsn=schema_dsn)
    await store.init(min_size=1, max_size=3)
    locks = PostgresAdvisoryLockManager(schema_dsn, max_size=8)
    await locks.init()
    kit = RoomKit(store=store, lock_manager=locks)
    kit.register_channel(SimpleChannel("ops"))
    pilot = await _Pilot.start(schema_dsn, room_id)
    try:
        holder = (await pilot.line_starting("installed ")).removeprefix("installed ")
        agents = [AIChannel(name, provider=_Answering(name, generated)) for name in AGENTS]
        await kit.install_strategy(room_id, Discussion(agents))
        await _say(kit, room_id, "@investigator what do the logs say?")
        await pilot.line_starting("generating investigator")
        await _say(kit, room_id, "@sre and the metrics?")

        async def sre_queued() -> SpeakQueueState | None:
            state = await _stored(store, room_id)
            return state if [e.agent for e in state.entries] == ["sre"] else None

        # The pilot holds the lease and runs investigator's turn, sre queued.
        held = await _until(sre_queued, "sre queued in the stored queue")
        assert (held.lease_holder, held.speaking, held.turns_given) == (holder, "investigator", 1)
        # Compiled once per process, about a second: the bound below is the lease's.
        await compile_mark_patterns()
        await pilot.kill()
        killed_at = time.time()
        expires = (await _stored(store, room_id)).lease_expires

        async def at_rest() -> bool:
            queue = kit.speak_queue(room_id)
            return bool(generated) and queue is not None and not queue.queue and not queue.speaking

        await _until(at_rest, "the queued turn given and ended", within=3 * LEASE + 10)
        after = await _stored(store, room_id)
        answers = [
            e.source.channel_id
            for e in await store.list_events(room_id, limit=100)
            if e.type == EventType.MESSAGE and e.source.channel_id in AGENTS
        ]
        room = await store.get_room(room_id)
    finally:
        await pilot.kill()
        await kit.close()
        await locks.close()
        await store.close()

    # Killed holding the lease; its turn ended, given to no one again.
    assert expires > killed_at
    [(agent, started)] = generated
    assert agent == "sre"
    assert expires <= started < expires + 3 * LEASE
    assert answers == ["sre"]
    assert (after.lease_holder, after.turns_given, after.entries) == (kit._instance_id, 2, [])
    assert room is not None and DiscussionConfig.stored(room.metadata) is not None
