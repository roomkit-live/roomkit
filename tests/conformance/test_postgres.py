"""The shared scenarios on PostgreSQL: those whose contract writes to the store,
which test_conformance's ``_stores()`` gives PostgreSQL, and the other contracts
whose outcome depends on its reads and writes (``SCENARIOS``).

Each case runs in a schema of its own, created before it and dropped after it,
as in the Rust PostgreSQL runner: the cases never see each other's rows and run
in parallel. A case with other processes (``setup.processes``) gives each kit its
own store and advisory lock manager on that schema, as another process opens
them, and runs alone among such cases (``_alone_with_processes``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.store import PostgresAdvisoryLockManager, PostgresStore
from roomkit.store.postgres import _MIGRATION_LOCK_KEY
from tests.conformance._harness import Harness
from tests.conformance._shape import unknown_members
from tests.conformance.test_conformance import SCENARIOS as SCENARIO_FILES
from tests.conformance.test_conformance import _cases
from tests.conformance.test_scenario_timing import WORKER_BOUNDS, delay_child_preparation
from tests.test_store_postgres import POSTGRES_DSN

try:
    import asyncpg
except ImportError:  # the `postgres` extra; without it POSTGRES_DSN is unset too
    asyncpg = None

pytestmark = pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set")

SCENARIOS = [
    "regenerate/a-regenerated-turn-reads-no-message-committed-after-the-choice",
    "regenerate/a-regenerated-turn-past-the-first-event-reads-no-message-committed-after-the-choice",
    "hooks/an-injection-takes-its-own-lock-after-its-blocked-cause",
    "hooks/an-injection-takes-its-own-lock-after-its-delivered-cause",
    "tasks/a-failed-inline-preparation-keeps-its-child-and-reports-no-task",
    "tasks/a-failed-background-preparation-keeps-its-child-and-reports-no-task",
    "streaming/the-stream-goes-to-the-first-streaming-transport-bound",
    "store/updated-bindings-keep-the-first-streaming-target",
    "store/reattached-binding-moves-behind-existing-streaming-targets",
    *(f"orchestration/{name}" for name in WORKER_BOUNDS),
    *(
        f"orchestration/a-{door}-cut-{boundary}-announcement-reports-only-what-started"
        for door in ("worker", "team")
        for boundary in ("before", "after")
    ),
]

# A case with other processes runs alone among them on the database, as the Rust
# runner runs one case at a time there: their kits take the server's advisory
# locks, which are the database's whatever the schema, and their scenarios name
# the same rooms. A key of the two-int space, which no single-key lock shares.
_PROCESSES_KEY = (0x726B, 0x7072)


def _in_schema(dsn: str, schema: str) -> str:
    """*dsn* with its sessions writing to *schema*."""
    separator = "&" if "?" in dsn else "?"
    return f"{dsn}{separator}options=-csearch_path%3D{schema}"


@pytest.fixture
async def schema() -> AsyncIterator[str]:
    """The connection string of a schema of the case's own, created before the
    case and dropped after it."""
    assert asyncpg is not None and POSTGRES_DSN is not None
    name = f"rk_case_{uuid4().hex}"
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


@asynccontextmanager
async def _alone_with_processes() -> AsyncIterator[None]:
    """Holds the database's lock of the cases with other processes."""
    assert asyncpg is not None
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        await conn.execute("SELECT pg_advisory_lock($1, $2)", *_PROCESSES_KEY)
        yield
    finally:
        await conn.close()


@asynccontextmanager
async def _harness(scenario: Mapping[str, Any], dsn: str) -> AsyncIterator[Harness]:
    """A run of *scenario* on the schema *dsn* writes to, whose stores and lock
    managers are closed once the block is over; the kits close theirs first."""
    async with AsyncExitStack() as opened:

        async def store() -> PostgresStore:
            on = PostgresStore(dsn=dsn)
            await on.init(min_size=1, max_size=3)
            opened.push_async_callback(on.close)
            return on

        async def lock_manager() -> PostgresAdvisoryLockManager:
            locks = PostgresAdvisoryLockManager(dsn, max_size=8)
            await locks.init()
            opened.push_async_callback(locks.close)
            return locks

        async def open_process() -> tuple[PostgresStore, PostgresAdvisoryLockManager]:
            return await store(), await lock_manager()

        await compile_mark_patterns()
        if not scenario.get("setup", {}).get("processes"):
            yield Harness(scenario, await store(), observation_store=await store())
            return
        await opened.enter_async_context(_alone_with_processes())
        yield Harness(
            scenario,
            await store(),
            observation_store=await store(),
            lock_manager=await lock_manager(),
            open_process=open_process,
        )


async def _load(path: Path) -> dict[str, Any]:
    return json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))


@pytest.mark.parametrize(("path", "store"), _cases(stores=("postgres",)))
async def test_scenario(path: Path, store: str, schema: str) -> None:
    """A scenario test_conformance gives PostgreSQL (*store*), on a schema of
    the case's own."""
    scenario = await _load(path)
    assert scenario["id"] == path.relative_to(SCENARIO_FILES).with_suffix("").as_posix()
    assert store == "postgres"
    if unknown := unknown_members(scenario):
        pytest.fail("members the runner does not know:\n" + "\n".join(unknown))
    async with _harness(scenario, schema) as harness:
        problems = await harness.run()
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("scenario_id", "slow_preparation"),
    [(name, False) for name in SCENARIOS]
    + [
        pytest.param(f"orchestration/{name}", True, marks=pytest.mark.wallclock)
        for name in WORKER_BOUNDS
    ],
)
async def test_shared_postgres_contract(
    schema: str,
    scenario_id: str,
    slow_preparation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = await _load(SCENARIO_FILES / f"{scenario_id}.json")
    async with _harness(scenario, schema) as harness:
        prepared = delay_child_preparation(harness, monkeypatch) if slow_preparation else None
        problems = await harness.run()
    if prepared is not None:
        assert len(prepared) == 1
    assert not problems, "\n".join(problems)
