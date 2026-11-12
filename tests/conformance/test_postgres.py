"""Shared contracts whose outcome depends on PostgreSQL reads and writes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.store.postgres import PostgresStore
from tests.conformance._harness import Harness
from tests.conformance.test_scenario_timing import WORKER_BOUNDS, delay_child_preparation
from tests.test_store_postgres import POSTGRES_DSN
from tests.test_store_postgres import store as store

pytestmark = pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set")

SCENARIOS = [
    "delivery/a-hook-waiting-on-an-adopted-call-whose-answer-sends-into-the-room-ends-at-the-timeout",
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


@pytest.mark.parametrize(
    ("scenario_id", "slow_preparation"),
    [(name, False) for name in SCENARIOS]
    + [
        pytest.param(f"orchestration/{name}", True, marks=pytest.mark.wallclock)
        for name in WORKER_BOUNDS
    ],
)
async def test_shared_postgres_contract(
    store: PostgresStore,  # noqa: F811 - imported pytest fixture
    scenario_id: str,
    slow_preparation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).parent / "scenarios" / f"{scenario_id}.json"
    scenario = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
    await compile_mark_patterns()
    reader = PostgresStore(dsn=POSTGRES_DSN)
    await reader.init(min_size=1, max_size=3)
    harness = Harness(scenario, store, observation_store=reader)
    prepared = delay_child_preparation(harness, monkeypatch) if slow_preparation else None
    try:
        problems = await harness.run()
        if prepared is not None:
            assert len(prepared) == 1
        assert not problems, "\n".join(problems)
    finally:
        await reader.close()
