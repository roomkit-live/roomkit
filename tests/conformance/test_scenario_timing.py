"""A timeout scenario must leave time for the worker expected to complete."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.store.sqlite import SQLiteStore
from tests.conformance._harness import Harness

_SCENARIO = (
    Path(__file__).parent
    / "scenarios/orchestration"
    / "a-team-worker-past-its-task-timeout-is-read-as-timed-out-beside-the-others.json"
)


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_a_team_timeout_leaves_room_for_a_slow_successful_worker(
    store: str, tmp_path: Path
) -> None:
    """RMK-1331: a 250 ms pause must not turn the successful sibling into
    another timeout. Run the shared scenario's full assertions unchanged.

    This delay deliberately reproduces CI latency; it does not order workers.
    The analyst remains held forever, so its real timeout still has to fire.
    """
    scenario = json.loads(await asyncio.to_thread(_SCENARIO.read_text, encoding="utf-8"))
    critic = next(
        channel for channel in scenario["setup"]["channels"] if channel["id"] == "critic"
    )
    critic["model"]["rounds"][0]["gate"] = "slow-critic"
    await compile_mark_patterns()
    on = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    reader = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    harness = Harness(scenario, on, observation_store=reader)
    run = asyncio.create_task(harness.run())
    try:
        await harness.gates.wait_for("slow-critic")
        await asyncio.sleep(0.25)
        harness.gates.open("slow-critic")
        problems = await run
        assert not problems, "\n".join(problems)
    finally:
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        if reader is not None:
            await reader.close()
