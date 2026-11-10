"""A timeout scenario must leave time for every turn expected to complete."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.store.sqlite import SQLiteStore
from tests.conformance._harness import Harness

_SCENARIOS = Path(__file__).parent / "scenarios/orchestration"


@pytest.mark.wallclock
@pytest.mark.parametrize("store", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("scenario_name", "channel_id", "round_index"),
    [
        (
            "a-team-worker-past-its-task-timeout-is-read-as-timed-out-beside-the-others",
            "critic",
            0,
        ),
        ("a-supervised-step-past-its-task-timeout-stops-the-chain-unreviewed", "sup", 1),
    ],
    ids=["parallel-worker", "supervisor-dispatch"],
)
async def test_a_team_timeout_leaves_room_for_a_slow_successful_turn(
    store: str, tmp_path: Path, scenario_name: str, channel_id: str, round_index: int
) -> None:
    """RMK-1331: a 250 ms pause must not time out the successful sibling or
    the supervisor's task-framing turn. Run the shared scenario's full assertions unchanged.

    This delay deliberately reproduces CI latency; it does not order workers.
    The analyst remains held forever, so its real timeout still has to fire.
    """
    path = _SCENARIOS / f"{scenario_name}.json"
    scenario = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
    slow = next(
        channel for channel in scenario["setup"]["channels"] if channel["id"] == channel_id
    )
    slow["model"]["rounds"][round_index]["gate"] = "slow-successful-turn"
    await compile_mark_patterns()
    on = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    reader = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    harness = Harness(scenario, on, observation_store=reader)
    run = asyncio.create_task(harness.run())
    try:
        await harness.gates.wait_for("slow-successful-turn")
        await asyncio.sleep(0.25)
        harness.gates.open("slow-successful-turn")
        problems = await run
        assert not problems, "\n".join(problems)
    finally:
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        if reader is not None:
            await reader.close()
