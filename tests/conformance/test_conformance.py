"""The shared conformance suite, run on the Python core (RFC §1.4, §26.1).

The scenarios come from roomkit-specs ``conformance/``, copied here by ``make
conformance-sync``, which records the specs commit in ``SPECS_COMMIT`` and the
copy's fingerprint in ``SCENARIOS_SHA256``; the specs' checker validates them
against their schema there. A
scenario this implementation is known to fail is listed in
``KNOWN_FAILURES`` with what it breaks; it is expected to fail, and passes
the suite only while it does, so a fix shows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.store.sqlite import SQLiteStore
from tests.conformance._harness import Harness
from tests.conformance._shape import unknown_members

HERE = Path(__file__).parent
SCENARIOS = HERE / "scenarios"

# Scenario id → the rule this implementation breaks (roomkit-specs
# conformance/README.md, "Known failures").
KNOWN_FAILURES: dict[str, str] = {}


def _scenarios() -> list[Path]:
    return sorted(SCENARIOS.glob("**/*.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# A turn that calls the kit it holds shares its context with the owner when
# the turn runs in memory; across Redis it goes through its TurnHost instead
# (tests/redis_turn_mode.py), which the scripted agents do not use.
_IN_PROCESS = (
    "a scripted agent's turn calls the kit it holds, whose context an in-memory turn "
    "shares with the owner; across Redis it goes through its TurnHost"
)


def _turn_calls_the_kit(scenario: dict[str, Any]) -> bool:
    """Whether an intelligence channel's turn makes a kit call while it runs."""
    for channel in scenario.get("setup", {}).get("channels", []):
        intelligence = channel.get("category", "intelligence" if channel["type"] == "ai" else "")
        if intelligence == "intelligence" and any(
            r.get("do") for r in channel.get("on_event", [])
        ):
            return True
    return False


# An ASYNC hook on a trigger a turn fires may time out once its turn is over:
# its hook_timeout is then work after the turn, which crosses Redis as such and
# reaches a kit the runner, having seen it idle, is closing already.
_AFTER_ITS_TURN = (
    "an ASYNC hook of a turn's trigger times out after its turn: its hook_timeout "
    "crosses Redis as work after the turn, which the runner's idle wait cannot see"
)
_TURN_TRIGGERS = {"before_tool_use", "on_tool_call"}


def _times_out_after_its_turn(scenario: dict[str, Any]) -> bool:
    """Whether the scenario expects the hook_timeout of an ASYNC hook on a
    trigger a turn fires."""
    observers = {
        hook["name"]
        for hook in scenario.get("setup", {}).get("hooks", [])
        if hook.get("execution") == "async" and hook["trigger"] in _TURN_TRIGGERS
    }
    expected = json.dumps(scenario.get("expect", {}).get("framework_events", {}))
    return any(f'"hook_name": "{name}"' in expected for name in observers)


# A call whose tool answers right after its turn's cut: in memory the cut has
# reached the turn when the gate opens; across Redis it reaches the executor
# later, the tool answers first, and the round's TOOL_CALL_END is not published.
_ORDER_OF_ITS_CUT = (
    "the tool answers right after its turn's cut, which reaches an executor across "
    "Redis later, so the round's TOOL_CALL_END depends on that order"
)
_ANSWERS_AFTER_ITS_CUT = {"tools/a-call-that-returns-once-its-turn-is-cut-ends-aborted"}


# Store and room lifecycle scenarios run on each store this implementation ships that
# needs no server: the in-memory store, the kit's default, and SQLiteStore.
_STORES = ("memory", "sqlite")


def _cases() -> list[Any]:
    cases = []
    for path in _scenarios():
        scenario_id = path.relative_to(SCENARIOS).with_suffix("").as_posix()
        marks = []
        if _turn_calls_the_kit(_load(path)):
            marks.append(pytest.mark.in_process(_IN_PROCESS))
        if _times_out_after_its_turn(_load(path)):
            marks.append(pytest.mark.in_process(_AFTER_ITS_TURN))
        if scenario_id in _ANSWERS_AFTER_ITS_CUT:
            marks.append(pytest.mark.in_process(_ORDER_OF_ITS_CUT))
        if scenario_id in KNOWN_FAILURES:
            marks.append(
                pytest.mark.xfail(
                    reason=KNOWN_FAILURES[scenario_id], strict=True, raises=AssertionError
                )
            )
        for store in _STORES if scenario_id.startswith(("store/", "rooms/")) else ("memory",):
            name = scenario_id if store == "memory" else f"{scenario_id}[{store}]"
            cases.append(pytest.param(path, store, id=name, marks=marks))
    return cases


def test_the_copy_is_the_one_synced_from_its_specs_commit() -> None:
    """The fingerprint ``make conformance-sync`` recorded, ``sha256sum`` over
    ``sha256sum`` of each file of ``scenarios/`` by path in byte order, is the
    copy's."""
    commit = (HERE / "SPECS_COMMIT").read_text(encoding="utf-8").strip()
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    files = sorted(
        path.relative_to(HERE).as_posix() for path in SCENARIOS.rglob("*") if path.is_file()
    )
    lines = "".join(
        f"{hashlib.sha256((HERE / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(files, key=str.encode)
    )
    recorded = (HERE / "SCENARIOS_SHA256").read_text(encoding="utf-8").strip()
    assert hashlib.sha256(lines.encode()).hexdigest() == recorded, (
        "tests/conformance/scenarios differs from its sync: run make conformance-sync"
    )


@pytest.fixture(autouse=True)
async def _compiled_mark_patterns() -> None:
    """The patterns that find a copy of a runtime mark compile once per
    process, as the first agent joins a room (RMK-888): about a second, many
    on a loaded machine. Compiled before the scenario's calls, so a call's
    bound measures the call, not that compilation (RMK-1202)."""
    await compile_mark_patterns()


@pytest.mark.parametrize(("path", "store"), _cases())
async def test_scenario(path: Path, store: str, tmp_path: Path) -> None:
    scenario = _load(path)
    assert scenario["id"] == path.relative_to(SCENARIOS).with_suffix("").as_posix()
    # A member the runner does not know fails the scenario, a known failure's
    # too: pytest.fail is not the AssertionError its xfail expects.
    if unknown := unknown_members(scenario):
        pytest.fail("members the runner does not know:\n" + "\n".join(unknown))
    on = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    problems = await Harness(scenario, on).run()
    assert not problems, "\n".join(problems)
