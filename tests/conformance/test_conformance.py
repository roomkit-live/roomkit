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


# The store's scenarios run on each store this implementation ships that
# needs no server: the in-memory store, the kit's default, and SQLiteStore.
_STORES = ("memory", "sqlite")


def _cases() -> list[Any]:
    cases = []
    for path in _scenarios():
        scenario_id = path.relative_to(SCENARIOS).with_suffix("").as_posix()
        marks = []
        if _turn_calls_the_kit(_load(path)):
            marks.append(pytest.mark.in_process(_IN_PROCESS))
        if scenario_id in KNOWN_FAILURES:
            marks.append(
                pytest.mark.xfail(
                    reason=KNOWN_FAILURES[scenario_id], strict=True, raises=AssertionError
                )
            )
        for store in _STORES if scenario_id.startswith("store/") else ("memory",):
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
