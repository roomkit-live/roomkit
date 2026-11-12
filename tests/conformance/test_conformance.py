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
    """Whether an intelligence channel's turn makes a kit call while it runs;
    a verb of its turn's host (``host.*``) and a read of what an executor did
    (``executor.calls``) are none."""
    for channel in scenario.get("setup", {}).get("channels", []):
        intelligence = channel.get("category", "intelligence" if channel["type"] == "ai" else "")
        if intelligence == "intelligence" and any(
            not action["call"].startswith(("host.", "executor.calls"))
            for reaction in channel.get("on_event", [])
            for action in reaction.get("do", [])
        ):
            return True
    return False


# The scenarios whose contract the store carries run on each store that needs no
# server: the in-memory store, the kit's default, and SQLiteStore. Store, routing,
# tenancy, idempotency and room lifecycle; composition, which leaves no persisted
# partial calls; locks, where a failed release undoes nothing the store holds and the
# delivery cursor is a store write (§13.5), with the two discussion regenerations that
# carry their selection lock's failed release; identity verification, whose requests,
# verifications and held messages are stored (§11.7); and every scenario with other
# processes, whose kits share only the store and the lock manager: a discussion keeps
# its state, queue and lease in its room (§19.7.5 rule 16).
_STORES = ("memory", "sqlite")

_ON_EACH_STORE = (
    "store/",
    "rooms/",
    "idempotency/",
    "routing/",
    "tenancy/",
    "delivery/",
    "broadcast/",
    "tool-composition/",
    "channel-call/",
    "orchestration/a-worker-cut-",
    "orchestration/a-team-cut-",
    "locks/",
    "verification/",
    "discussion/a-discussion-regeneration-carries-its-selection-lock-release-error",
)

# Of those, the ones that run on PostgreSQL too (tests/conformance/test_postgres.py),
# each case on a schema of its own: the idempotency key read under the lock (§13.4),
# the room a message routes to (§10.4 step 1), the delivery a closing kit abandons or
# runs whatever the store (§12.10.4), a channel's state kept with its room (§10.6.2),
# the lock releases and the delivery cursor (§13.5), identity verification (§11.7),
# and every scenario with other processes, whose kits each open their own store and
# advisory lock manager there.
_ON_POSTGRES = (
    "idempotency/",
    "routing/",
    "delivery/",
    "channel-call/",
    "locks/",
    "verification/",
    "discussion/a-discussion-regeneration-carries-its-selection-lock-release-error",
)

_WORKER_DEADLINES = {
    "orchestration/a-delegation-its-call-bound-cuts-posts-the-worker-cancelled",
    "orchestration/a-team-call-its-bound-cuts-posts-each-worker-cancelled",
    "orchestration/a-worker-past-its-task-timeout-is-cut-and-read-as-timed-out",
}

# Scenario id → why it runs in memory only, and its card. Each passes there only
# because the in-memory store never yields; each entry goes once its card is fixed.
_MEMORY_ONLY = {
    "locks/a-room-taken-from-its-message-leaves-its-failed-release-out": (
        "RMK-1661: its two ON_ROOM_CREATED hooks run in parallel, and on SQLite the "
        "attach's release fails after the scenario set releases failing"
    ),
    "verification/a-release-from-a-hook-that-closes-the-room-is-answered-by-no-agent": (
        "RMK-1660: the released answer leaves the lane before the hook's work is over, "
        "while close_room waits on the file"
    ),
}


def _runs_in_processes(scenario: dict[str, Any]) -> bool:
    """Whether other processes serve the scenario's rooms too (``setup.processes``)."""
    return bool(scenario.get("setup", {}).get("processes"))


def _stores(scenario_id: str, scenario: dict[str, Any]) -> tuple[str, ...]:
    if scenario_id in _MEMORY_ONLY:
        return ("memory",)
    if scenario_id.startswith(_ON_POSTGRES) or _runs_in_processes(scenario):
        return (*_STORES, "postgres")
    if scenario_id.startswith(_ON_EACH_STORE) or scenario_id in _WORKER_DEADLINES:
        return _STORES
    return ("memory",)


def _cases(stores: tuple[str, ...] = _STORES) -> list[Any]:
    """The cases of every scenario on each of *stores* it runs on."""
    cases = []
    for path in _scenarios():
        scenario_id = path.relative_to(SCENARIOS).with_suffix("").as_posix()
        scenario = _load(path)
        marks = []
        if _turn_calls_the_kit(scenario):
            marks.append(pytest.mark.in_process(_IN_PROCESS))
        if scenario_id in KNOWN_FAILURES:
            marks.append(
                pytest.mark.xfail(
                    reason=KNOWN_FAILURES[scenario_id], strict=True, raises=AssertionError
                )
            )
        for store in (s for s in _stores(scenario_id, scenario) if s in stores):
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
    reader = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    try:
        problems = await Harness(scenario, on, observation_store=reader).run()
    finally:
        if reader is not None:
            await reader.close()
    assert not problems, "\n".join(problems)
