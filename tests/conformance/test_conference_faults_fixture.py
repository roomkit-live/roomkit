"""The scripted SFU's faults (roomkit-specs ``conformance/README.md``,
"Scripted conference faults"): what a scenario's mistake and an untaken
fault come to, and what a held call does when it is cancelled."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.conference import MockConferenceCall
from tests.conformance._conference_faults import (
    ScriptedBackendError,
    SFUFaults,
    UnarmableFaultError,
)
from tests.conformance._script import Gates


@pytest.mark.parametrize(
    "spec",
    [
        {"method": "join", "fail": "refused"},
        {"method": "leave"},
        {"method": "leave", "fail": "refused", "times": 2},
    ],
    ids=["no-such-call", "neither-fail-nor-gate", "unknown-member"],
)
def test_a_fault_no_call_could_take_is_the_scenario_s_mistake(spec: dict[str, object]) -> None:
    with pytest.raises(UnarmableFaultError):
        SFUFaults(Gates()).arm(spec)


async def test_a_fault_lets_its_after_calls_through_and_an_untaken_one_is_reported() -> None:
    faults = SFUFaults(Gates())
    faults.arm({"method": "leave", "fail": "refused", "after": 1})
    faults.arm({"method": "close", "fail": "refused"})

    await faults.take("leave", MockConferenceCall("leave"))
    with pytest.raises(ScriptedBackendError, match="refused"):
        await faults.take("leave", MockConferenceCall("leave"))

    assert faults.untaken() == ["conference fault on close was never taken"]


@pytest.mark.parametrize("ignores_cancel", [False, True])
async def test_a_call_cancelled_at_its_gate_reads_cancelled(ignores_cancel: bool) -> None:
    gates = Gates()
    faults = SFUFaults(gates)
    faults.arm({"method": "mint_access", "gate": "minting", "ignores_cancel": ignores_cancel})
    record = MockConferenceCall("mint_access")
    call = asyncio.ensure_future(faults.take("mint_access", record))
    await gates.wait_for("minting")

    call.cancel()
    await asyncio.sleep(0)
    if ignores_cancel:
        # It holds on until the gate opens, then does its work.
        assert not call.done()
        gates.open("minting")
        await call
    else:
        with pytest.raises(asyncio.CancelledError):
            await call

    assert record.args["cancelled"] is True
