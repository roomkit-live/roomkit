"""Malformed speaking controls never become the product's fallback decision."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.speaking import SpeakDecision, SpeakDecisionEvent, SpeakTurn
from tests.conformance._script import Gates
from tests.conformance._shape import unknown_members
from tests.conformance._speaking import (
    ScriptedSpeakPolicy,
    SpeakPolicyFailedError,
    patched_decision,
)
from tests.conftest import make_event


@pytest.mark.parametrize(
    "policy",
    [
        None,
        [],
        {},
        {"decisions": []},
        {"decisions": [None]},
        {"decisions": [{"decision": None}]},
        {"decisions": [{"decision": {"mode": "maybe"}}]},
        {"decisions": [{"decision": {"mode": "speak", "final": True}}]},
        {"decisions": [{"decision": {"mode": "silent", "notes": "not an array"}}]},
        {"decisions": [{"decision": {"mode": "silent", "judgments": None}}]},
        {"decisions": [{"decision": {"mode": "silent", "judgments": {"p": True}}}]},
        {"decisions": [{"fail": None}]},
        {"decisions": [{"fail": "failed", "decision": {"mode": "silent"}}]},
        {"decisions": [{"fail": "failed", "gate": None}]},
        {"decisions": [{"decision": {"mode": "silent", "misspelled": 1}}]},
        {"decisions": [{"fail": "failed", "misspelled": 1}]},
        {"decisions": [{"fail": "failed"}], "misspelled": 1},
    ],
)
def test_invalid_policy_is_refused_before_product_execution(policy: Any) -> None:
    scenario = {"setup": {"channels": [{"id": "ai", "type": "ai", "speak_policy": policy}]}}
    assert any("speak_policy" in issue for issue in unknown_members(scenario))
    with pytest.raises(ValueError, match="speak_policy"):
        ScriptedSpeakPolicy(policy, Gates(), [])


async def test_policy_records_real_turns_before_failure_and_repeats_the_last_entry() -> None:
    calls: list[Any] = []
    policy = ScriptedSpeakPolicy(
        {
            "decisions": [
                {"decision": {"mode": "silent", "reason": "side talk", "judgments": {"p": 0.1}}},
                {"fail": "unavailable"},
            ]
        },
        Gates(),
        calls,
    )
    event = make_event(room_id="r1", channel_id="sms", body="Hello")
    turn = SpeakTurn(event, channel_id="ai", speakers={event.id: "Alice"})
    assert await policy.decide(turn) == SpeakDecision("silent", "side talk", {"p": 0.1})
    for _ in range(2):
        with pytest.raises(SpeakPolicyFailedError, match="unavailable"):
            await policy.decide(turn)
    assert len(calls) == 3
    assert all(record["event"]["id"] == event.id for record in calls)
    assert calls[0]["speakers"] == {event.id: "Alice"}


def test_decision_patch_replaces_the_decision_and_preserves_its_event() -> None:
    report = SpeakDecisionEvent("r1", "ai", make_event(), SpeakDecision("offer"))
    patched = patched_decision(report, {"decision": {"mode": "silent", "reason": "hand up"}})
    assert patched.decision == SpeakDecision("silent", "hand up")
    assert patched.event is report.event and patched.channel_id == "ai"
    with pytest.raises(ValueError, match="only decision"):
        patched_decision(report, {"decision": {"mode": "silent"}, "misspelled": 1})


@pytest.mark.parametrize(
    "patch",
    [None, {}, {"decision": None}, {"decision": {"mode": "maybe"}}, {"misspelled": 1}],
)
def test_invalid_decision_patch_is_refused_before_a_hook_can_swallow_it(patch: Any) -> None:
    scenario = {
        "setup": {
            "hooks": [
                {
                    "name": "decision",
                    "trigger": "on_speak_decision",
                    "reactions": [{"result": {"action": "modify", "event": patch}}],
                }
            ]
        }
    }
    assert unknown_members(scenario)
