"""Malformed speaking controls never become the product's fallback decision."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.classifiers import ChoiceQuestion, ClassifierError, ScoreQuestion, YesNoQuestion
from roomkit.speaking import SpeakDecision, SpeakDecisionEvent, SpeakTurn
from tests.conformance._classifier import ScriptedClassifier
from tests.conformance._script import Gates
from tests.conformance._shape import unknown_members
from tests.conformance._speaking import (
    ScriptedSpeakPolicy,
    SpeakPolicyFailedError,
    patched_decision,
)
from tests.conftest import make_event

_SPEAK = {"decisions": [{"decision": {"mode": "speak"}}]}
_CLASSIFIER = {"classifier": {"calls": [{"answers": {}}]}, "agent_name": "Nova"}


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
        {"classifier": {"calls": []}, "agent_name": "Nova"},
        {"classifier": {"calls": [{"answers": []}]}, "agent_name": "Nova"},
        {"classifier": {"calls": [{"answers": {"hush": True}}]}, "agent_name": "Nova"},
        {"classifier": {"calls": [{"answers": {}, "fail": "down"}]}, "agent_name": "Nova"},
        {"classifier": {"calls": [{"answers": {}, "misspelled": 1}]}, "agent_name": "Nova"},
        {"classifier": {"calls": [{"answers": {}}]}},
        {"classifier": {"calls": [{"answers": {}}]}, "agent_name": "Nova", **_SPEAK},
        {"agent_name": "Nova", **_SPEAK},
        {**_CLASSIFIER, "questions": {"hush": {"instructions": 1}}},
        {**_CLASSIFIER, "questions": {"hush": {"instructions": "?", "options": {}, "levels": []}}},
        {**_CLASSIFIER, "languages": {"French": 1}},
        {**_CLASSIFIER, "composition": {"decision": {"mode": "maybe"}}},
        {**_CLASSIFIER, "composition": {"decision": {"mode": "speak"}, "misspelled": 1}},
        {"answer_only": {"people": []}, **_SPEAK},
        {"answer_only": ["alice"], **_SPEAK},
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


async def test_a_scripted_classifier_records_each_call_before_its_gate_or_failure() -> None:
    """Each call is recorded with its state and questions, takes the next
    entry (the last repeating), and answers as the public mock does."""
    calls: list[Any] = []
    gates = Gates()
    problems: list[str] = []
    classifier = ScriptedClassifier(
        {"calls": [{"answers": {"hush": 0.9}, "gate": "held"}, {"fail": "down"}]},
        gates,
        calls,
        problems,
    )
    questions = {"hush": YesNoQuestion("Quiet?"), "level": ScoreQuestion("How?", ("a", "b"))}
    held = asyncio.ensure_future(classifier.classify({"said": "hi"}, questions))
    await gates.wait_for("held")
    assert calls == [
        {
            "state": {"said": "hi"},
            "questions": {
                "hush": {"instructions": "Quiet?", "yes": None, "no": None},
                "level": {"instructions": "How?", "levels": ["a", "b"]},
            },
        }
    ]
    gates.open("held")
    answers = await held
    assert (answers.yes("hush"), answers.score("level")) == (0.9, 0.0)
    for _ in range(2):
        with pytest.raises(ClassifierError, match="down"):
            await classifier.classify("again", questions)
    assert len(calls) == 3
    assert problems == []


@pytest.mark.parametrize(
    "answers",
    [{"hush": "yes"}, {"level": "high"}, {"pick": "c"}, {"pick": 1}, {"husk": 0.9}],
)
async def test_a_scripted_answer_the_question_cannot_take_is_a_problem_not_a_fallback(
    answers: dict[str, Any],
) -> None:
    """The mock would fail such a call, and the policy would fall back to
    speaking: the scenario's mistake would read as the product's behaviour."""
    problems: list[str] = []
    classifier = ScriptedClassifier({"calls": [{"answers": answers}]}, Gates(), [], problems)
    questions = {
        "hush": YesNoQuestion("Quiet?"),
        "level": ScoreQuestion("How?", ("a", "b")),
        "pick": ChoiceQuestion("Which?", {"a": "", "b": ""}),
    }
    with pytest.raises(ClassifierError):
        await classifier.classify("state", questions)
    assert len(problems) == 1
