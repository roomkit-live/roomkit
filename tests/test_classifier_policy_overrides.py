"""Public method overrides keep their order and the policy's failure boundary."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from roomkit import (
    ClassifierSpeakPolicy,
    MockClassifier,
    ScoreAnswer,
    ScoreQuestion,
    SpeakDecision,
    SpeakTurn,
)
from roomkit.classifiers import Answers, Question, State
from tests.conftest import make_event


def _turn(body: str = "A question.") -> SpeakTurn:
    return SpeakTurn(make_event(room_id="r1", channel_id="sms", body=body), channel_id="ai")


async def test_all_three_overrides_dispatch_in_order_and_can_delegate_to_the_same_policy() -> None:
    seen: list[str] = []
    states: list[dict[str, Any]] = []
    asked: list[dict[str, Question]] = []
    standards: list[SpeakDecision] = []
    extra = ScoreQuestion("How urgent?", ("later", "now"))

    class Classifier(MockClassifier):
        async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
            seen.append("classify")
            assert state is states[-1] and questions is asked[-1]
            return await super().classify(state, questions)

    class Policy(ClassifierSpeakPolicy):
        def state(self, turn: SpeakTurn) -> dict[str, Any]:
            seen.append("state")
            value = super().state(turn)
            value["application"] = "live"
            states.append(value)
            return value

        def questions(self, turn: SpeakTurn) -> dict[str, Question]:
            seen.append("questions")
            value = super().questions(turn)
            value["urgency"] = extra
            asked.append(value)
            return value

        def decision(self, turn: SpeakTurn, answers: Answers) -> SpeakDecision:
            seen.append("decision")
            standard = super().decision(turn, answers)
            standards.append(standard)
            assert answers.score("urgency") == 1.0
            return dataclasses.replace(standard, mode="speak", reason="application decision")

    classifier = Classifier({"urgency": 1.0})
    policy = Policy(classifier, agent_name="Nova")
    chosen = await policy.decide(_turn())
    assert seen == ["state", "questions", "classify", "decision"]
    assert states[0]["agent"] == {"name": "Nova"}
    assert classifier.calls[0][0] is states[0]
    assert classifier.calls[0][1]["urgency"] is extra
    assert (standards[0].mode, standards[0].reason) == ("silent", "not addressed")
    assert (chosen.mode, chosen.reason) == ("speak", "application decision")
    seen.clear()
    assert policy.state(_turn())["application"] == "live"
    assert policy.questions(_turn())["urgency"] is extra
    direct = policy.decision(_turn(), Answers(urgency=ScoreAnswer(1.0)))
    assert direct.reason == "application decision"
    assert seen == ["state", "questions", "decision"]


@pytest.mark.parametrize("method", ["state", "questions", "decision"])
@pytest.mark.parametrize("listening", [False, True], ids=["open", "listening"])
async def test_override_errors_keep_the_classification_boundary(
    method: str, listening: bool
) -> None:
    failure = RuntimeError(f"{method} failed")

    class Policy(ClassifierSpeakPolicy):
        failed_method: str | None = None

        def state(self, turn: SpeakTurn) -> dict[str, Any]:
            if self.failed_method == "state":
                raise failure
            return super().state(turn)

        def questions(self, turn: SpeakTurn) -> dict[str, Question]:
            if self.failed_method == "questions":
                raise failure
            return super().questions(turn)

        def decision(self, turn: SpeakTurn, answers: Answers) -> SpeakDecision:
            if self.failed_method == "decision":
                raise failure
            return super().decision(turn, answers)

    classifier = MockClassifier([{"listen_request": 1.0}, {}] if listening else [{}])
    policy = Policy(classifier, agent_name="Nova")
    if listening:
        initial = await policy.decide(_turn("Only listen."))
        assert (initial.mode, initial.reason, initial.final) == ("silent", "asked to listen", True)
    before = len(classifier.calls)
    policy.failed_method = method
    if listening and method != "decision":
        kept = await policy.decide(_turn())
        assert kept == SpeakDecision("silent", "listening", {"listening": 1.0}, final=True)
    else:
        with pytest.raises(RuntimeError) as raised:
            await policy.decide(_turn())
        assert raised.value is failure
    assert len(classifier.calls) - before == (1 if method == "decision" else 0)


async def test_overrides_can_replace_all_three_rules_without_running_the_standard() -> None:
    state: dict[str, Any] = {"custom": "state"}
    questions: dict[str, Question] = {"listen_request": ScoreQuestion("Listen?", ("no", "yes"))}

    class Policy(ClassifierSpeakPolicy):
        def state(self, turn: SpeakTurn) -> dict[str, Any]:
            return state

        def questions(self, turn: SpeakTurn) -> dict[str, Question]:
            return questions

        def decision(self, turn: SpeakTurn, answers: Answers) -> SpeakDecision:
            assert answers.score("listen_request") == 1.0
            return SpeakDecision("offer", "custom rule")

    classifier = MockClassifier({"listen_request": 1.0})
    policy = Policy(classifier, agent_name="Nova")
    assert await policy.decide(_turn()) == SpeakDecision("offer", "custom rule")
    assert classifier.calls[0][0] is state
    assert list(classifier.calls[0][1]) == ["listen_request"]
    # The standard decision would have put this room into its listening state.
    assert "listening_only" not in ClassifierSpeakPolicy.state(policy, _turn())["agent"]
    assert "listen_request" in ClassifierSpeakPolicy.questions(policy, _turn())


@pytest.mark.parametrize("listening", [False, True], ids=["open", "listening"])
async def test_a_turn_without_text_bypasses_all_three_overrides(listening: bool) -> None:
    seen: list[str] = []

    class Policy(ClassifierSpeakPolicy):
        fail: bool = False

        def state(self, turn: SpeakTurn) -> dict[str, Any]:
            seen.append("state")
            assert not self.fail, "state must not run without text"
            return super().state(turn)

        def questions(self, turn: SpeakTurn) -> dict[str, Question]:
            seen.append("questions")
            assert not self.fail, "questions must not run without text"
            return super().questions(turn)

        def decision(self, turn: SpeakTurn, answers: Answers) -> SpeakDecision:
            seen.append("decision")
            assert not self.fail, "decision must not run without text"
            return super().decision(turn, answers)

    classifier = MockClassifier({"listen_request": 1.0})
    policy = Policy(classifier, agent_name="Nova")
    if listening:
        assert (await policy.decide(_turn())).reason == "asked to listen"
    before = len(classifier.calls)
    seen.clear()
    policy.fail = True
    chosen = await policy.decide(_turn(""))
    expected = (
        SpeakDecision("silent", "listening", {"listening": 1.0}, final=True)
        if listening
        else SpeakDecision("speak", "nothing to judge")
    )
    assert chosen == expected
    assert len(classifier.calls) == before
    assert seen == []
