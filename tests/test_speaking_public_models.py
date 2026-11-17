"""Public speaking/classifier carriers and the aliases their callers retain."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

import pytest

from roomkit import (
    AlwaysSpeak,
    AnswerOnly,
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierError,
    ClassifierSpeakPolicy,
    CutReply,
    MockClassifier,
    MockSpeakPolicy,
    MockThinker,
    ScoreAnswer,
    ScoreQuestion,
    SpeakDecision,
    SpeakDecisionEvent,
    SpeakPolicy,
    SpeakTurn,
    Thinker,
    Thought,
    ThoughtEvent,
    YesNoAnswer,
    YesNoQuestion,
)
from roomkit.classifiers import Answers, Classifier, check_answers
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext


def _event(channel_id: str = "sms") -> RoomEvent:
    return RoomEvent(
        id="e1",
        room_id="r1",
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.SMS),
        content=TextContent(body="Hello"),
    )


def test_decision_defaults_are_isolated_and_serialize_as_plain_data() -> None:
    first, second = SpeakDecision("silent"), SpeakDecision("silent")
    assert dataclasses.asdict(first) == {
        "mode": "silent",
        "reason": "",
        "judgments": {},
        "notes": (),
        "final": False,
    }
    first.judgments["heard"] = 0.75
    assert second.judgments == {}
    assert json.loads(json.dumps(dataclasses.asdict(first))) == {
        "mode": "silent",
        "reason": "",
        "judgments": {"heard": 0.75},
        "notes": [],
        "final": False,
    }


def test_frozen_decision_keeps_the_supplied_judgment_dictionary_live() -> None:
    judgments = {"heard": 0.25}
    decision = SpeakDecision("silent", judgments=judgments)
    judgments["heard"] = 0.75
    assert decision.judgments is judgments
    assert decision.judgments == {"heard": 0.75}
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.reason = "new"  # type: ignore[misc]


def test_speak_turn_retains_its_event_speakers_and_optional_carriers() -> None:
    event, speakers = _event(), {"e1": "Alice"}
    thought = Thought("A question.", ("I can help.",))
    cut = CutReply("Earlier", 125, datetime(2026, 1, 1, tzinfo=UTC))
    turn = SpeakTurn(event, speakers=speakers, thought=thought, cut=cut)
    assert (turn.recent, turn.people, turn.channel_id) == ((), (), "")
    assert turn.event is event and turn.speakers is speakers
    assert turn.thought is thought and turn.cut is cut
    speakers["e1"] = "Bob"
    event.metadata["seen"] = True
    assert turn.speakers["e1"] == "Bob" and turn.event.metadata == {"seen": True}
    assert not turn.by_agent(event)
    own = dataclasses.replace(turn, channel_id="ai")
    assert own.by_agent(_event("ai")) and not own.by_agent(event)
    assert SpeakTurn(_event()).speakers == {}


def test_hook_carriers_preserve_defaults_and_the_supplied_values() -> None:
    event, decision = _event(), SpeakDecision("offer")
    report = SpeakDecisionEvent("r1", "ai", event, decision)
    assert report.event is event and report.decision is decision
    assert (report.duration_ms, report.asked_again, report.thought, report.late) == (
        0,
        False,
        None,
        False,
    )
    thought = Thought("The topic.", ("An answer.",), True)
    change = ThoughtEvent("r1", "ai", thought)
    assert change.thought is thought
    assert (change.previous, change.duration_ms) == (Thought(), None)
    assert thought.as_state() == {
        "thinking": "The topic.",
        "want_to_say": ["An answer."],
        "urgent": True,
    }
    state = thought.as_state()
    state["want_to_say"].append("Another")
    assert thought.want_to_say == ("An answer.",)
    assert thought.said() == Thought("The topic.")
    assert thought.want_to_say == ("An answer.",)


@pytest.mark.parametrize("duration", [-1, -0.25, 0.25])
def test_public_time_carriers_do_not_reject_negative_or_fractional_numbers(
    duration: float,
) -> None:
    event = SpeakDecisionEvent("", "", _event(), SpeakDecision("silent"), duration_ms=duration)
    thought = ThoughtEvent("", "", Thought(), duration_ms=duration)
    cut = CutReply("", duration, datetime(2026, 1, 1, tzinfo=UTC))
    assert event.duration_ms == thought.duration_ms == cut.played_ms == duration
    assert (event.room_id, event.channel_id, thought.room_id, thought.channel_id) == (
        "",
        "",
        "",
        "",
    )


@pytest.mark.parametrize("mode", ["speak", "offer"])
def test_only_a_silent_decision_can_be_final(mode: str) -> None:
    with pytest.raises(ValueError, match="^only a silent decision can be final$"):
        SpeakDecision(mode, final=True)  # type: ignore[arg-type]
    assert SpeakDecision("silent", final=True).final


def test_question_and_answer_maps_keep_order_and_the_supplied_aliases() -> None:
    options = {"z": "Last alphabetically, first supplied", "a": "Second supplied"}
    question = ChoiceQuestion("Which?", options)
    probabilities = {"z": 0.8, "a": 0.2}
    answer = ChoiceAnswer("z", probabilities)
    options["z"] = "Changed"
    probabilities["z"] = 0.9
    assert question.options is options and list(question.options) == ["z", "a"]
    assert question.options["z"] == "Changed"
    assert answer.probabilities is probabilities and answer.probabilities["z"] == 0.9
    assert YesNoQuestion("Ready?").yes is None and YesNoQuestion("Ready?").no is None
    assert ChoiceAnswer("z").probabilities == {}
    assert ScoreAnswer(0.5).probabilities == ()
    assert ScoreQuestion("How?", ("low", "high")).levels == ("low", "high")


def test_check_answers_preserves_extra_entries_and_typed_reads_keep_their_errors() -> None:
    questions = {"ready": YesNoQuestion("Ready?")}
    answers = Answers(ready=YesNoAnswer(0.5), extra=ScoreAnswer(2.0))
    assert check_answers(questions, answers) is answers
    assert answers.score("extra") == 2.0
    answers["ready"] = ChoiceAnswer("yes")
    with pytest.raises(TypeError, match="'ready' was answered as ChoiceAnswer, not YesNoAnswer"):
        answers.yes("ready")
    with pytest.raises(KeyError) as missing:
        answers.yes("absent")
    assert missing.value.args == ("absent",)
    with pytest.raises(ClassifierError, match="no YesNoAnswer for 'ready'"):
        check_answers(questions, answers)


async def test_mock_classifier_preserves_state_and_answers_but_copies_the_question_map() -> None:
    state, scripted = {"text": "Hi"}, YesNoAnswer(0.7)
    question = YesNoQuestion("Ready?")
    questions = {"ready": question}
    classifier = MockClassifier({"ready": scripted})
    result = await classifier.classify(state, questions)
    recorded_state, recorded_questions = classifier.calls[0]
    assert recorded_state is state and result["ready"] is scripted
    assert recorded_questions is not questions and recorded_questions["ready"] is question
    questions.clear()
    state["text"] = "Changed"
    assert list(recorded_questions) == ["ready"] and recorded_state["text"] == "Changed"


async def test_mock_classifier_uses_insertion_order_and_repeats_the_last_script() -> None:
    questions = {"pick": ChoiceQuestion("Pick", {"z": "First", "a": "Second"})}
    assert (await MockClassifier().classify([], questions)).choice("pick") == "z"
    classifier = MockClassifier([{"pick": "a"}, {"pick": "z"}])
    assert [(await classifier.classify([], questions)).choice("pick") for _ in range(3)] == [
        "a",
        "z",
        "z",
    ]


@pytest.mark.parametrize("in_sequence", [False, True])
async def test_mock_classifier_keeps_the_supplied_script_mapping_live(in_sequence: bool) -> None:
    script: dict[str, float] = {"ready": 0.25}
    questions = {"ready": YesNoQuestion("Ready?")}
    classifier = MockClassifier([script] if in_sequence else script)
    assert (await classifier.classify("first", questions)).yes("ready") == 0.25
    script["ready"] = 0.75
    assert (await classifier.classify("second", questions)).yes("ready") == 0.75
    script.clear()
    assert (await classifier.classify("third", questions)).yes("ready") == 0.0


async def test_mock_classifier_copies_the_script_sequence_but_keeps_each_mapping_live() -> None:
    first, last = {"ready": 0.25}, {"ready": 0.5}
    scripts = [first, last]
    classifier = MockClassifier(scripts)
    scripts.reverse()
    scripts.append({"ready": 1.0})
    questions = {"ready": YesNoQuestion("Ready?")}
    assert (await classifier.classify("first", questions)).yes("ready") == 0.25
    last["ready"] = 0.75
    assert (await classifier.classify("second", questions)).yes("ready") == 0.75
    scripts.clear()
    assert (await classifier.classify("third", questions)).yes("ready") == 0.75


async def test_mock_policy_records_before_failure_and_repeats_the_same_decision() -> None:
    turn, decision = SpeakTurn(_event()), SpeakDecision("offer", reason="a reason")
    policy = MockSpeakPolicy([decision])
    assert await policy.decide(turn) is decision
    assert await policy.decide(turn) is decision
    assert policy.turns == [turn, turn] and policy.turns[0] is turn
    error = RuntimeError("policy unavailable")
    failed = MockSpeakPolicy(error=error)
    with pytest.raises(RuntimeError) as raised:
        await failed.decide(turn)
    assert raised.value is error and failed.turns == [turn]
    assert (await MockSpeakPolicy([]).decide(turn)).mode == "speak"


async def test_mock_thinker_records_live_inputs_and_keeps_or_repeats_the_thought() -> None:
    context, previous = AIContext(messages=[]), Thought("Previous")
    thinker = MockThinker()
    assert await thinker.think(previous, context) is previous
    assert thinker.calls[0][0] is previous and thinker.calls[0][1] is context
    next_thought = Thought("Next")
    scripted = MockThinker([next_thought])
    assert await scripted.think(previous, context) is next_thought
    assert await scripted.think(previous, context) is next_thought
    error = RuntimeError("thinker unavailable")
    failed = MockThinker(error=error)
    with pytest.raises(RuntimeError) as raised:
        await failed.think(previous, context)
    assert raised.value is error and failed.calls == [(previous, context)]


async def test_protocol_optional_hooks_and_answer_only_delegation() -> None:
    forgotten: list[str] = []
    closed: list[bool] = []

    class Policy(AlwaysSpeak):
        def forget_room(self, room_id: str) -> None:
            forgotten.append(room_id)

        async def close(self) -> None:
            closed.append(True)

    wrapped = AnswerOnly(Policy(), ["Alice"])
    wrapped.forget_room("r1")
    await wrapped.close()
    assert forgotten == ["r1"] and closed == [True]
    assert SpeakPolicy.forget_room(wrapped, "r2") is None
    assert await SpeakPolicy.close(wrapped) is None
    assert await Thinker.close(MockThinker()) is None
    assert await Classifier.close(MockClassifier()) is None


async def test_answer_only_preserves_the_wrapped_lifecycle_errors() -> None:
    forgotten = OSError("forget failed")
    closed = RuntimeError("close failed")

    class Policy(AlwaysSpeak):
        def forget_room(self, room_id: str) -> None:
            raise forgotten

        async def close(self) -> None:
            raise closed

    wrapped = AnswerOnly(Policy(), ["Alice"])
    with pytest.raises(OSError) as first:
        wrapped.forget_room("r1")
    with pytest.raises(RuntimeError) as second:
        await wrapped.close()
    assert first.value is forgotten and second.value is closed


async def test_classifier_decision_override_is_used_by_decide_and_can_call_the_standard_rule() -> (
    None
):
    classifier = MockClassifier({"directness": 0.0, "hush": 0.0})
    composed: list[SpeakDecision] = []

    class Policy(ClassifierSpeakPolicy):
        def decision(self, turn: SpeakTurn, answers: Answers) -> SpeakDecision:
            standard = super().decision(turn, answers)
            composed.append(standard)
            return dataclasses.replace(standard, mode="speak", reason="application decision")

    policy = Policy(classifier, agent_name="Nova")
    result = await policy.decide(SpeakTurn(_event()))
    assert len(classifier.calls) == len(composed) == 1
    assert (composed[0].mode, composed[0].reason) == ("silent", "not addressed")
    assert (result.mode, result.reason) == ("speak", "application decision")
