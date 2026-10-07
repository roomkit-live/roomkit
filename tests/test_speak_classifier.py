"""A speak policy on a classifier: narrow judgments composed in code (RMK-561,
RFC §6.4), and the turn it reads: who said what, who takes part."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import (
    ChoiceAnswer,
    ClassifierError,
    ClassifierSpeakPolicy,
    MockClassifier,
    ScoreQuestion,
    SpeakTurn,
    YesNoQuestion,
)
from roomkit.channels._ai_speaking import _speak_turn
from roomkit.channels.ai import AIChannel
from roomkit.classifiers.base import Answers, ScoreAnswer, YesNoAnswer
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, ParticipantRole
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.classifier import QUESTIONS, addressed, compose, judgments_of
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_BINDING = ChannelBinding(
    channel_id="ai1",
    room_id="r1",
    channel_type=ChannelType.AI,
    category=ChannelCategory.INTELLIGENCE,
)
_SMS = ChannelBinding(
    channel_id="sms1",
    room_id="r1",
    channel_type=ChannelType.SMS,
    category=ChannelCategory.TRANSPORT,
)


def _person(pid: str, name: str, **kwargs: Any) -> Participant:
    return Participant(id=pid, room_id="r1", channel_id="sms1", display_name=name, **kwargs)


def _said(body: str, pid: str | None = "p1", channel_id: str = "sms1", **kwargs: Any) -> Any:
    return make_event(body=body, channel_id=channel_id, participant_id=pid, room_id="r1", **kwargs)


# -- compose: the judgments, in order --------------------------------------------


@pytest.mark.parametrize(
    ("judgments", "alone", "expected"),
    [
        ({"directness": 3.0, "unfinished": 0.9}, False, ("silent", "not finished")),
        ({"directness": 3.0, "deferred": 0.8}, False, ("silent", "postponed")),
        ({"directness": 3.0, "hush": 0.7}, False, ("silent", "asked to keep quiet")),
        (
            {"quiet_rule": 0.9, "request": 0.1, "directness": 2.0},
            False,
            ("silent", "keeping quiet"),
        ),
        ({"quiet_rule": 0.9, "request": 0.8, "directness": 3.0}, False, ("speak", "addressed")),
        ({"answered": 0.8}, False, ("speak", "answers its question")),
        ({"directness": 1.5}, False, ("speak", "addressed")),
        ({"directness": 0.8, "request": 0.7}, False, ("speak", "addressed")),
        ({"directness": 0.2, "request": 0.9}, True, ("speak", "addressed")),
        ({"directness": 0.2, "request": 0.9}, False, ("silent", "not addressed")),
        ({"directness": 0.9, "request": 0.2}, False, ("offer", "wondered about")),
        ({"directness": 0.6}, False, ("silent", "not addressed")),
        ({}, False, ("silent", "not addressed")),
    ],
)
def test_compose(judgments: dict[str, float], alone: bool, expected: tuple[str, str]) -> None:
    assert compose(judgments, alone=alone) == expected


def test_addressed_bands() -> None:
    assert addressed(1.5, 0.0)
    assert not addressed(1.4, 0.4)
    assert addressed(0.75, 0.5)
    assert addressed(0.0, 0.5, alone=True)
    assert not addressed(0.0, 0.5)


def test_judgments_of_reads_every_kind() -> None:
    answers = Answers(
        hush=YesNoAnswer(0.2),
        directness=ScoreAnswer(2.4),
        language=ChoiceAnswer("French", {"French": 0.97, "other": 0.03}),
    )
    assert judgments_of(answers) == {"hush": 0.2, "directness": 2.4, "language=French": 0.97}


# -- the policy -------------------------------------------------------------------


def _turn(*recent: Any, event: Any, people: tuple[str, ...] = ("Sylvain", "Paul")) -> SpeakTurn:
    speakers = {e.id: n for e, n in [*recent, (event, "Sylvain")] if n}
    return SpeakTurn(
        event=event,
        recent=tuple(e for e, _ in recent),
        people=people,
        channel_id="ai1",
        speakers=speakers,
    )


async def test_the_classifier_reads_the_turns_by_speaker_the_agent_under_its_name() -> None:
    classifier = MockClassifier({"directness": 3.0})
    policy = ClassifierSpeakPolicy(classifier, agent_name="Nova", agent_role="meeting assistant")
    question = _said("Nova, can you sum up?")
    turn = _turn(
        (_said("We are talking about the budget", pid="p2"), "Paul"),
        (_said("Shall I sum up?", pid=None, channel_id="ai1"), ""),
        (_said("A picture", pid="p3"), ""),
        event=question,
    )

    decision = await policy.decide(turn)

    assert (decision.mode, decision.reason) == ("speak", "addressed")
    state, questions = classifier.calls[0]
    assert state == {
        "agent": {"name": "Nova", "role": "meeting assistant"},
        "people": ["Sylvain", "Paul"],
        "recent_turns": [
            {"speaker": "Paul", "text": "We are talking about the budget"},
            {"speaker": "Nova", "text": "Shall I sum up?"},
            {"speaker": "someone", "text": "A picture"},
        ],
        "last_turn": {"speaker": "Sylvain", "text": "Nova, can you sum up?"},
    }
    assert set(questions) == set(QUESTIONS)
    assert decision.judgments["directness"] == 3.0
    assert decision.notes == ()


async def test_history_bounds_the_turns_read() -> None:
    classifier = MockClassifier()
    policy = ClassifierSpeakPolicy(classifier, agent_name="Nova", history=2)
    earlier = [(_said(f"turn {i}"), "Sylvain") for i in range(5)]
    await policy.decide(_turn(*earlier, event=_said("and then")))
    state, _ = classifier.calls[0]
    assert [t["text"] for t in state["recent_turns"]] == ["turn 3", "turn 4"]
    assert "role" not in state["agent"]


async def test_one_person_makes_any_request_the_agents() -> None:
    classifier = MockClassifier({"directness": 0.3, "request": 0.9})
    policy = ClassifierSpeakPolicy(classifier, agent_name="Nova")
    decision = await policy.decide(
        _turn(event=_said("Quelle heure est-il ?"), people=("Sylvain",))
    )
    assert decision.mode == "speak"


async def test_the_language_is_judged_and_named_in_the_notes() -> None:
    languages = {"English": "Answer in English only.", "German": "Answer in German only."}
    classifier = MockClassifier({"directness": 3.0, "language": "German"})
    policy = ClassifierSpeakPolicy(classifier, agent_name="Nova", languages=languages)

    decision = await policy.decide(_turn(event=_said("Nova ?")))

    _, questions = classifier.calls[0]
    assert set(questions["language"].options) == {"English", "German", "other"}  # type: ignore[union-attr]
    assert decision.notes == ("Answer in German only.",)
    assert decision.judgments["language=German"] == 1.0


async def test_another_language_adds_no_note() -> None:
    classifier = MockClassifier({"directness": 3.0, "language": "other"})
    policy = ClassifierSpeakPolicy(
        classifier, agent_name="Nova", languages={"English": "Answer in English only."}
    )
    assert (await policy.decide(_turn(event=_said("Nova ?")))).notes == ()


def test_other_is_not_a_language_name() -> None:
    with pytest.raises(ValueError, match="'other'"):
        ClassifierSpeakPolicy(MockClassifier(), agent_name="Nova", languages={"other": "x"})


async def test_a_question_is_replaced_by_name_and_compose_overridden() -> None:
    hush = YesNoQuestion("Does `last_turn` say 'shh' to the assistant?")
    urgency = ScoreQuestion("How urgent?", ("later", "now"))
    classifier = MockClassifier({"urgency": 1.0})

    class Urgent(ClassifierSpeakPolicy):
        def decision(self, turn: SpeakTurn, answers: Answers) -> Any:
            decision = super().decision(turn, answers)
            if answers.score("urgency") >= 0.5:
                return type(decision)("speak", "urgent", decision.judgments)
            return decision

    policy = Urgent(classifier, agent_name="Nova", questions={"hush": hush, "urgency": urgency})
    decision = await policy.decide(_turn(event=_said("The building is on fire")))

    _, questions = classifier.calls[0]
    assert questions["hush"] is hush
    assert "urgency" in questions
    assert (decision.mode, decision.reason) == ("speak", "urgent")


async def test_an_event_without_text_is_not_judged() -> None:
    classifier = MockClassifier()
    policy = ClassifierSpeakPolicy(classifier, agent_name="Nova")
    empty = _said("")
    decision = await policy.decide(_turn(event=empty))
    assert (decision.mode, decision.reason) == ("speak", "nothing to judge")
    assert classifier.calls == []


# -- on the AI channel --------------------------------------------------------------


def _channel(classifier: MockClassifier) -> tuple[AIChannel, MockAIProvider]:
    provider = MockAIProvider(responses=["Yes?"])
    policy = ClassifierSpeakPolicy(classifier, agent_name="Nova")
    return AIChannel("ai1", provider=provider, speak_policy=policy), provider


def _context(*events: Any, participants: list[Participant] | None = None) -> RoomContext:
    people = participants or [_person("p1", "Sylvain"), _person("p2", "Paul")]
    return RoomContext(
        room=Room(id="r1"),
        bindings=[_BINDING, _SMS],
        participants=people,
        recent_events=list(events),
    )


async def test_on_the_channel_a_turn_for_someone_else_runs_nothing() -> None:
    channel, provider = _channel(MockClassifier({"directness": 0.1}))
    event = _said("Paul, do you have the numbers?")
    output = await channel.on_event(event, _BINDING, _context(event))
    assert output.responded is False
    assert provider.calls == []


async def test_on_the_channel_a_failing_classifier_lets_the_agent_speak() -> None:
    channel, provider = _channel(MockClassifier(error=ClassifierError("down")))
    event = _said("Nova ?")
    run = await respond(channel, event, _BINDING, _context(event))
    assert run.text == "Yes?"


# -- the turn the channel builds ------------------------------------------------------


def test_the_turn_names_speakers_and_marks_the_agents_answers() -> None:
    before = _said("Hello", pid="p2")
    answer = _said("Hello Paul", pid=None, channel_id="ai1")
    stamped = _said("Et moi", pid="p1", metadata={"sender_name": "Marie"})
    event = _said("Nova ?")
    turn = _speak_turn(event, _context(before, answer, stamped, event), "ai1")

    assert turn.channel_id == "ai1"
    assert turn.speakers == {before.id: "Paul", stamped.id: "Marie", event.id: "Sylvain"}
    assert turn.by_agent(answer) and not turn.by_agent(before)


def test_people_leave_out_agents_bots_and_the_agents_channel() -> None:
    participants = [
        _person("p1", "Sylvain"),
        _person("p2", "Paul", status="left"),
        _person("a1", "Codex", role=ParticipantRole.AGENT),
        _person("b1", "Notifier", role=ParticipantRole.BOT),
        Participant(id="n1", room_id="r1", channel_id="ai1", display_name="Nova"),
    ]
    event = _said("Nova ?")
    assert _speak_turn(event, _context(event, participants=participants), "ai1").people == (
        "Sylvain",
    )


def test_people_count_the_voices_of_one_microphone() -> None:
    voices = [
        _said("Hi", metadata={"sender_name": "Speaker 1"}),
        _said("Hello to you", metadata={"sender_name": "Speaker 2"}),
    ]
    event = _said("Nova ?", metadata={"sender_name": "Speaker 1"})
    turn = _speak_turn(event, _context(*voices, event, participants=[_person("p1", "Mic")]), "ai1")
    assert turn.people == ("Speaker 1", "Speaker 2")


def test_the_turn_holds_only_what_the_channel_may_know() -> None:
    """RFC §7.5 rule 8: an event withheld from the AI channel at delivery does not
    reach its policy, which may send the turn to a classifier outside."""
    shown = _said("Le budget est de 40 000 $")
    to_sms_only = _said("The access code is 4417", visibility="sms1")
    internal = _said("note interne", visibility="internal")
    own = _said("Noted.", pid=None, channel_id="ai1", visibility="sms1")
    event = _said("Nova ?")

    turn = _speak_turn(event, _context(shown, to_sms_only, internal, own, event), "ai1")

    assert [e.id for e in turn.recent] == [shown.id, own.id]
