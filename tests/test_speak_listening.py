"""Staying quiet when asked (RMK-641, RFC §6.4): a state of the room the classifier
policy keeps, read by the classifier with every turn, and the final silences that
let the channel think without waiting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from roomkit import (
    AnswerOnly,
    ClassifierError,
    ClassifierSpeakPolicy,
    MockClassifier,
    MockThinker,
    SpeakDecision,
    SpeakDecisionEvent,
    SpeakTurn,
    Thought,
    YesNoQuestion,
)
from roomkit.channels.ai import AIChannel
from roomkit.classifiers.base import Answers, ScoreAnswer, YesNoAnswer
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.base import CutReply
from roomkit.speaking.listening import ASKED_LIMIT, LISTEN_REQUEST, ListeningRooms
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
STAY_QUIET = "You can stay in listening mode for now."
ASIDE = {"directness": 0.0, "asked_me": 0.1}
ASKED = {"directness": 3.0, "asked_me": 0.9}
LIFTED = {"directness": 3.0, "lift": 0.9, "request": 0.9}


def _said(body: str, room_id: str = "r1", name: str = "Sylvain") -> Any:
    return make_event(
        body=body, channel_id="sms1", room_id=room_id, metadata={"sender_name": name}
    )


def _turn(body: str, room_id: str = "r1", **kwargs: Any) -> SpeakTurn:
    event = _said(body, room_id)
    return SpeakTurn(event=event, people=("Sylvain",), speakers={event.id: "Sylvain"}, **kwargs)


def _policy(*scripts: dict[str, float], **kwargs: Any) -> tuple[ClassifierSpeakPolicy, Any]:
    classifier = MockClassifier([{"listen_request": 0.9}, *scripts])
    return ClassifierSpeakPolicy(classifier, agent_name="Nova", **kwargs), classifier


# -- the policy -------------------------------------------------------------------


async def test_a_request_to_only_listen_starts_the_state_with_a_final_silence() -> None:
    policy, classifier = _policy(ASIDE)

    decision = await policy.decide(_turn(STAY_QUIET))

    assert (decision.mode, decision.reason, decision.final) == ("silent", "asked to listen", True)
    [(state, questions)] = classifier.calls
    assert "listen_request" in questions and "asked_me" not in questions
    assert "listening_only" not in state["agent"]

    await policy.decide(_turn("My tablet does not seem to work today."))

    state, questions = classifier.calls[1]
    assert state["agent"]["listening_only"] == {"speaker": "Sylvain", "text": STAY_QUIET}
    assert {"asked_me", "lift"} <= set(questions) and "listen_request" not in questions


async def test_while_listening_an_aside_is_a_final_silence() -> None:
    """One person talking with the agent does not make every request its own:
    "what is the base URL?" said aside reads as a request without the address."""
    policy, _ = _policy({"directness": 0.4, "request": 0.9, "asked_me": 0.3})
    await policy.decide(_turn(STAY_QUIET))

    for body in ("Funny, my jacket is here.", "What is the base URL?"):
        decision = await policy.decide(_turn(body))
        assert (decision.mode, decision.reason, decision.final) == ("silent", "listening", True)
        assert decision.judgments["listening"] == 1.0


async def test_a_question_put_to_the_agent_is_answered_and_the_room_goes_on_listening() -> None:
    policy, classifier = _policy(ASKED, ASIDE, languages={"French": "Réponds en français."})
    await policy.decide(_turn(STAY_QUIET))

    asked = await policy.decide(_turn("What do you think, Nova?"))
    after = await policy.decide(_turn("Right, I have to tidy my desk."))

    assert (asked.mode, asked.reason, asked.final) == ("speak", "asked while listening", False)
    assert asked.notes == ("Réponds en français.",)
    assert (after.mode, after.reason) == ("silent", "listening")
    assert "listening_only" in classifier.calls[-1][0]["agent"]


async def test_a_question_without_the_address_is_not_answered_while_listening() -> None:
    policy, _ = _policy({"directness": 1.0, "asked_me": 0.9})
    await policy.decide(_turn(STAY_QUIET))

    decision = await policy.decide(_turn("What are we talking about?"))

    assert (decision.mode, decision.reason) == ("silent", "listening")


async def test_a_turn_that_lets_the_agent_talk_again_opens_the_room() -> None:
    policy, classifier = _policy(LIFTED, {"directness": 0.0})
    await policy.decide(_turn(STAY_QUIET))

    lifted = await policy.decide(_turn("You can talk again, Nova: what time is it?"))
    await policy.decide(_turn("Thanks."))

    assert (lifted.mode, lifted.reason, lifted.final) == ("speak", "addressed", False)
    assert "listening" not in lifted.judgments
    state, questions = classifier.calls[-1]
    assert "listening_only" not in state["agent"] and "listen_request" in questions


async def test_the_state_is_per_room() -> None:
    policy, _ = _policy({"directness": 3.0, "request": 0.9})
    await policy.decide(_turn(STAY_QUIET, room_id="r1"))

    other = await policy.decide(_turn("Nova, what time is it?", room_id="r2"))

    assert (other.mode, other.reason) == ("speak", "addressed")


async def test_a_listening_room_resumes_no_cut_answer() -> None:
    policy, _ = _policy({"resume": 0.9})
    await policy.decide(_turn(STAY_QUIET))

    decision = await policy.decide(
        _turn("Okay.", cut=CutReply("It will rain", 800, datetime.now(UTC)))
    )

    assert (decision.mode, decision.reason) == ("silent", "listening")
    assert not decision.notes


async def test_a_listening_question_is_replaced_by_name() -> None:
    mine = YesNoQuestion("Does `last_turn` say the word 'pineapple'?")
    policy, classifier = _policy(questions={"listen_request": mine})

    await policy.decide(_turn("Pineapple."))

    [(_, questions)] = classifier.calls
    assert (
        questions["listen_request"] is mine and questions["listen_request"] is not LISTEN_REQUEST
    )


def test_the_request_kept_is_bounded_at_a_word() -> None:
    rooms = ListeningRooms()
    rooms.start("r1", "Sylvain", "stay quiet " * ASKED_LIMIT)
    listening = rooms.of("r1")
    assert listening is not None and len(listening.text) <= ASKED_LIMIT
    assert listening.text.endswith(("quiet…", "stay…"))
    rooms.stop("r1")
    assert rooms.of("r1") is None


@pytest.mark.parametrize("held", ["unfinished", "deferred", "hush"])
async def test_a_question_the_speaker_holds_back_is_not_answered_while_listening(
    held: str,
) -> None:
    """ "Nova, can you tell me... no wait, later": a listening room is never more
    eager than an open one."""
    policy, _ = _policy({**ASKED, held: 0.9})
    await policy.decide(_turn(STAY_QUIET))

    decision = await policy.decide(_turn("Nova, can you tell me... no wait, later."))

    assert (decision.mode, decision.reason, decision.final) == ("silent", "listening", True)


async def test_a_turn_a_listening_room_cannot_judge_stays_silent() -> None:
    """No text, or a classifier that fails: the agent was asked to only listen."""
    classifier = MockClassifier({"listen_request": 0.9})
    policy = ClassifierSpeakPolicy(classifier, agent_name="Nova")
    await policy.decide(_turn(STAY_QUIET))
    image = make_event(body="", channel_id="sms1", room_id="r1")

    no_text = await policy.decide(SpeakTurn(event=image, people=("Sylvain",)))
    classifier._error = ClassifierError("down")
    failed = await policy.decide(_turn("Nova, what time is it?"))

    for decision in (no_text, failed):
        assert (decision.mode, decision.reason, decision.final) == ("silent", "listening", True)


async def test_an_open_room_still_falls_back_on_a_failing_classifier() -> None:
    policy = ClassifierSpeakPolicy(
        MockClassifier(error=ClassifierError("down")), agent_name="Nova"
    )
    with pytest.raises(ClassifierError):
        await policy.decide(_turn("Nova, what time is it?"))


async def test_a_turn_is_decided_as_it_was_asked() -> None:
    """Another turn may start the listening state during the classifier call: a
    turn judged open is decided open, not dropped as a listening silence."""
    policy, _ = _policy()
    turn = _turn("Nova, what time is it?")
    asked_open = Answers(
        directness=ScoreAnswer(3.0), request=YesNoAnswer(0.9), listen_request=YesNoAnswer(0.0)
    )
    policy._rooms.start("r1", "Paul", STAY_QUIET)  # meanwhile, Paul asked for quiet

    decision = policy.decision(turn, asked_open)

    assert (decision.mode, decision.reason) == ("speak", "addressed")
    assert policy._rooms.of("r1") is not None  # Paul's request stands for the next turn


async def test_asked_me_and_lift_are_replaced_by_name() -> None:
    mine = YesNoQuestion("Does `last_turn` say 'pineapple'?")
    policy, classifier = _policy(questions={"asked_me": mine, "lift": mine})
    await policy.decide(_turn(STAY_QUIET))

    await policy.decide(_turn("Pineapple."))

    _, questions = classifier.calls[-1]
    assert questions["asked_me"] is mine and questions["lift"] is mine


async def test_a_voice_only_listened_to_neither_starts_nor_lifts_the_state() -> None:
    classifier = MockClassifier({"listen_request": 0.9, "lift": 0.9})
    inner = ClassifierSpeakPolicy(classifier, agent_name="Nova")
    policy = AnswerOnly(inner, ["Sylvain"])
    tv = make_event(body="Stay quiet, everyone.", channel_id="sms1", room_id="r1")

    decision = await policy.decide(
        SpeakTurn(event=tv, people=("Sylvain",), speakers={tv.id: "TV"})
    )

    assert decision.final and classifier.calls == []
    assert inner._rooms.of("r1") is None


def test_only_a_silence_is_final() -> None:
    with pytest.raises(ValueError, match="only a silent decision"):
        SpeakDecision("speak", final=True)


# -- through the channel --------------------------------------------------------------


def _context(*events: Any) -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[_BINDING, _SMS], recent_events=list(events))


def _channel(policy: Any, thinker: MockThinker) -> tuple[AIChannel, list[SpeakDecisionEvent]]:
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["I would start with the printer."]),
        system_prompt="You are Nova.",
        speak_policy=policy,
        thinker=thinker,
        think_wait=5.0,
    )
    decisions: list[SpeakDecisionEvent] = []

    async def hook(event: SpeakDecisionEvent) -> None:
        decisions.append(event)

    channel._speak_decision_hook = hook
    return channel, decisions


async def test_the_channel_forgets_the_state_of_a_room_it_leaves_or_joins() -> None:
    """A room that reuses an id inherits neither the thought nor the request to
    only listen, nor the words of whoever made it (RFC §6.4)."""
    classifier = MockClassifier({"listen_request": 0.9})
    inner = ClassifierSpeakPolicy(classifier, agent_name="Nova")
    channel, _ = _channel(AnswerOnly(inner, ["Sylvain"]), MockThinker([Thought()]))

    for forget in (channel.on_room_detached, lambda r: channel.on_room_attached(r, _BINDING)):
        await inner.decide(_turn(STAY_QUIET))
        assert inner._rooms.of("r1") is not None
        await forget("r1")
        assert inner._rooms.of("r1") is None


async def test_a_listening_room_thinks_without_holding_the_turn_back() -> None:
    """A final silence is thought about, but the channel neither waits for the
    thought (think_wait is 5 s here) nor asks the policy again with it."""
    policy, classifier = _policy({"directness": 0.0}, ASKED)
    thinker = MockThinker([Thought("Sylvain lists what is broken.", ("Check the printer.",))])
    channel, decisions = _channel(policy, thinker)
    request, aside = _said(STAY_QUIET), _said("My printer does not work either.")

    for event in (request, aside):
        output = await channel.on_event(event, _BINDING, _context(event))
        assert output.responded is False

    assert [(d.decision.reason, d.asked_again) for d in decisions] == [
        ("asked to listen", False),
        ("listening", False),
    ]
    [mind] = channel._minds.values()
    assert await mind.settled(2, 1.0)
    question = _said("What do you think, Nova?")
    run = await respond(channel, question, _BINDING, _context(request, aside, question))

    assert run.text == "I would start with the printer."
    assert decisions[-1].decision.reason == "asked while listening"
    assert len(classifier.calls) == 3
