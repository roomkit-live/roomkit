"""Staying quiet when asked (RMK-641, RFC §6.4): a state of the room the classifier
policy keeps, read by the classifier with every turn, and the final silences that
let the channel think without waiting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from roomkit import (
    ClassifierSpeakPolicy,
    MockClassifier,
    MockThinker,
    SpeakDecisionEvent,
    SpeakTurn,
    Thought,
    YesNoQuestion,
)
from roomkit.channels.ai import AIChannel
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
    assert state["agent"]["listening_only"] == {"asked": STAY_QUIET}
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


def test_the_request_kept_is_bounded() -> None:
    rooms = ListeningRooms()
    rooms.start("r1", "x" * (ASKED_LIMIT * 3))
    listening = rooms.of("r1")
    assert listening is not None and len(listening.asked) == ASKED_LIMIT
    rooms.stop("r1")
    assert rooms.of("r1") is None


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
