"""AnswerOnly: an agent that listens to everyone and answers only some people
(RMK-625, RFC §6.4)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import AlwaysSpeak, AnswerOnly, MockSpeakPolicy, MockThinker, SpeakTurn, Thought
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.answer_only import LISTENED_TO
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_BINDING = ChannelBinding(
    channel_id="ai1",
    room_id="r1",
    channel_type=ChannelType.AI,
    category=ChannelCategory.INTELLIGENCE,
)
_VOICE = ChannelBinding(
    channel_id="voice1",
    room_id="r1",
    channel_type=ChannelType.VOICE,
    category=ChannelCategory.TRANSPORT,
)
PRICE = Thought("They talk about the price; I know it.", ("It costs 1,200 $ a year.",))


def _said(body: str, speaker: str | None) -> Any:
    metadata = {"sender_name": speaker} if speaker is not None else {}
    return make_event(body=body, channel_id="voice1", room_id="r1", metadata=metadata)


def _turn(event: Any, people: tuple[str, ...] = ("Sylvain", "TV")) -> SpeakTurn:
    name = event.metadata.get("sender_name")
    speakers = {event.id: name} if name else {}
    return SpeakTurn(event=event, people=people, channel_id="ai1", speakers=speakers)


async def test_a_speaker_not_answered_is_only_listened_to_without_asking_the_policy() -> None:
    inner = MockSpeakPolicy(["speak"])

    decision = await AnswerOnly(inner, ["Sylvain"]).decide(
        _turn(_said("And you, what do you think?", "TV"))
    )

    assert (decision.mode, decision.reason) == ("silent", LISTENED_TO)
    assert inner.turns == []


async def test_a_speaker_answered_is_the_wrapped_policys_with_only_the_people_answered() -> None:
    inner = MockSpeakPolicy(["offer"])
    event = _said("Did you hear that?", "Sylvain")

    decision = await AnswerOnly(inner, ["Sylvain"]).decide(_turn(event))

    assert decision.mode == "offer"
    [turn] = inner.turns
    assert turn.event is event
    # The television is no one the agent talks with: Sylvain alone is a one-to-one.
    assert turn.people == ("Sylvain",)


@pytest.mark.parametrize("speaker", [None, ""], ids=["unnamed", "empty"])
async def test_a_speaker_the_room_does_not_name_is_not_answered(speaker: str | None) -> None:
    inner = MockSpeakPolicy(["speak"])

    decision = await AnswerOnly(inner, ["Sylvain"]).decide(_turn(_said("Hello?", speaker)))

    assert decision.reason == LISTENED_TO
    assert inner.turns == []


@pytest.mark.parametrize("configured", ["sylvain", "  SYLVAIN ", "Sylvain:"])
async def test_names_compare_as_the_room_gives_them(configured: str) -> None:
    policy = AnswerOnly(AlwaysSpeak(), [configured])

    decision = await policy.decide(_turn(_said("Nova?", "Sylvain")))

    assert decision.mode == "speak"
    assert policy.answers("sylvain") and not policy.answers("Paul")


@pytest.mark.parametrize("people", [[], ["", "  "], [":"]])
def test_answering_no_one_is_refused(people: list[str]) -> None:
    with pytest.raises(ValueError, match="at least one person"):
        AnswerOnly(AlwaysSpeak(), people)


async def test_close_closes_the_wrapped_policy() -> None:
    closed: list[bool] = []

    class _Policy(AlwaysSpeak):
        async def close(self) -> None:
            closed.append(True)

    await AnswerOnly(_Policy(), ["Sylvain"]).close()

    assert closed == [True]


# --- on an AI channel -------------------------------------------------------------------


def _context(*events: Any) -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[_BINDING, _VOICE], recent_events=list(events))


async def test_a_voice_only_listened_to_is_thought_about_and_never_answered() -> None:
    """The agent hears the television and thinks about it; with something to say,
    it is asked again and stays silent; Sylvain then gets the thought."""
    inner = MockSpeakPolicy(["speak"])
    thinker = MockThinker([PRICE])
    provider = MockAIProvider(responses=["It costs 1,200 $ a year."])
    channel = AIChannel(
        "ai1",
        provider=provider,
        speak_policy=AnswerOnly(inner, ["Sylvain"]),
        thinker=thinker,
        think_wait=1.0,
    )
    tv = _said("So how much does this licence cost, do you think?", "TV")

    output = await channel.on_event(tv, _BINDING, _context(tv))

    assert output.responded is False
    assert provider.calls == []
    assert len(thinker.calls) == 1
    assert inner.turns == []  # asked again with the thought, still never the TV's
    sylvain = _said("Nova, did you catch that?", "Sylvain")
    run = await respond(channel, sylvain, _BINDING, _context(tv, sylvain))

    assert run.text == "It costs 1,200 $ a year."
    [turn] = inner.turns
    assert turn.event is sylvain and turn.thought == PRICE
