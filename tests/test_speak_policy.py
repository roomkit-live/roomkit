"""Speaking turns: an AI channel's speak policy decides whether the agent speaks,
offers or stays silent on an event (RMK-560, RFC §6.4)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from roomkit import (
    AlwaysSpeak,
    HookExecution,
    HookTrigger,
    MockSpeakPolicy,
    RoomKit,
    SpeakDecision,
    SpeakDecisionEvent,
)
from roomkit.channels._ai_speaking import OFFER_NOTE
from roomkit.channels.ai import AIChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import TextContent
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
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


def _channel(policy: Any, **kwargs: Any) -> tuple[AIChannel, MockAIProvider]:
    provider = MockAIProvider(responses=["Yes?"])
    return AIChannel("ai1", provider=provider, speak_policy=policy, **kwargs), provider


def _context(*events: Any, people: tuple[str, ...] = ("Sylvain",)) -> RoomContext:
    participants = [
        Participant(id=f"p{i}", room_id="r1", channel_id="sms1", display_name=n)
        for i, n in enumerate(people)
    ]
    return RoomContext(
        room=Room(id="r1"),
        bindings=[_BINDING, _SMS],
        participants=participants,
        recent_events=list(events),
    )


def _last_input(provider: MockAIProvider) -> str:
    message = provider.calls[-1].messages[-1]
    return message.content if isinstance(message.content, str) else str(message.content)


class _Memory(SlidingWindowMemory):
    def __init__(self) -> None:
        super().__init__()
        self.ingested: list[str] = []

    async def ingest(self, room_id: str, event: Any, *, channel_id: str | None = None) -> None:
        self.ingested.append(event.id)


async def test_silent_runs_no_turn_but_the_memory_learns_the_event() -> None:
    memory = _Memory()
    channel, provider = _channel(MockSpeakPolicy(["silent"]), memory=memory)
    event = make_event(body="Paul, do you have the numbers?", channel_id="sms1", room_id="r1")

    output = await channel.on_event(event, _BINDING, _context())

    assert output.responded is False
    assert provider.calls == []
    assert memory.ingested == [event.id]


async def test_speak_runs_the_turn_with_the_decision_notes() -> None:
    decision = SpeakDecision("speak", notes=("Answer in French only.",))
    channel, provider = _channel(MockSpeakPolicy([decision]))

    run = await respond(
        channel, make_event(body="Nova ?", channel_id="sms1", room_id="r1"), _BINDING, _context()
    )

    assert run.text == "Yes?"
    assert "Answer in French only." in _last_input(provider)


async def test_offer_asks_to_offer_rather_than_answer() -> None:
    channel, provider = _channel(MockSpeakPolicy(["offer"]))

    await respond(
        channel,
        make_event(body="We are looking for the figure.", channel_id="sms1", room_id="r1"),
        _BINDING,
        _context(),
    )

    assert OFFER_NOTE in _last_input(provider)


async def test_the_policy_reads_the_conversation_before_the_event_and_who_is_there() -> None:
    policy = MockSpeakPolicy(["speak"])
    channel, _ = _channel(policy)
    before = make_event(body="Hello", channel_id="sms1", room_id="r1")
    event = make_event(body="Nova, a question", channel_id="sms1", room_id="r1")

    await respond(channel, event, _BINDING, _context(before, event, people=("Sylvain", "Paul")))

    [turn] = policy.turns
    assert turn.event is event
    assert [e.id for e in turn.recent] == [before.id]
    assert turn.people == ("Sylvain", "Paul")


async def test_an_instruction_is_not_submitted_to_the_policy() -> None:
    policy = MockSpeakPolicy(["silent"])
    channel, provider = _channel(policy)
    instruction = make_event(
        body="Give the task's result.", channel_id="sms1", room_id="r1", type=EventType.INSTRUCTION
    )

    run = await respond(channel, instruction, _BINDING, _context())

    assert policy.turns == []
    assert run.text == "Yes?"


@pytest.mark.parametrize(
    "policy", [MockSpeakPolicy(error=RuntimeError("down")), MockSpeakPolicy(["silent"], delay=1.0)]
)
async def test_a_policy_that_fails_or_is_late_does_not_silence_the_agent(
    policy: MockSpeakPolicy, caplog: pytest.LogCaptureFixture
) -> None:
    channel, provider = _channel(policy, speak_timeout=0.1)
    decisions: list[SpeakDecisionEvent] = []

    async def hook(event: SpeakDecisionEvent) -> None:
        decisions.append(event)

    channel._speak_decision_hook = hook
    with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
        run = await respond(
            channel,
            make_event(body="Nova ?", channel_id="sms1", room_id="r1"),
            _BINDING,
            _context(),
        )

    assert run.text == "Yes?"
    assert decisions[0].decision.reason == "fallback"
    assert "speaking" in caplog.text


@pytest.mark.parametrize(
    ("policy", "bound", "at_least_ms"),
    [
        (MockSpeakPolicy(["speak"], delay=0.05), 2.0, 40),
        (MockSpeakPolicy(["silent"], delay=1.0), 0.1, 90),
    ],
    ids=["decided", "late"],
)
async def test_a_decision_reports_how_long_the_policy_took(
    policy: MockSpeakPolicy, bound: float, at_least_ms: int
) -> None:
    """RMK-627: a hook measures the policy without wrapping it; a late one
    reports the bound, not the time it would have taken."""
    channel, _ = _channel(policy, speak_timeout=bound)
    decisions: list[SpeakDecisionEvent] = []

    async def hook(event: SpeakDecisionEvent) -> None:
        decisions.append(event)

    channel._speak_decision_hook = hook
    await respond(
        channel, make_event(body="Nova ?", channel_id="sms1", room_id="r1"), _BINDING, _context()
    )

    [event] = decisions
    assert at_least_ms <= event.duration_ms < 900
    assert event.asked_again is False


def test_a_decision_refuses_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="maybe"):
        SpeakDecision("maybe")  # type: ignore[arg-type]


def test_the_bound_must_be_positive() -> None:
    with pytest.raises(ValueError, match="speak_timeout"):
        AIChannel("ai1", provider=MockAIProvider(responses=["x"]), speak_timeout=0)


# --- through the framework -------------------------------------------------------------


async def _kit(policy: Any) -> tuple[RoomKit, MockAIProvider]:
    from roomkit.channels import SMSChannel
    from roomkit.providers.sms.mock import MockSMSProvider

    kit = RoomKit()
    provider = MockAIProvider(responses=["Yes?"])
    kit.register_channel(AIChannel("ai1", provider=provider, speak_policy=policy))
    kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r1", "sms1")
    return kit, provider


async def test_every_decision_reaches_on_speak_decision(advance: Any) -> None:
    kit, provider = await _kit(
        MockSpeakPolicy(
            [SpeakDecision("silent", reason="side talk", judgments={"addressed": 0.1})]
        )
    )
    seen: list[SpeakDecisionEvent] = []

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: Any) -> None:
        seen.append(event)

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def before(event: Any, ctx: Any) -> None:
        raise AssertionError("a silent decision runs no turn")

    await kit.process_inbound(
        InboundMessage(
            channel_id="sms1", sender_id="u1", content=TextContent(body="Paul, are you coming?")
        )
    )
    await advance()
    await kit.close()

    [event] = seen
    assert (event.room_id, event.channel_id) == ("r1", "ai1")
    assert event.decision.reason == "side talk"
    assert event.decision.judgments == {"addressed": 0.1}
    assert provider.calls == []


async def test_without_a_policy_nothing_is_decided(advance: Any) -> None:
    kit, provider = await _kit(None)
    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: Any, ctx: Any) -> None:
        seen.append(event)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="Hello"))
    )
    await advance()
    await kit.close()

    assert seen == []
    assert len(provider.calls) == 1


async def test_always_speak_answers_and_reports_each_decision(advance: Any) -> None:
    kit, provider = await _kit(AlwaysSpeak())
    seen: list[SpeakDecisionEvent] = []

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: Any) -> None:
        seen.append(event)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="Hello"))
    )
    await advance()
    await kit.close()

    assert [e.decision.mode for e in seen] == ["speak"]
    assert len(provider.calls) == 1
