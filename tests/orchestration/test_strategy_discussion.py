"""The Discussion strategy end to end (RFC §19.7.5).

Agents and a person in one room: a message queues the agents it asks for, the
strategy gives their turns one at a time, each reading the room as it is, and
an agent's answer that names another queues it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.event_router import CHAIN_DEPTH_LIMIT
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.orchestration.strategies.discussion import (
    Discussion,
    SpeakQueueChange,
    SpeakQueueEvent,
)
from roomkit.providers.ai.base import AIResponse, StreamTextDelta
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel

NOTES = "[Discussion: the room and how it works]"


class _Chunked(MockAIProvider):
    """Streams its text one character at a time, as a slow model would."""

    def __init__(self, *answers: str) -> None:
        super().__init__(ai_responses=[AIResponse(content=a) for a in answers], streaming=True)

    async def _scripted_events(self, context: Any) -> AsyncIterator[Any]:
        async for event in super()._scripted_events(context):
            if isinstance(event, StreamTextDelta):
                for char in event.text:
                    yield StreamTextDelta(text=char)
            else:
                yield event


class _StreamingTransport(SimpleChannel):
    """A transport that takes the live text of a streamed answer."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.live: list[str] = []

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for chunk in text_stream:
            self.live.append(str(chunk))
        return ChannelOutput.empty()


def _provider(*answers: str) -> MockAIProvider:
    return MockAIProvider(ai_responses=[AIResponse(content=a) for a in answers])


async def _room(
    kit: RoomKit,
    agents: dict[str, MockAIProvider],
    human: SimpleChannel | None = None,
    **options: Any,
) -> SimpleChannel:
    human = human or SimpleChannel("ops")
    kit.register_channel(human)
    channels = [AIChannel(name, provider=p) for name, p in agents.items()]
    await kit.create_room(room_id="r1", orchestration=Discussion(channels, **options))
    await kit.attach_channel("r1", human.channel_id)
    return human


async def _say(
    kit: RoomKit,
    body: str,
    addressed_to: list[str] | None = None,
    *,
    event_type: EventType = EventType.MESSAGE,
) -> InboundResult:
    return await kit.process_inbound(
        InboundMessage(
            channel_id="ops",
            sender_id="oncall",
            content=TextContent(body=body),
            addressed_to=addressed_to,
            event_type=event_type,
        )
    )


async def _settle(kit: RoomKit, room_id: str = "r1") -> None:
    """Wait until the discussion runs no turn and has none it can give."""
    idle = 0
    async with asyncio.timeout(5):
        while idle < 5:
            await asyncio.sleep(0.01)
            queue = kit.speak_queue(room_id)
            assert queue is not None
            settled = not queue.queue or queue.waiting or queue.over
            idle = idle + 1 if queue.speaking is None and settled else 0


def _texts(provider: MockAIProvider, call: int = -1) -> list[str]:
    return [str(m.content) for m in provider.calls[call].messages]


async def _timeline(kit: RoomKit) -> list[RoomEvent]:
    """The room's messages, its system events left out."""
    events = await kit.store.list_events(
        "r1", limit=100, event_filter=EventFilter(include_blocked=True)
    )
    return [e for e in events if e.type == EventType.MESSAGE]


async def test_a_person_naming_an_agent_asks_that_agent_alone() -> None:
    a, b = _provider("a answers"), _provider("b answers")
    kit = RoomKit()
    human = await _room(kit, {"a": a, "b": b})

    result = await _say(kit, "@a what is the error rate?")
    # The answers come in the turns that follow, never in the result.
    assert result.response_events == []
    await _settle(kit)

    assert (len(a.calls), len(b.calls)) == (1, 0)
    assert [e.content.body for e in human.delivered if e.source.channel_id == "a"] == ["a answers"]
    await kit.close()


async def test_an_unaddressed_message_asks_everyone_one_at_a_time() -> None:
    a, b = _provider("a: the database is down"), _provider("b: agreed")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b})

    await _say(kit, "what happened tonight?")
    await _settle(kit)

    assert (len(a.calls), len(b.calls)) == (1, 1)
    texts = _texts(b)
    asked = next(i for i, t in enumerate(texts) if "what happened tonight?" in t)
    first = next(i for i, t in enumerate(texts) if "the database is down" in t)
    # b's turn reads the person's message at its place, then a's answer.
    assert asked < first
    assert any(NOTES in t and "@a" in t for t in texts)
    await kit.close()


async def test_everyone_opens_in_the_hosts_order() -> None:
    a, b = _provider("a here"), _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, everyone=["b", "a"])

    await _say(kit, "status?")
    await _settle(kit)

    speakers = [e.source.channel_id for e in await _timeline(kit) if e.source.channel_id != "ops"]
    assert speakers == ["b", "a"]
    await kit.close()


async def test_an_agent_naming_another_queues_it_once_its_turn_has_ended() -> None:
    a = _provider("@b can you check the logs? @b they are in /var/log")
    b = _provider("logs are clean")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b})

    await _say(kit, "@a please investigate")
    await _settle(kit)

    assert (len(a.calls), len(b.calls)) == (1, 1)
    asked = next(e for e in await _timeline(kit) if e.source.channel_id == "a")
    assert asked.addressed_to == ["b"]
    assert "can you check the logs" in _texts(b)[-1] or any(
        "can you check the logs" in t for t in _texts(b)
    )
    await kit.close()


async def test_the_depth_limit_stops_a_turn_once_and_a_person_resumes_it() -> None:
    a, b = _provider("@b your turn"), _provider("@a your turn")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, max_depth=3)

    await _say(kit, "@a start")
    await _settle(kit)

    # a at depth 1, b at depth 2; a's turn at depth 3 is stopped.
    assert (len(a.calls), len(b.calls)) == (1, 1)
    records = [e for e in await _timeline(kit) if e.blocked_by == CHAIN_DEPTH_LIMIT]
    assert [(e.source.channel_id, e.status) for e in records] == [("a", EventStatus.BLOCKED)]
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.waiting and queue.queue == ("a",)

    await _say(kit, "@a carry on")
    await _settle(kit)
    assert (len(a.calls), len(b.calls)) == (2, 2)
    await kit.close()


async def test_max_turns_ends_the_discussion_and_refuses_later_instructions() -> None:
    a, b = _provider("a here"), _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, max_turns=2)

    await _say(kit, "status?")
    await _settle(kit)
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.over

    refused = await _say(kit, "summarise", ["a"], event_type=EventType.INSTRUCTION)
    assert (refused.blocked, refused.reason) == (True, "discussion_over")
    await _say(kit, "@a still there?")
    await _settle(kit)
    assert (len(a.calls), len(b.calls)) == (1, 1)
    await kit.close()


async def test_done_ends_the_discussion_before_the_next_turn() -> None:
    a, b = _provider("a here"), _provider("b here")
    over = {"r1": False}
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, done=lambda room_id: over[room_id])

    await _say(kit, "@a hello")
    await _settle(kit)
    over["r1"] = True
    await _say(kit, "@b hello")
    await _settle(kit)

    assert (len(a.calls), len(b.calls)) == (1, 0)
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.over
    await kit.close()


async def test_a_silent_agent_says_nothing_to_the_room() -> None:
    a, b = _provider("a: found it"), _provider("(silent)")
    kit = RoomKit()
    human = await _room(kit, {"a": a, "b": b})

    await _say(kit, "anyone?")
    await _settle(kit)

    rows = [e for e in await _timeline(kit) if e.source.channel_id == "b"]
    assert [(e.status, e.blocked_by) for e in rows] == [(EventStatus.BLOCKED, "discussion_silent")]
    assert [e.source.channel_id for e in human.delivered] == ["a"]
    await kit.close()


async def test_a_streamed_silence_never_reaches_a_live_transport() -> None:
    a, b = _Chunked("(Silent)."), _Chunked("(si) is the chemical symbol")
    kit = RoomKit()
    human = await _room(kit, {"a": a, "b": b}, human=_StreamingTransport("ops"))
    assert isinstance(human, _StreamingTransport)

    await _say(kit, "anyone?")
    await _settle(kit)

    live = "".join(human.live)
    assert "Silent" not in live
    # A start that could have been the token is held, then released whole.
    assert "(si) is the chemical symbol" in live
    rows = {e.source.channel_id: e for e in await _timeline(kit) if e.source.channel_id != "ops"}
    assert rows["a"].blocked_by == "discussion_silent"
    assert rows["b"].status == EventStatus.DELIVERED
    await kit.close()


async def test_an_instruction_gives_its_agent_a_turn_taking_it_as_input() -> None:
    a = _provider("summary: all good")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": _provider()})

    result = await _say(kit, "summarise the incident", ["a"], event_type=EventType.INSTRUCTION)
    assert not result.blocked
    await _settle(kit)

    assert len(a.calls) == 1
    assert "summarise the incident" in _texts(a)[-1] or any(
        "summarise the incident" in t for t in _texts(a)
    )
    assert all(e.type != EventType.INSTRUCTION for e in await _timeline(kit))
    await kit.close()


async def test_an_agent_that_only_listens_answers_only_a_person_naming_it() -> None:
    a, b = _provider("@b please look"), _provider("b looked")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b})
    await kit.listen_only("r1", ["b"])

    await _say(kit, "@a go")
    await _settle(kit)
    assert (len(a.calls), len(b.calls)) == (1, 0)
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.listening == frozenset({"b"}) and queue.waiting

    await _say(kit, "@b your view?")
    await _settle(kit)
    assert len(b.calls) == 1

    await kit.talk_again("r1", ["b"])
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.listening == frozenset()
    await kit.close()


class _Slow(MockAIProvider):
    """Answers once released: a turn still running when the host acts."""

    def __init__(self) -> None:
        super().__init__(ai_responses=[AIResponse(content="too late")])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, context: Any) -> AIResponse:
        self.started.set()
        await self.release.wait()
        return await super().generate(context)


async def test_listen_only_cuts_the_turn_its_agent_is_taking() -> None:
    a = _Slow()
    kit = RoomKit()
    human = await _room(kit, {"a": a, "b": _provider()})

    await _say(kit, "@a go")
    await asyncio.wait_for(a.started.wait(), 5)
    await kit.listen_only("r1", ["a"])
    # As a Cancel does: the answer generated meanwhile is dropped.
    a.release.set()
    await _settle(kit)

    queue = kit.speak_queue("r1")
    assert queue is not None and queue.speaking is None and queue.listening == frozenset({"a"})
    assert [e for e in human.delivered if e.source.channel_id == "a"] == []
    await kit.close()


async def test_an_agent_asking_a_person_is_answered_by_their_next_message() -> None:
    a, b = _provider("@ops which region?", "thanks, rolling back"), _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, people=["ops"])

    await _say(kit, "@a deploy failed")
    await _settle(kit)
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.asked == (("a", "ops"),) and queue.waiting

    await _say(kit, "eu-west")
    await _settle(kit)
    # The unaddressed answer goes to the agent that asked, not to everyone.
    assert (len(a.calls), len(b.calls)) == (2, 0)
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.asked == ()
    await kit.close()


async def test_with_addressed_only_an_unaddressed_message_asks_no_agent() -> None:
    a, b = _provider("a here"), _provider("b here")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": b}, addressed_only=True)

    await _say(kit, "just chatting among ourselves")
    await _settle(kit)
    assert (len(a.calls), len(b.calls)) == (0, 0)

    await _say(kit, "@b over to you")
    await _settle(kit)
    assert (len(a.calls), len(b.calls)) == (0, 1)
    await kit.close()


async def test_speak_queue_changes_are_announced_in_order() -> None:
    kit = RoomKit()
    seen: list[tuple[SpeakQueueChange, tuple[str, ...]]] = []

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def follow(event: SpeakQueueEvent, context: RoomContext) -> None:
        seen.append((event.change, event.channel_ids))

    await _room(kit, {"a": _provider("@b over to you"), "b": _provider("done")})
    await _say(kit, "@a start")
    await _settle(kit)
    await asyncio.sleep(0.05)

    assert seen == [
        (SpeakQueueChange.QUEUED, ("a",)),
        (SpeakQueueChange.TURN_GIVEN, ("a",)),
        (SpeakQueueChange.TURN_ENDED, ("a",)),
        (SpeakQueueChange.QUEUED, ("b",)),
        (SpeakQueueChange.TURN_GIVEN, ("b",)),
        (SpeakQueueChange.TURN_ENDED, ("b",)),
    ]
    await kit.close()


async def test_the_queue_outlives_the_process_but_not_who_speaks() -> None:
    store = InMemoryStore()
    kit = RoomKit(store=store)
    await _room(kit, {"a": _provider("@b your turn"), "b": _provider("b")}, max_depth=2)
    await _say(kit, "@a start")
    await _settle(kit)
    queue = kit.speak_queue("r1")
    # b's turn at depth 2 is stopped by max_depth=2: b stays queued.
    assert queue is not None and queue.queue == ("b",)
    await kit.close()

    b = _provider("b resumes")
    again = RoomKit(store=store)
    again.register_channel(SimpleChannel("ops"))
    strategy = Discussion([AIChannel("a", provider=_provider()), AIChannel("b", provider=b)])
    await strategy.install(again, "r1")
    restored = again.speak_queue("r1")
    assert restored is not None
    assert (restored.queue, restored.speaking, restored.waiting) == (("b",), None, True)
    await again.close()


async def test_uninstalling_gives_the_room_back_to_its_policy() -> None:
    a = _provider("a answers")
    kit = RoomKit()
    await _room(kit, {"a": a})
    strategy = Discussion([kit.channels["a"]])  # type: ignore[list-item]

    await strategy.uninstall(kit, "r1")
    assert kit.speak_queue("r1") is None
    result = await _say(kit, "hello")
    assert [e.content.body for e in result.response_events] == ["a answers"]
    await kit.close()


async def test_a_regenerated_answer_is_a_turn_of_its_own() -> None:
    a = _provider("first answer", "second answer")
    kit = RoomKit()
    await _room(kit, {"a": a, "b": _provider()})
    await _say(kit, "@a hello")
    await _settle(kit)

    result = await kit.regenerate_response("r1")
    assert result is not None and not result.blocked
    await _settle(kit)

    bodies = [e.content.body for e in await _timeline(kit) if e.source.channel_id == "a"]
    assert bodies == ["first answer", "second answer"]
    await kit.close()


# -- Refusals (rule 1) --


async def test_a_room_with_another_intelligence_channel_is_refused() -> None:
    kit = RoomKit()
    kit.register_channel(AIChannel("other", provider=_provider()))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "other", category=ChannelCategory.INTELLIGENCE)

    strategy = Discussion([AIChannel("a", provider=_provider())])
    with pytest.raises(ValueError, match="not one of its agents"):
        await strategy.install(kit, "r1")
    assert kit.speak_queue("r1") is None
    await kit.close()


async def test_a_room_with_a_voice_channel_is_refused() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("phone", channel_type=ChannelType.VOICE))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "phone")

    with pytest.raises(ValueError, match="voice or realtime"):
        await Discussion([AIChannel("a", provider=_provider())]).install(kit, "r1")
    await kit.close()


async def test_a_second_discussion_in_the_room_is_refused() -> None:
    kit = RoomKit()
    await _room(kit, {"a": _provider()})

    with pytest.raises(ValueError, match="holds a discussion already"):
        await Discussion([kit.channels["a"]]).install(kit, "r1")  # type: ignore[list-item]
    await kit.close()


def test_a_discussion_is_checked_when_built() -> None:
    a = AIChannel("a", provider=_provider())
    with pytest.raises(ValueError, match="at least one agent"):
        Discussion([])
    with pytest.raises(ValueError, match="distinct"):
        Discussion([a, AIChannel("a", provider=_provider())])
    with pytest.raises(ValueError, match="not agents"):
        Discussion([a], everyone=["b"])
    with pytest.raises(ValueError, match="max_depth"):
        Discussion([a], max_depth=1)
    with pytest.raises(TypeError, match="not an AI channel"):
        Discussion([SimpleChannel("x")])  # type: ignore[list-item]
