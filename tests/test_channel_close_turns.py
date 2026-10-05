"""A channel's close cuts the calls its turns run, on every door (RMK-511, RFC §9.3, §12.4).

Closed while a tool runs, by itself or by the kit, an AI channel cancels the
call and reports it once, cancelled, with its end row, and asks no further
round, as a speech-to-speech channel's close interrupts its calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels import _ai_steering
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import RoomKitError
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.tools.external import PolicyExternalToolHandler, ToolDecision
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.test_framework import SimpleChannel
from tests.tool_doors import TOOL, TOOL_DICT


class _Hangs:
    """A tool handler that runs until it is cancelled, recording both."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "late"


def _observe(kit: RoomKit) -> list[ToolCallEvent]:
    reports: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        reports.append(event)

    return reports


async def _close(kit: RoomKit, channel: Any, how: str) -> None:
    if how == "channel":
        await channel.close()
    await kit.close()


class _HangsDeciding(PolicyExternalToolHandler):
    """An external handler whose decision runs until it is cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = False
        self.told_cancelled: list[str] = []

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any], **kw: Any
    ) -> ToolDecision:
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return await super().process_tool_call(tool_name, tool_input, **kw)

    async def on_tool_cancelled(
        self, tool_name: str, tool_input: dict[str, Any], **kw: Any
    ) -> None:
        self.told_cancelled.append(kw["tool_call_id"])
        await super().on_tool_cancelled(tool_name, tool_input, **kw)


def _text_channel(door: str) -> tuple[Any, AIChannel, MockAIProvider]:
    """The channel of a text door, with a call that runs until cancelled:
    the channel's own tool, or one its external handler is deciding."""
    external = door == "external"
    call = AIToolCall(id="c1", name="remote" if external else "lookup", arguments={"q": "x"})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ],
        streaming=door != "text-nostream",
    )
    if external:
        handler: Any = _HangsDeciding()
        return handler, AIChannel("ai", provider=provider, external_tool_handler=handler), provider
    handler = _Hangs()
    channel = AIChannel("ai", provider=provider, tools=[TOOL], tool_handler=handler)
    return handler, channel, provider


async def _text_door(door: str, how: str) -> tuple[Any, list[ToolCallEvent], Any]:
    handler, channel, provider = _text_channel(door)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    reports = _observe(kit)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    turn = asyncio.create_task(kit.process_inbound(message))
    await asyncio.wait_for(handler.started.wait(), 3)
    store = kit.store
    await _close(kit, channel, how)
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(turn, 3)
    rows = [
        event.content.outcome
        for event in await store.list_events("r")
        if event.type == EventType.TOOL_CALL_END and isinstance(event.content, ToolCallContent)
    ]
    return handler, reports, (len(provider.calls), rows)


async def _realtime_door(how: str) -> tuple[_Hangs, list[ToolCallEvent], Any]:
    handler = _Hangs()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=[TOOL_DICT],
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    reports = _observe(kit)
    session = await channel.start_session("r", "u", "ws")
    await provider.simulate_tool_call(session, "c1", "lookup", {"q": "x"})
    await asyncio.wait_for(handler.started.wait(), 3)
    await _close(kit, channel, how)
    return handler, reports, len(provider.tool_results)


async def _backend_door(how: str) -> tuple[_Hangs, list[ToolCallEvent], Any]:
    """A realtime delegation run by an agent reasoning backend whose model
    calls the channel's tool."""
    handler = _Hangs()
    call = AIToolCall(id="c1", name="lookup", arguments={"q": "x"})
    model = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    provider = MockRealtimeProvider(full_duplex=True)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=[TOOL_DICT],
        reasoning_backend=AgentReasoningBackend(Agent("reasoner", provider=model)),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    reports = _observe(kit)
    session = await channel.start_session("r", "u", "ws")
    await provider.simulate_delegation(session, "d1", "integrator")
    await asyncio.wait_for(handler.started.wait(), 3)
    await _close(kit, channel, how)
    return handler, reports, None


_DOORS = ["text-stream", "text-nostream", "external", "realtime", "rt-agent-backend"]


@pytest.mark.parametrize("how", ["channel", "kit"])
@pytest.mark.parametrize("door", _DOORS)
async def test_a_close_cancels_the_running_call_and_reports_it_once(door: str, how: str) -> None:
    """The channel's own tool, an external handler's pending decision, a
    realtime session's call, a reasoning backend's call: cancelled, reported
    once, cancelled."""
    call_id = "d1:c1" if door == "rt-agent-backend" else "c1"
    if door == "realtime":
        handler, reports, results_sent = await _realtime_door(how)
        assert results_sent == 0
    elif door == "rt-agent-backend":
        handler, reports, _ = await _backend_door(how)
    else:
        handler, reports, (rounds, rows) = await _text_door(door, how)
        # No further round, and the call's end row says it was cancelled.
        assert (rounds, rows) == (1, ["cancelled"])
    if door == "external":
        assert handler.told_cancelled == ["c1"]

    assert handler.cancelled is True
    assert [(r.tool_call_id, r.cancelled, r.is_error) for r in reports] == [(call_id, True, True)]
    assert "Tool call cancelled" in str(reports[0].result)


async def test_a_handler_that_closes_its_own_channel_runs_on() -> None:
    """A call whose handler closes the channel is not cut by that close: it
    reports its own outcome, and the close does not wait for its turn."""
    holder: dict[str, AIChannel] = {}

    async def closes_its_channel(name: str, arguments: dict[str, Any]) -> str:
        await holder["ai"].close()
        return "closed"

    call = AIToolCall(id="c1", name="lookup", arguments={"q": "x"})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="bye"),
        ]
    )
    channel = AIChannel("ai", provider=provider, tools=[TOOL], tool_handler=closes_its_channel)
    holder["ai"] = channel
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    reports = _observe(kit)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))

    await asyncio.wait_for(kit.process_inbound(message), 3)
    for _ in range(100):
        if reports:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert [(r.cancelled, r.is_error, str(r.result)) for r in reports] == [
        (False, False, "closed")
    ]
    # The channel closed: the turn asked no further round.
    assert len(provider.calls) == 1


async def test_a_close_cuts_a_delegated_turn_s_running_call() -> None:
    """The delegated turn runs on the worker's channel: the kit's close cuts
    its call the same way, reported once, cancelled."""
    handler = _Hangs()
    call = AIToolCall(id="c1", name="lookup", arguments={"q": "x"})
    worker = Agent(
        "worker",
        provider=MockAIProvider(
            ai_responses=[
                AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
                AIResponse(content="done"),
            ]
        ),
        tools=[TOOL],
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(worker)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    reports = _observe(kit)

    delegation = asyncio.create_task(kit.delegate("r", "worker", "Look it up.", wait=True))
    await asyncio.wait_for(handler.started.wait(), 3)
    await kit.close()
    with contextlib.suppress(asyncio.CancelledError, RoomKitError):
        await asyncio.wait_for(delegation, 3)

    assert handler.cancelled is True
    assert [(r.cancelled, r.is_error) for r in reports] == [(True, True)]


class _ClosesOnStart(InMemoryRealtime):
    """A realtime backend whose round START publish yields (a network round
    trip), during which the channel closes."""

    def __init__(self) -> None:
        super().__init__()
        self.channel: AIChannel | None = None
        self.close_task: asyncio.Task[None] | None = None

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.TOOL_CALL_START and self.close_task is None:
            assert self.channel is not None
            self.close_task = asyncio.create_task(self.channel.close())
            await asyncio.sleep(0.05)
        await super().publish(channel, event)


async def test_a_close_while_the_round_s_start_is_published_runs_none_of_its_calls() -> None:
    """The close lands after the loop's check and before the calls start:
    none runs, each is reported cancelled with its end row."""
    ran: list[str] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ran after the close"

    backend = _ClosesOnStart()
    provider = _one_call_provider()
    channel = AIChannel("ai", provider=provider, tools=[TOOL], tool_handler=lookup)
    backend.channel = channel
    kit = RoomKit(realtime=backend)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    reports = _observe(kit)

    await kit.process_inbound(_message())
    assert backend.close_task is not None
    await backend.close_task
    rows = await _end_rows(kit)
    await kit.close()

    assert ran == []
    assert [(r.cancelled, r.is_error) for r in reports] == [(True, True)]
    assert (len(provider.calls), rows) == (1, ["cancelled"])


async def test_a_handler_closing_its_channel_spares_itself_not_its_round() -> None:
    """A hang-up call closes the channel: it runs on and reports its own
    outcome; the other call of its round is cancelled and reported so."""
    holder: dict[str, AIChannel] = {}
    sibling = _Hangs()

    async def tools(name: str, arguments: dict[str, Any]) -> str:
        if name == "hangup":
            await sibling.started.wait()
            await holder["ai"].close()
            return "closed"
        return await sibling(name, arguments)

    calls = [
        AIToolCall(id="h1", name="hangup", arguments={"q": "x"}),
        AIToolCall(id="c1", name="lookup", arguments={"q": "x"}),
    ]
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=calls),
            AIResponse(content="bye"),
        ]
    )
    hangup = TOOL.model_copy(update={"name": "hangup"})
    channel = AIChannel("ai", provider=provider, tools=[TOOL, hangup], tool_handler=tools)
    holder["ai"] = channel
    kit, reports = await _text_room(channel)

    await asyncio.wait_for(kit.process_inbound(_message()), 3)
    rows = await _end_rows(kit)
    await kit.close()

    assert sibling.cancelled is True
    assert sorted((r.tool_call_id, r.cancelled) for r in reports) == [("c1", True), ("h1", False)]
    assert sorted(rows) == ["cancelled", "served"]
    assert len(provider.calls) == 1


@pytest.mark.parametrize("trigger", [HookTrigger.AFTER_TOOL_ROUND, HookTrigger.ON_AI_RESPONSE])
async def test_a_close_from_a_hook_of_the_turn_does_not_wait_for_that_turn(
    trigger: HookTrigger, caplog: pytest.LogCaptureFixture
) -> None:
    """The hook runs inside the turn: the close cuts it and returns at once,
    with no turn said to be still running."""

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    channel = AIChannel("ai", provider=_one_call_provider(), tools=[TOOL], tool_handler=lookup)
    kit, _ = await _text_room(channel)
    took: list[float] = []

    @kit.hook(trigger, name="closer")
    async def closer(event: Any, ctx: Any) -> HookResult:
        loop = asyncio.get_running_loop()
        started = loop.time()
        await channel.close()
        took.append(loop.time() - started)
        return HookResult.allow()

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(kit.process_inbound(_message()), 10)
    await kit.close()

    assert took and took[0] < 1.0
    assert "still running" not in caplog.text


async def test_a_second_close_does_not_wait_again_for_a_turn_the_first_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call that outlives its cancellation keeps its turn running past the
    first close's wait; the kit's close after it does not wait a second time."""
    monkeypatch.setattr(_ai_steering, "CLOSE_WAIT_S", 0.2)
    started = asyncio.Event()

    async def outlives(name: str, arguments: dict[str, Any]) -> str:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(1)
        return "late"

    channel = AIChannel("ai", provider=_one_call_provider(), tools=[TOOL], tool_handler=outlives)
    kit, _ = await _text_room(channel)
    turn = asyncio.create_task(kit.process_inbound(_message()))
    await asyncio.wait_for(started.wait(), 3)
    loop = asyncio.get_running_loop()

    first = loop.time()
    await channel.close()
    first_took = loop.time() - first
    second = loop.time()
    await kit.close()
    second_took = loop.time() - second
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(turn, 3)

    assert first_took >= 0.2
    assert second_took < 0.2


def _message() -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))


async def _text_room(channel: AIChannel) -> tuple[RoomKit, list[ToolCallEvent]]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, _observe(kit)


async def _end_rows(kit: RoomKit) -> list[str | None]:
    return [
        event.content.outcome
        for event in await kit.store.list_events("r")
        if event.type == EventType.TOOL_CALL_END and isinstance(event.content, ToolCallContent)
    ]


def _one_call_provider() -> MockAIProvider:
    call = AIToolCall(id="c1", name="lookup", arguments={"q": "x"})
    return MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )


class _ClosingProvider(MockAIProvider):
    closed = False

    async def close(self) -> None:
        self.closed = True


async def test_a_realtime_channel_s_close_closes_its_reasoning_backend_s_agent() -> None:
    """The backend owns its agent: the channel's close releases the agent's
    provider, as it releases its own."""
    model = _ClosingProvider(responses=["x"])
    channel = RealtimeVoiceChannel(
        "rt",
        provider=MockRealtimeProvider(full_duplex=True),
        transport=MockRealtimeTransport(),
        reasoning_backend=AgentReasoningBackend(Agent("reasoner", provider=model)),
    )

    await channel.close()

    assert model.closed is True
