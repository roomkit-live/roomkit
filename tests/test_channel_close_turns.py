"""A channel's close cuts the calls its turns run, on every door (RMK-511, RFC §9.3, §12.4).

Closed while a tool runs, by itself or by the kit, an AI channel cancels the
call and reports it once, cancelled, with its end row, and asks no further
round, as a speech-to-speech channel's close interrupts its calls.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
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


async def _text_door(streaming: bool, how: str) -> tuple[_Hangs, list[ToolCallEvent], Any]:
    handler = _Hangs()
    call = AIToolCall(id="c1", name="lookup", arguments={"q": "x"})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    channel = AIChannel("ai", provider=provider, tools=[TOOL], tool_handler=handler)
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


@pytest.mark.parametrize("how", ["channel", "kit"])
@pytest.mark.parametrize("door", ["text-stream", "text-nostream", "realtime"])
async def test_a_close_cancels_the_running_call_and_reports_it_once(door: str, how: str) -> None:
    if door == "realtime":
        handler, reports, results_sent = await _realtime_door(how)
        assert results_sent == 0
    else:
        handler, reports, (rounds, rows) = await _text_door(door == "text-stream", how)
        # No further round, and the call's end row says it was cancelled.
        assert (rounds, rows) == (1, ["cancelled"])

    assert handler.cancelled is True
    assert [(r.tool_call_id, r.cancelled, r.is_error) for r in reports] == [("c1", True, True)]
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
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(delegation, 3)

    assert handler.cancelled is True
    assert [(r.cancelled, r.is_error) for r in reports] == [(True, True)]
