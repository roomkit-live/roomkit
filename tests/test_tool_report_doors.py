"""A tool call's report and its rows say the same thing on every door
(RMK-498, RFC §9.3, §12.4).

A call the turn cuts while it runs is closed with the arguments it ran with,
as its report is; two calls under one id in a round are two calls, the second
refused as a realtime session refuses it; a reasoning backend's calls are
reported under the ids its model gave them.
"""

from __future__ import annotations

import asyncio
import contextlib
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
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.event import ToolCallContent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.test_framework import SimpleChannel

SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}
LOOKUP = AITool(name="lookup", description="Look up", parameters=SCHEMA)
MODEL = {"q": "<EMAIL_1>"}
REAL = {"q": "real@x"}


def _provider(calls: list[AIToolCall], *, streaming: bool) -> MockAIProvider:
    return MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=calls),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )


async def _room(kit: RoomKit, channel: AIChannel) -> list[ToolCallEvent]:
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", channel.channel_id, category=ChannelCategory.INTELLIGENCE)
    reports: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, context: Any) -> None:
        reports.append(event)

    return reports


async def _end_rows(kit: RoomKit) -> list[tuple[str | None, dict[str, Any]]]:
    return [
        (event.content.outcome, dict(event.content.arguments))
        for event in await kit.store.list_events("r")
        if isinstance(event.content, ToolCallContent) and event.type.value == "tool_call_end"
    ]


@pytest.mark.parametrize("streaming", [True, False], ids=["stream", "generate"])
async def test_a_call_the_turn_cuts_is_closed_with_the_arguments_it_ran_with(
    streaming: bool,
) -> None:
    """A ``BEFORE_TOOL_USE`` hook puts the real value back; the handler runs
    with it and hangs; the turn is cancelled. Its END row carries what ran,
    as its report does."""
    ran: list[dict[str, Any]] = []

    async def hangs(name: str, arguments: dict[str, Any]) -> str:
        ran.append(dict(arguments))
        await asyncio.sleep(30)
        return "late"

    call = AIToolCall(id="c1", name="lookup", arguments=dict(MODEL))
    channel = AIChannel(
        "ai", provider=_provider([call], streaming=streaming), tools=[LOOKUP], tool_handler=hangs
    )
    kit = RoomKit()
    reports = await _room(kit, channel)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="detokenise")
    async def detokenise(event: Any, context: Any) -> HookResult:
        return HookResult(action="allow", metadata={"arguments": dict(REAL)})

    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))
    turn = asyncio.create_task(kit.process_inbound(message))
    for _ in range(300):
        if ran:
            break
        await asyncio.sleep(0.01)
    turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn
    for _ in range(300):
        if reports:
            break
        await asyncio.sleep(0.01)
    rows = await _end_rows(kit)
    await kit.close()

    assert [report.arguments for report in reports] == [REAL]
    assert rows == [("cancelled", REAL)]


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _blocks(event: Any, context: Any) -> HookResult:
    return HookResult.block("no")


_BACKEND_CASES = {
    "served": (AIToolCall(id="c1", name="lookup", arguments={"q": "x"}), None),
    "gate-refused": (AIToolCall(id="c1", name="lookup", arguments={"q": "x"}), _blocks),
    "loop-refused": (AIToolCall(id="c1", name="nope", arguments={}), None),
    "provider-served": (
        AIToolCall(
            id="c1", name="web_search", arguments={"q": "x"}, served=ServedCall(result="3 hits")
        ),
        None,
    ),
}


@pytest.mark.parametrize("case", list(_BACKEND_CASES))
async def test_a_reasoning_backend_s_call_is_reported_under_its_model_s_id(case: str) -> None:
    """Served through the gate, refused by it, refused by the backend's own
    loop or served by its provider: one report, under ``<delegation>:<id>``."""
    call, before = _BACKEND_CASES[case]
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
        tool_handler=_found,
        tools=[{"name": "lookup", "description": "Look up", "parameters": SCHEMA}],
        reasoning_backend=AgentReasoningBackend(Agent("reasoner", provider=model)),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    reports: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, context: Any) -> None:
        reports.append(event)

    if before is not None:
        kit.hook(HookTrigger.BEFORE_TOOL_USE, name="gate")(before)
    session = await channel.start_session("r", "u", "ws")
    await provider.simulate_delegation(session, "d1", "integrator")
    for _ in range(300):
        if len(model.calls) > 1 and reports:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    await kit.close()

    assert [report.tool_call_id for report in reports] == ["d1:c1"]


async def _two_calls_text(streaming: bool) -> tuple[list[ToolCallEvent], list[dict[str, Any]]]:
    ran: list[dict[str, Any]] = []

    async def serves(name: str, arguments: dict[str, Any]) -> str:
        ran.append(dict(arguments))
        await asyncio.sleep(0.05)
        return f"fine {arguments['q']}"

    calls = [
        AIToolCall(id="c1", name="lookup", arguments={"q": "first"}),
        AIToolCall(id="c1", name="lookup", arguments={"q": "second"}),
    ]
    channel = AIChannel(
        "ai", provider=_provider(calls, streaming=streaming), tools=[LOOKUP], tool_handler=serves
    )
    kit = RoomKit()
    reports = await _room(kit, channel)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))
    await kit.process_inbound(message)
    rows = await _end_rows(kit)
    await kit.close()
    assert rows == [("served", {"q": "first"}), ("refused", {"q": "second"})]
    return reports, ran


async def _two_calls_realtime() -> tuple[list[ToolCallEvent], list[dict[str, Any]]]:
    ran: list[dict[str, Any]] = []

    async def serves(name: str, arguments: dict[str, Any]) -> str:
        ran.append(dict(arguments))
        await asyncio.sleep(0.2)
        return f"fine {arguments['q']}"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=serves,
        tools=[{"name": "lookup", "description": "Look up", "parameters": SCHEMA}],
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    reports: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, context: Any) -> None:
        reports.append(event)

    session = await channel.start_session("r", "u", "ws")
    await provider.simulate_tool_call(session, "c1", "lookup", {"q": "first"})
    await provider.simulate_tool_call(session, "c1", "lookup", {"q": "second"})
    for _ in range(300):
        if len(reports) == 2:
            break
        await asyncio.sleep(0.01)
    await kit.close()
    return reports, ran


@pytest.mark.parametrize("door", ["text-stream", "text-generate", "realtime"])
async def test_two_calls_under_one_id_are_two_calls_the_second_refused(door: str) -> None:
    """The first keeps the id and runs; the second is refused as a call whose
    id is still in flight, and reported as a call of its own (RFC §12.4)."""
    if door == "realtime":
        reports, ran = await _two_calls_realtime()
    else:
        reports, ran = await _two_calls_text(door == "text-stream")

    assert ran == [{"q": "first"}]
    outcomes = sorted((r.arguments["q"], r.refused, r.is_error) for r in reports)
    assert outcomes == [("first", False, False), ("second", True, True)]
    [refused] = [r for r in reports if r.refused]
    assert "has not had its result yet" in str(refused.result)
