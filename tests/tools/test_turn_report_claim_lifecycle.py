"""Public claims are shared with provider reports and cancellation cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from tests.test_framework import SimpleChannel

from roomkit import (
    AIChannel,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider, ServedCall
from roomkit.tools import turn_report_claim


async def _room(channel: AIChannel) -> tuple[RoomKit, list[ToolCallEvent]]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ai")
    reports: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def observed(event: ToolCallEvent, context: Any) -> None:
        reports.append(event)

    return kit, reports


def _message() -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go"))


async def test_provider_occurrences_keep_independent_claims_under_one_id(streaming: bool) -> None:
    calls = [
        AIToolCall(id="c1", name="lookup", arguments={"q": query}, served=ServedCall(result=query))
        for query in ("first", "second")
    ]
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[AIResponse(content="", tool_calls=calls), AIResponse(content="done")],
    )
    kit, reports = await _room(AIChannel("ai", provider=provider))
    claimed: list[tuple[str, bool, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def claim_report(event: ToolCallEvent, context: Any) -> HookResult:
        claim = turn_report_claim(event.tool_call_id, event.channel_id)
        assert claim is not None
        claimed.append((str(event.arguments["q"]), claim(), claim()))
        return HookResult.allow()

    try:
        await kit.process_inbound(_message(), room_id="r1")
    finally:
        await kit.close()
    rows = [
        event.content
        for event in await kit.store.list_events("r1")
        if event.type.value == "tool_call_end" and isinstance(event.content, ToolCallContent)
    ]
    assert claimed == [("first", True, False), ("second", True, False)]
    assert reports == []
    assert [(row.arguments["q"], row.outcome) for row in rows] == [
        ("first", "served"),
        ("second", "served"),
    ]


async def test_a_claimed_local_call_keeps_its_end_row_when_cancelled(streaming: bool) -> None:
    entered = asyncio.Event()
    retained: list[Callable[[], bool]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None and claim()
        retained.append(claim)
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled handler must not finish")

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="lookup")])],
    )
    kit, reports = await _room(
        AIChannel(
            "ai",
            provider=provider,
            tools=[AITool(name="lookup", description="Lookup")],
            tool_handler=handler,
        )
    )
    turn = asyncio.create_task(kit.process_inbound(_message(), room_id="r1"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
    finally:
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await kit.close()
    rows = [
        event.content.outcome
        for event in await kit.store.list_events("r1")
        if event.type.value == "tool_call_end" and isinstance(event.content, ToolCallContent)
    ]
    assert rows == ["cancelled"]
    assert reports == []
    assert len(retained) == 1 and retained[0]() is False
