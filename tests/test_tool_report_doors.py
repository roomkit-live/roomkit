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
from roomkit.channels.ai import AIChannel
from roomkit.models.event import ToolCallContent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
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
