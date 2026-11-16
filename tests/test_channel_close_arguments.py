"""A closed call is reported with the arguments that reached its handler."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, ToolCallEvent
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_channel_close_turns import _message, _text_room
from tests.tool_doors import TOOL


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_closed_call_reports_the_arguments_its_handler_ran_with(streaming: bool) -> None:
    started = asyncio.Event()
    received: list[dict[str, Any]] = []

    async def held(name: str, arguments: dict[str, Any]) -> str:
        received.append(arguments)
        started.set()
        await asyncio.Event().wait()
        return "late"

    model = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"q": "model"})],
            ),
            AIResponse(content="late"),
        ],
        streaming=streaming,
    )
    channel = AIChannel("ai", provider=model, tools=[TOOL], tool_handler=held)
    kit, reports = await _text_room(channel)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="rewrite")
    async def rewrite(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult.modify(replace(event, arguments={"q": "rewritten"}))

    turn = asyncio.create_task(kit.process_inbound(_message()))
    try:
        await asyncio.wait_for(started.wait(), 3)
        await channel.close()
        await asyncio.wait_for(turn, 3)
        await kit.hook_engine.drain_observers()
        rows = [
            row.content
            for row in await kit.store.list_events("r")
            if row.type == EventType.TOOL_CALL_END
        ]
        assert received == [{"q": "rewritten"}]
        assert len(reports) == len(rows) == 1
        assert reports[0].arguments == {"q": "rewritten"}
        assert isinstance(rows[0], ToolCallContent)
        assert rows[0].arguments == {"q": "rewritten"}
        assert "The channel closed before its result." in str(reports[0].result)
        assert "The channel closed before its result." in str(rows[0].result)
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await kit.close()
