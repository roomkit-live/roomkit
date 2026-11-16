"""A channel closed while its start row is read never starts that call."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomEvent
from roomkit.channels import _ai_steering
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai import AIResponse, AIToolCall, MockAIProvider
from tests.test_channel_close_turns import _message, _text_room
from tests.tool_doors import TOOL


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_close_while_the_start_row_is_read_never_starts_the_call(
    streaming: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_ai_steering, "CLOSE_WAIT_S", 0.05)
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "must not execute"

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="", tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"q": "x"})]
            ),
            AIResponse(content="late"),
        ],
        streaming=streaming,
    )
    agent = AIChannel("ai", provider=provider, tools=[TOOL], tool_handler=handler)
    kit, reports = await _text_room(agent)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="close before call start")
    async def close(event: RoomEvent, ctx: Any) -> HookResult:
        if event.type == EventType.TOOL_CALL_START:
            await agent.close()
        return HookResult.allow()

    try:
        await asyncio.wait_for(kit.process_inbound(_message()), 3)
        await kit.hook_engine.drain_observers()
        rows = [
            row.content
            for row in await kit.store.list_events("r")
            if row.type == EventType.TOOL_CALL_END
        ]
        assert ran == [] and len(provider.calls) == 1
        assert len(reports) == len(rows) == 1
        assert reports[0].cancelled and reports[0].is_error
        assert "The turn ended before its result." in str(reports[0].result)
        assert isinstance(rows[0], ToolCallContent)
        assert rows[0].outcome == "cancelled"
        assert rows[0].result is None
        assert rows[0].duration_ms == 0
        assert rows[0].error == "cancelled"
    finally:
        await kit.close()
