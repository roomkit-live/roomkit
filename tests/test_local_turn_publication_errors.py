"""A failed ephemeral leaves the text tool loop and later publications intact."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit
from roomkit.providers.ai import AIResponse, AIToolCall, MockAIProvider
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from roomkit.realtime.memory import InMemoryRealtime
from tests.test_ai_streaming_tool_events import _run_turn, _tool_events


class _FailedStart(InMemoryRealtime):
    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.TOOL_CALL_START:
            raise ConnectionError("realtime down")
        await super().publish_to_room(room_id, event)


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_failed_tool_start_does_not_stop_the_tool_or_its_end(
    streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []

    async def tool(name: str, _arguments: dict[str, Any]) -> str:
        calls.append(name)
        return "found"

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={})],
                finish_reason="tool_calls",
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    kit = RoomKit(realtime=_FailedStart())
    try:
        with caplog.at_level(logging.DEBUG):
            received = await _run_turn(kit, AIChannel("ai1", provider=provider, tool_handler=tool))
        starts, ends = _tool_events(received)
        assert calls == ["search"]
        assert starts == []
        assert len(ends) == 1
        assert ends[0].data["tool_calls"][0]["status"] == "completed"
        failures = [
            record
            for record in caplog.records
            if "a ephemeral could not be published" in record.message
        ]
        assert len(failures) == 1
        assert failures[0].levelno == logging.ERROR
    finally:
        await kit.close()
