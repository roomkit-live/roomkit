"""Public composition contracts shared with the Rust implementation (RFC §6.4)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from roomkit import AIChannel, RoomKit
from roomkit.models.steering import Cancel
from roomkit.providers.ai import (
    AIContext,
    AIResponse,
    AIToolCall,
    MockAIProvider,
    StreamDone,
    StreamEvent,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from tests.test_ai_streaming_tool_events import _run_turn


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("chunks", [-1, 0, 1, 4, 100])
async def test_mock_fragments_keep_python_json_and_complete_call(
    streaming: bool, chunks: int
) -> None:
    arguments = {"q": "é猫🙂"}
    provider = MockAIProvider(
        streaming=streaming,
        tool_call_delta_chunks=chunks,
        ai_responses=[
            AIResponse(
                content="", tool_calls=[AIToolCall(id="c1", name="lookup", arguments=arguments)]
            )
        ],
    )
    events = [event async for event in provider.generate_structured_stream(AIContext())]
    deltas = [event for event in events if isinstance(event, StreamToolCallDelta)]
    if streaming and chunks > 0:
        payload = json.dumps(arguments)
        size = max(1, -(-len(payload) // chunks))
        assert [event.arguments_delta for event in deltas] == [
            payload[i : i + size] for i in range(0, len(payload), size)
        ]
        assert all((event.id, event.name, event.index) == ("c1", "lookup", 0) for event in deltas)
    else:
        assert deltas == []
    calls = [event for event in events if isinstance(event, StreamToolCall)]
    assert len(calls) == 1
    assert calls[0].arguments == arguments


@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_composition_closes_on_the_public_bus(cancel: bool) -> None:
    ready, release = asyncio.Event(), asyncio.Event()

    class Provider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            yield StreamThinkingDelta(thinking="searching")
            yield StreamToolCallDelta(id="c1", name="lookup", arguments_delta="SECRETé")
            ready.set()
            await release.wait()
            if cancel:
                yield StreamToolCallDelta(id="c1", name="lookup", arguments_delta="unread")
            yield StreamToolCall(id="c1", name="lookup", arguments={})
            yield StreamDone(finish_reason="tool_calls")

    kit = RoomKit()
    ai = AIChannel(
        "ai1",
        provider=Provider(streaming=True),
        thinking_coalesce_ms=0,
        tool_loop_timeout_seconds=5 if cancel else 0.05,
    )
    turn = asyncio.create_task(_run_turn(kit, ai))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        if cancel:
            ai.steer(Cancel(), room_id="r1")
        else:
            # The loop budget is checked at the round boundary in both implementations.
            await asyncio.sleep(0.1)
        release.set()
        received = await asyncio.wait_for(turn, 2)
        events = [
            event for event in received if event.type.value.startswith(("thinking_", "tool_call_"))
        ]
        assert [event.type.value for event in events] == [
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "tool_call_delta",
            "tool_call_delta",
        ]
        assert events[-2].data == {
            "tool_calls": [{"id": "c1", "name": "lookup", "arguments_chars": 7}],
            "round": 0,
            "channel_id": "ai1",
        }
        assert events[-1].data == {"tool_calls": [], "round": 0, "channel_id": "ai1"}
        assert "SECRET" not in json.dumps([event.data for event in events])
    finally:
        release.set()
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await kit.close()


@pytest.mark.parametrize("later_thinking", [False, True])
async def test_abandoned_composition_closes_on_the_public_bus(later_thinking: bool) -> None:
    opened, closed = asyncio.Event(), asyncio.Event()
    received: list[EphemeralEvent] = []

    class Provider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            yield StreamToolCallDelta(id="c1", name="lookup", arguments_delta="SECRETé")
            if later_thinking:
                yield StreamThinkingDelta(thinking="late reasoning")
            await asyncio.Future()

    async def record(event: EphemeralEvent) -> None:
        if event.type.value.startswith(("thinking_", "tool_call_")):
            received.append(event)
        if later_thinking and event.type == EphemeralEventType.THINKING_DELTA:
            opened.set()
        if event.type == EphemeralEventType.TOOL_CALL_DELTA:
            if not event.data["tool_calls"]:
                closed.set()
            elif not later_thinking:
                opened.set()

    kit = RoomKit()
    ai = AIChannel("ai1", provider=Provider(streaming=True), thinking_coalesce_ms=0)
    await kit.realtime.subscribe_to_room("r1", record)
    turn = asyncio.create_task(_run_turn(kit, ai))
    try:
        await asyncio.wait_for(opened.wait(), 2)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await asyncio.wait_for(closed.wait(), 2)
        await kit.close()
        expected = ["tool_call_delta"]
        if later_thinking:
            expected += ["thinking_start", "thinking_delta", "thinking_end"]
        assert [event.type.value for event in received] == expected + ["tool_call_delta"]
        assert [
            event.data for event in received if event.type == EphemeralEventType.TOOL_CALL_DELTA
        ] == [
            {
                "tool_calls": [{"id": "c1", "name": "lookup", "arguments_chars": 7}],
                "round": 0,
                "channel_id": "ai1",
            },
            {"tool_calls": [], "round": 0, "channel_id": "ai1"},
        ]
    finally:
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await kit.close()
