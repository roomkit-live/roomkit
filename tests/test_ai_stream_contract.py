"""Observable streaming contracts that survive changes to internal components."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.channels.ai import AIChannel
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.models.streaming import LoopEndMarker, ThinkingDeltaMarker, ToolCallStartMarker
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import (
    AIContext,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_tool_room_id
from tests.conftest import make_event
from tests.test_ai_streaming_tool_loop import _binding, _ctx


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["Working.", "Done."], "Done."),
        (["Work", "ing.", "Done."], "Done."),
        (["Working.Done."], "Done."),
        (list("Working.Done."), "Done."),
        (["Work"], "Work"),
        (["Work", " differs"], "Work differs"),
        (["Working."], "Working."),
        (["Done."], "Done."),
    ],
)
async def test_prefix_filter_preserves_delivery_and_transcript(
    chunks: list[str], expected: str
) -> None:
    reports: list[AIResponseEvent] = []

    class Provider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            if not any(message.role == "tool" for message in context.messages):
                yield StreamTextDelta(text="Working.")
                yield StreamToolCall(id="call", name="search", arguments={})
            else:
                for chunk in chunks:
                    yield StreamTextDelta(text=chunk)
            yield StreamDone(finish_reason="stop")

    async def handler(name: str, arguments: dict) -> str:
        return "result"

    async def report(event: AIResponseEvent) -> None:
        reports.append(event)

    channel = AIChannel("ai1", provider=Provider(streaming=True), tool_handler=handler)
    channel._after_response_hook = report
    output = await channel.on_event(make_event(), _binding(), _ctx())
    assert output.response_stream is not None
    items = [item async for item in output.response_stream]
    assert "".join(item for item in items if isinstance(item, str)) == "Working." + expected
    assert reports[0].segments == ["Working.", expected]
    assert len([item for item in items if isinstance(item, LoopEndMarker)]) == 1


async def test_consumer_controls_progress_and_joins_async_finalizer() -> None:
    produced: list[str] = []
    closing = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class Provider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            try:
                for text in ("first", "second"):
                    produced.append(text)
                    yield StreamTextDelta(text=text)
            finally:
                closing.set()
                await release.wait()
                closed.set()

    channel = AIChannel("ai1", provider=Provider(streaming=True))
    output = await channel.on_event(make_event(), _binding(), _ctx())
    stream = output.response_stream
    assert stream is not None
    assert produced == []
    assert await anext(stream) == "first"
    assert produced == ["first"]
    task = asyncio.create_task(stream.aclose())
    try:
        await asyncio.wait_for(closing.wait(), 2)
        assert not task.done()
        assert channel.active_turns == 1
        release.set()
        await asyncio.wait_for(task, 2)
        assert closed.is_set()
        assert channel.active_turns == 0
        assert produced == ["first"]
    finally:
        release.set()
        await task


async def test_reasoning_composition_and_execution_keep_their_order() -> None:
    trace: list[str] = []

    class Provider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            if not any(message.role == "tool" for message in context.messages):
                yield StreamThinkingDelta(thinking="thought")
                yield StreamToolCallDelta(id="call", name="search", arguments_delta="{}")
                yield StreamToolCall(id="call", name="search", arguments={})
            else:
                yield StreamTextDelta(text="done")
            yield StreamDone(finish_reason="stop")

    async def handler(name: str, arguments: dict) -> str:
        trace.append("execute")
        return "result"

    async def publish_thinking(event_type, room_id, thinking, round_idx) -> None:
        trace.append(str(event_type))

    async def publish_tool(event_type, room_id, calls, round_idx, **kwargs) -> None:
        trace.append(str(event_type) + (":empty" if not calls else ""))

    channel = AIChannel(
        "ai1", provider=Provider(streaming=True), tool_handler=handler, thinking_coalesce_ms=0
    )
    channel._publish_thinking_event = publish_thinking
    channel._publish_tool_event = publish_tool
    output = await channel.on_event(make_event(), _binding(), _ctx())
    assert output.response_stream is not None
    async for item in output.response_stream:
        if isinstance(item, ThinkingDeltaMarker):
            trace.append("thinking_marker")
        elif isinstance(item, ToolCallStartMarker):
            trace.append("tool_start_marker")
    assert trace == [
        "thinking_start",
        "thinking_delta",
        "thinking_marker",
        "thinking_end",
        "tool_call_delta",
        "tool_call_delta:empty",
        "tool_start_marker",
        "tool_call_start",
        "execute",
        "tool_call_end",
    ]


async def test_task_cancellation_closes_an_open_tool_composition() -> None:
    composing = asyncio.Event()
    publications: list[list] = []

    class Provider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            yield StreamToolCallDelta(id="call", name="search", arguments_delta="{")
            composing.set()
            await asyncio.Future()

    async def publish_tool(event_type, room_id, calls, round_idx, **kwargs) -> None:
        publications.append(calls)

    channel = AIChannel("ai1", provider=Provider(streaming=True))
    channel._publish_tool_event = publish_tool
    output = await channel.on_event(make_event(), _binding(), _ctx())
    assert output.response_stream is not None

    async def consume() -> None:
        async for _ in output.response_stream:
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(composing.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(publications) == 2
        assert publications[-1] == []
        assert channel.active_turns == 0
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_interleaved_rounds_keep_room_state_and_usage_separate() -> None:
    ready: set[str] = set()
    both_started = asyncio.Event()
    reports: dict[str, AIResponseEvent] = {}
    executions: list[tuple[str | None, str]] = []

    class Provider(MockAIProvider):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            assert context.room is not None
            room = context.room.room.id
            if not any(message.role == "tool" for message in context.messages):
                yield StreamTextDelta(text=f"Working {room}.")
                ready.add(room)
                if len(ready) == 2:
                    both_started.set()
                await both_started.wait()
                yield StreamToolCall(id=room, name="search", arguments={"room": room})
            else:
                yield StreamTextDelta(text=f"Working {room}.Done {room}.")
            yield StreamDone(
                finish_reason="stop", usage={"input_tokens": 10 if room == "a" else 20}
            )

    async def handler(name: str, arguments: dict) -> str:
        executions.append((current_tool_room_id(), arguments["room"]))
        return "result"

    async def report(event: AIResponseEvent) -> None:
        assert event.room_id is not None
        reports[event.room_id] = event

    channel = AIChannel("ai1", provider=Provider(streaming=True), tool_handler=handler)
    channel._after_response_hook = report

    async def run(room: str) -> str:
        output = await channel.on_event(
            make_event(room_id=room),
            _binding().model_copy(update={"room_id": room}),
            RoomContext(room=Room(id=room)),
        )
        assert output.response_stream is not None
        return "".join([item async for item in output.response_stream if isinstance(item, str)])

    # The deadline measures the overlapping turns, not cold pattern compilation.
    await compile_mark_patterns()
    result = await asyncio.wait_for(asyncio.gather(run("a"), run("b")), 2)
    assert result == ["Working a.Done a.", "Working b.Done b."]
    assert sorted(executions) == [("a", "a"), ("b", "b")]
    for room, tokens in (("a", 20), ("b", 40)):
        assert reports[room].segments == [f"Working {room}.", f"Done {room}."]
        assert reports[room].usage["input_tokens"] == tokens
        assert reports[room].tool_calls_count == reports[room].round_count == 1
    assert channel.active_turns == 0
