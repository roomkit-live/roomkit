"""Tests for the streaming tool loop in AIChannel."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIResponse,
    AIToolCall,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.realtime.memory import InMemoryRealtime
from tests.conftest import make_event


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "tools": [{"name": "search", "description": "Search"}],
        },
    )


def _binding_no_tools() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


class TestStreamingToolLoop:
    """Test the streaming tool loop in AIChannel."""

    async def test_single_tool_round(self) -> None:
        """Provider returns tool call on round 1, text on round 2."""

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return f"Result for {name}"

        # Round 1: tool call, Round 2: text
        responses = [
            AIResponse(
                content="Let me search.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={"q": "test"}),
                ],
            ),
            AIResponse(
                content="Here are the results.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=tool_handler,
        )

        output = await ch.on_event(
            make_event(body="search for test", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.responded is True
        assert output.response_stream is not None

        # Collect all text deltas
        chunks = [chunk async for chunk in output.response_stream]
        text = "".join(chunks)

        # Both rounds' text should be yielded
        assert "Let me search." in text
        assert "Here are the results." in text

        # Provider was called twice (two rounds)
        assert len(provider.calls) == 2

    async def test_progressive_text_delivery(self) -> None:
        """Text from round 1 is yielded before tool execution happens."""
        execution_order: list[str] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            execution_order.append("tool_executed")
            return "done"

        responses = [
            AIResponse(
                content="Thinking...",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={}),
                ],
            ),
            AIResponse(
                content="Final answer.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        output = await ch.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None

        # Consume stream, tracking when text arrives vs tools execute
        chunks: list[str] = []
        async for chunk in output.response_stream:
            if not execution_order:
                # Tool hasn't been called yet — this text is from round 1
                chunks.append(f"pre_tool:{chunk}")
            else:
                chunks.append(f"post_tool:{chunk}")

        # Round 1 text arrived before tool execution
        pre_tool_text = [c for c in chunks if c.startswith("pre_tool:")]
        assert len(pre_tool_text) > 0
        assert "Thinking..." in "".join(c.split(":", 1)[1] for c in pre_tool_text)

    async def test_no_tools_single_round(self) -> None:
        """Without tools, stream completes after one generation."""
        provider = MockAIProvider(responses=["Just text."], streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(
            make_event(body="hi", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None
        chunks = [chunk async for chunk in output.response_stream]
        assert "".join(chunks) == "Just text."
        assert len(provider.calls) == 1

    async def test_no_tools_no_handler(self) -> None:
        """Tool calls without a handler end the loop after round 1."""
        responses = [
            AIResponse(
                content="I want to call tools but can't.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={}),
                ],
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        # No tool_handler
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None
        chunks = [chunk async for chunk in output.response_stream]
        assert "".join(chunks) == "I want to call tools but can't."
        # Only one round because no handler
        assert len(provider.calls) == 1

    async def test_max_rounds_honored(self) -> None:
        """Loop stops at max_tool_rounds even if provider keeps returning tools."""
        tool_executions = 0

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            nonlocal tool_executions
            tool_executions += 1
            return "ok"

        # Every call returns a tool call
        always_tool = AIResponse(
            content="",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[AIToolCall(id="tc1", name="search", arguments={})],
        )

        provider = MockAIProvider(ai_responses=[always_tool], streaming=True)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=tool_handler,
            max_tool_rounds=3,
        )

        output = await ch.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None
        async for _ in output.response_stream:
            pass

        # max_tool_rounds=3 → 4 generations (0,1,2,3) but only 3 tool executions
        # The last generation sees tool calls but does NOT execute them (no
        # generation would follow to use the results).
        assert len(provider.calls) == 4
        assert tool_executions == 3

    async def test_tool_execution_error_fed_back_to_llm(self) -> None:
        """Tool errors are fed back as tool results instead of propagating."""

        async def broken_handler(name: str, args: dict[str, Any]) -> str:
            raise RuntimeError("Tool failed!")

        responses = [
            AIResponse(
                content="Calling tool.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={}),
                ],
            ),
            AIResponse(
                content="The tool failed, let me explain.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=broken_handler)

        output = await ch.on_event(
            make_event(body="go", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None
        collected = []
        async for delta in output.response_stream:
            collected.append(delta)
        # Should get both rounds of text (tool error handled gracefully)
        assert len(collected) > 0

    async def test_context_updated_between_rounds(self) -> None:
        """Verify tool results are appended to context between rounds."""

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return "42"

        responses = [
            AIResponse(
                content="Checking.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="calculate", arguments={"x": "6*7"}),
                ],
            ),
            AIResponse(
                content="The answer is 42.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        output = await ch.on_event(
            make_event(body="what is 6*7?", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None
        async for _ in output.response_stream:
            pass

        # Second call should have assistant + tool messages appended
        second_ctx = provider.calls[1]
        roles = [m.role for m in second_ctx.messages]
        assert "assistant" in roles
        assert "tool" in roles


class TestDefaultFallback:
    """AIProvider.generate_structured_stream() default wraps generate()."""

    async def test_default_wraps_generate(self) -> None:
        """Provider without override returns events from generate() result."""
        provider = MockAIProvider(responses=["Hello world"])

        from roomkit.providers.ai.base import AIContext, AIMessage

        ctx = AIContext(messages=[AIMessage(role="user", content="hi")])

        events: list[StreamEvent] = []
        async for ev in provider.generate_structured_stream(ctx):
            events.append(ev)

        assert len(events) == 2  # text delta + done
        assert isinstance(events[0], StreamTextDelta)
        assert events[0].text == "Hello world"
        assert isinstance(events[1], StreamDone)
        assert events[1].finish_reason == "stop"

    async def test_default_wraps_generate_with_tool_calls(self) -> None:
        """Default fallback includes tool calls from generate()."""
        ai_resp = AIResponse(
            content="Let me check.",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[
                AIToolCall(id="tc1", name="search", arguments={"q": "test"}),
            ],
        )
        provider = MockAIProvider(ai_responses=[ai_resp])

        from roomkit.providers.ai.base import AIContext, AIMessage

        ctx = AIContext(messages=[AIMessage(role="user", content="search")])

        events: list[StreamEvent] = []
        async for ev in provider.generate_structured_stream(ctx):
            events.append(ev)

        assert len(events) == 3  # text delta + tool call + done
        assert isinstance(events[0], StreamTextDelta)
        assert events[0].text == "Let me check."
        assert isinstance(events[1], StreamToolCall)
        assert events[1].name == "search"
        assert isinstance(events[2], StreamDone)


class TestStreamingWithNoToolsBinding:
    """Streaming provider without tools still uses simple streaming path."""

    async def test_no_tools_uses_streaming_response(self) -> None:
        provider = MockAIProvider(responses=["streamed"], streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(
            make_event(body="hi", channel_id="sms1"),
            _binding_no_tools(),
            _ctx(),
        )

        assert output.response_stream is not None
        chunks = [chunk async for chunk in output.response_stream]
        assert "".join(chunks) == "streamed"

    async def test_tools_uses_streaming_tool_loop(self) -> None:
        provider = MockAIProvider(responses=["with tools"], streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(
            make_event(body="hi", channel_id="sms1"),
            _binding(),  # has tools
            _ctx(),
        )

        assert output.response_stream is not None
        chunks = [chunk async for chunk in output.response_stream]
        assert "".join(chunks) == "with tools"


class TestToolCallEphemeralEvents:
    """Tool calls emit ephemeral events instead of inline XML."""

    async def test_tool_calls_emit_ephemeral_events(self) -> None:
        """Streaming tool loop publishes START/END events and no XML in text."""

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return f"Result for {name}"

        responses = [
            AIResponse(
                content="Let me search.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="search", arguments={"q": "test"}),
                ],
            ),
            AIResponse(
                content="Here are the results.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        provider = MockAIProvider(ai_responses=responses, streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        # Inject InMemoryRealtime so ephemeral events are captured
        realtime = InMemoryRealtime()
        ch._realtime = realtime

        received: list[EphemeralEvent] = []

        async def on_event(event: EphemeralEvent) -> None:
            received.append(event)

        await realtime.subscribe_to_room("r1", on_event)

        output = await ch.on_event(
            make_event(body="search for test", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        assert output.response_stream is not None
        chunks = [chunk async for chunk in output.response_stream]
        text = "".join(chunks)

        # Allow events to propagate
        await asyncio.sleep(0.02)

        # Text should contain both rounds of text
        assert "Let me search." in text
        assert "Here are the results." in text

        # No XML fragments in the stream
        assert "<invoke" not in text
        assert "<result>" not in text

        # Exactly one START and one END event
        starts = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_START]
        ends = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_END]
        assert len(starts) == 1
        assert len(ends) == 1

        # Validate START payload
        assert starts[0].data["tool_calls"][0]["name"] == "search"
        assert starts[0].data["tool_calls"][0]["id"] == "tc1"
        assert starts[0].data["round"] == 0
        assert starts[0].data["channel_id"] == "ai1"

        # Validate END payload
        assert ends[0].data["tool_calls"][0]["id"] == "tc1"
        assert ends[0].data["tool_calls"][0]["name"] == "search"
        assert "duration_ms" in ends[0].data
        assert ends[0].data["duration_ms"] >= 0

        await realtime.close()

    async def test_non_streaming_emits_tool_events(self) -> None:
        """Non-streaming tool loop also publishes START/END events."""

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return f"Result for {name}"

        responses = [
            AIResponse(
                content="Checking.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[
                    AIToolCall(id="tc1", name="calculate", arguments={"x": "6*7"}),
                ],
            ),
            AIResponse(
                content="The answer is 42.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]

        # Non-streaming provider
        provider = MockAIProvider(ai_responses=responses, streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        realtime = InMemoryRealtime()
        ch._realtime = realtime

        received: list[EphemeralEvent] = []

        async def on_event(event: EphemeralEvent) -> None:
            received.append(event)

        await realtime.subscribe_to_room("r1", on_event)

        output = await ch.on_event(
            make_event(body="what is 6*7?", channel_id="sms1"),
            _binding(),
            _ctx(),
        )

        # Allow events to propagate
        await asyncio.sleep(0.02)

        assert output.responded is True

        # Exactly one START and one END event
        starts = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_START]
        ends = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_END]
        assert len(starts) == 1
        assert len(ends) == 1

        # Validate payloads
        assert starts[0].data["tool_calls"][0]["name"] == "calculate"
        assert ends[0].data["duration_ms"] >= 0

        await realtime.close()
