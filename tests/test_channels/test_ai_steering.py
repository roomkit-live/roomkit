"""Tests for AIChannel steering: cancel, inject message, update system prompt."""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels.ai import AIChannel
from roomkit.models.steering import Cancel, InjectMessage, UpdateSystemPrompt
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AIToolCall,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider


def _tool_response(content: str = "", tool_name: str = "search") -> AIResponse:
    return AIResponse(
        content=content,
        tool_calls=[AIToolCall(id="tc1", name=tool_name, arguments={"q": "test"})],
    )


def _final_response(content: str = "Done") -> AIResponse:
    return AIResponse(content=content, tool_calls=[])


class TestSteerMethod:
    def test_steer_enqueues_directive(self) -> None:
        """steer() puts directive on the queue."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        directive = InjectMessage(content="hello")
        ch.steer(directive)

        assert not ch._steering_queue.empty()
        assert ch._steering_queue.get_nowait() is directive

    def test_steer_cancel_sets_event(self) -> None:
        """steer(Cancel) sets the cancel event."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ch.steer(Cancel(reason="user abort"))

        assert ch._cancel_event.is_set()
        assert not ch._steering_queue.empty()

    def test_steer_non_cancel_does_not_set_event(self) -> None:
        """Non-cancel directives don't set the cancel event."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ch.steer(InjectMessage(content="hi"))
        ch.steer(UpdateSystemPrompt(append=" extra"))

        assert not ch._cancel_event.is_set()


class TestDrainSteeringQueue:
    def test_drain_inject_message(self) -> None:
        """InjectMessage appends to context.messages."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ch.steer(InjectMessage(content="injected", role="user"))

        context = AIContext(messages=[AIMessage(role="user", content="original")])
        updated, cancelled = ch._drain_steering_queue(context)

        assert not cancelled
        assert len(updated.messages) == 2
        assert updated.messages[1].content == "injected"
        assert updated.messages[1].role == "user"

    def test_drain_cancel(self) -> None:
        """Cancel directive returns should_cancel=True."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ch.steer(Cancel(reason="abort"))

        context = AIContext(messages=[])
        _, cancelled = ch._drain_steering_queue(context)

        assert cancelled

    def test_drain_update_system_prompt(self) -> None:
        """UpdateSystemPrompt appends to system_prompt."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ch.steer(UpdateSystemPrompt(append="\nBe concise."))

        context = AIContext(messages=[], system_prompt="You are helpful.")
        updated, cancelled = ch._drain_steering_queue(context)

        assert not cancelled
        assert updated.system_prompt == "You are helpful.\nBe concise."

    def test_drain_multiple_directives(self) -> None:
        """Multiple directives are applied in order."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ch.steer(InjectMessage(content="msg1"))
        ch.steer(UpdateSystemPrompt(append=" extra"))
        ch.steer(InjectMessage(content="msg2"))

        context = AIContext(messages=[], system_prompt="base")
        updated, cancelled = ch._drain_steering_queue(context)

        assert not cancelled
        assert len(updated.messages) == 2
        assert updated.messages[0].content == "msg1"
        assert updated.messages[1].content == "msg2"
        assert updated.system_prompt == "base extra"

    def test_drain_empty_queue(self) -> None:
        """Draining an empty queue is a no-op."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        updated, cancelled = ch._drain_steering_queue(context)

        assert not cancelled
        assert len(updated.messages) == 1


class TestToolLoopCancellation:
    async def test_cancel_before_first_round(self) -> None:
        """Pre-queued cancel stops the tool loop immediately."""
        responses = [_tool_response()] * 5 + [_final_response()]
        provider = MockAIProvider(ai_responses=responses)
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        # Pre-queue cancel before loop starts
        ch.steer(Cancel(reason="preemptive"))

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        # First generate still runs (it happens before the for-loop),
        # but tool execution should be skipped
        assert handler.call_count == 0
        assert response is not None

    async def test_cancel_mid_loop(self) -> None:
        """Cancel injected during tool execution stops subsequent rounds."""
        call_count = 0

        async def handler_that_cancels(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Cancel during first tool execution
                ch.steer(Cancel(reason="mid-run"))
            return "ok"

        responses = [_tool_response()] * 10 + [_final_response()]
        provider = MockAIProvider(ai_responses=responses)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler_that_cancels,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        # Tool handler runs once, then drain catches the cancel
        assert call_count == 1
        assert response is not None

    async def test_inject_message_during_tool_loop(self) -> None:
        """Injected message appears in context for the next generate call."""
        generate_calls: list[AIContext] = []

        async def tracking_generate(context: AIContext) -> AIResponse:
            generate_calls.append(context)
            if len(generate_calls) == 1:
                return _tool_response()
            return _final_response("saw it")

        provider = MockAIProvider()
        provider.generate = tracking_generate  # type: ignore[assignment]

        call_count = 0

        async def handler_that_injects(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            ch.steer(InjectMessage(content="urgent update"))
            return "ok"

        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler_that_injects,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        assert response.content == "saw it"
        # Second generate call should have the injected message
        second_ctx = generate_calls[1]
        injected = [m for m in second_ctx.messages if m.content == "urgent update"]
        assert len(injected) == 1


class TestStreamingToolLoopCancellation:
    async def test_cancel_stops_streaming_loop(self) -> None:
        """Cancel event stops the streaming tool loop."""
        round_count = 0

        async def structured_stream(context):
            nonlocal round_count
            round_count += 1
            yield StreamTextDelta(text=f"round{round_count} ")
            yield StreamToolCall(id=f"tc{round_count}", name="search", arguments={"q": "x"})

        provider = MockAIProvider()
        provider.generate_structured_stream = structured_stream  # type: ignore[assignment]
        provider._supports_structured_streaming = True

        call_count = 0

        async def handler_that_cancels(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                ch.steer(Cancel(reason="stop"))
            return "ok"

        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler_that_cancels,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        chunks: list[str] = []
        async for chunk in ch._run_streaming_tool_loop(context):
            chunks.append(chunk)

        # First round streams text + executes tool, then cancel kicks in
        assert call_count == 1
        assert round_count <= 2  # at most one extra generate before cancel caught

    async def test_cancel_event_prequeued_streaming(self) -> None:
        """Pre-set cancel event stops streaming loop before first generate."""
        generate_called = False

        async def structured_stream(context):
            nonlocal generate_called
            generate_called = True
            yield StreamTextDelta(text="should not appear")

        provider = MockAIProvider()
        provider.generate_structured_stream = structured_stream  # type: ignore[assignment]
        provider._supports_structured_streaming = True

        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(return_value="ok"),
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        ch.steer(Cancel(reason="immediate"))

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        chunks: list[str] = []
        async for chunk in ch._run_streaming_tool_loop(context):
            chunks.append(chunk)

        assert not generate_called
        assert chunks == []
