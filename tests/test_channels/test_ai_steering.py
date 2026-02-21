"""Tests for AIChannel steering: cancel, inject message, update system prompt."""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels.ai import AIChannel, _ToolLoopContext
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


def _register_loop(ch: AIChannel) -> _ToolLoopContext:
    """Register a _ToolLoopContext so steer() has an active loop to target."""
    ctx = _ToolLoopContext(loop_id="test-loop")
    ch._active_loops["test-loop"] = ctx
    return ctx


class TestSteerMethod:
    def test_steer_enqueues_directive(self) -> None:
        """steer() puts directive on the queue of the active loop."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _register_loop(ch)

        directive = InjectMessage(content="hello")
        ch.steer(directive)

        assert not loop_ctx.steering_queue.empty()
        assert loop_ctx.steering_queue.get_nowait() is directive

    def test_steer_cancel_sets_event(self) -> None:
        """steer(Cancel) sets the cancel event on the active loop."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _register_loop(ch)

        ch.steer(Cancel(reason="user abort"))

        assert loop_ctx.cancel_event.is_set()
        assert not loop_ctx.steering_queue.empty()

    def test_steer_non_cancel_does_not_set_event(self) -> None:
        """Non-cancel directives don't set the cancel event."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _register_loop(ch)

        ch.steer(InjectMessage(content="hi"))
        ch.steer(UpdateSystemPrompt(append=" extra"))

        assert not loop_ctx.cancel_event.is_set()

    def test_steer_no_active_loop_warns(self) -> None:
        """steer() with no active loop logs a warning and is a no-op."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        # No active loops — should not raise, just warn
        ch.steer(InjectMessage(content="lost"))

    def test_steer_targets_specific_loop(self) -> None:
        """steer() with explicit loop_id targets that specific loop."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ctx1 = _ToolLoopContext(loop_id="loop-1")
        ctx2 = _ToolLoopContext(loop_id="loop-2")
        ch._active_loops["loop-1"] = ctx1
        ch._active_loops["loop-2"] = ctx2

        ch.steer(InjectMessage(content="for loop 1"), loop_id="loop-1")

        assert not ctx1.steering_queue.empty()
        assert ctx2.steering_queue.empty()

    def test_steer_defaults_to_latest_loop(self) -> None:
        """steer() without loop_id targets the most recently added loop."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        ctx1 = _ToolLoopContext(loop_id="loop-1")
        ctx2 = _ToolLoopContext(loop_id="loop-2")
        ch._active_loops["loop-1"] = ctx1
        ch._active_loops["loop-2"] = ctx2

        ch.steer(InjectMessage(content="for latest"))

        assert ctx1.steering_queue.empty()
        assert not ctx2.steering_queue.empty()


class TestDrainSteeringQueue:
    def test_drain_inject_message(self) -> None:
        """InjectMessage appends to context.messages."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _ToolLoopContext()

        loop_ctx.steering_queue.put_nowait(InjectMessage(content="injected", role="user"))

        context = AIContext(messages=[AIMessage(role="user", content="original")])
        updated, cancelled = ch._drain_steering_queue(context, loop_ctx)

        assert not cancelled
        assert len(updated.messages) == 2
        assert updated.messages[1].content == "injected"
        assert updated.messages[1].role == "user"

    def test_drain_cancel(self) -> None:
        """Cancel directive returns should_cancel=True."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _ToolLoopContext()

        loop_ctx.steering_queue.put_nowait(Cancel(reason="abort"))

        context = AIContext(messages=[])
        _, cancelled = ch._drain_steering_queue(context, loop_ctx)

        assert cancelled

    def test_drain_update_system_prompt(self) -> None:
        """UpdateSystemPrompt appends to system_prompt."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _ToolLoopContext()

        loop_ctx.steering_queue.put_nowait(UpdateSystemPrompt(append="\nBe concise."))

        context = AIContext(messages=[], system_prompt="You are helpful.")
        updated, cancelled = ch._drain_steering_queue(context, loop_ctx)

        assert not cancelled
        assert updated.system_prompt == "You are helpful.\nBe concise."

    def test_drain_multiple_directives(self) -> None:
        """Multiple directives are applied in order."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _ToolLoopContext()

        loop_ctx.steering_queue.put_nowait(InjectMessage(content="msg1"))
        loop_ctx.steering_queue.put_nowait(UpdateSystemPrompt(append=" extra"))
        loop_ctx.steering_queue.put_nowait(InjectMessage(content="msg2"))

        context = AIContext(messages=[], system_prompt="base")
        updated, cancelled = ch._drain_steering_queue(context, loop_ctx)

        assert not cancelled
        assert len(updated.messages) == 2
        assert updated.messages[0].content == "msg1"
        assert updated.messages[1].content == "msg2"
        assert updated.system_prompt == "base extra"

    def test_drain_empty_queue(self) -> None:
        """Draining an empty queue is a no-op."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        loop_ctx = _ToolLoopContext()

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        updated, cancelled = ch._drain_steering_queue(context, loop_ctx)

        assert not cancelled
        assert len(updated.messages) == 1


class TestToolLoopCancellation:
    async def test_cancel_before_first_round(self) -> None:
        """Pre-queued cancel stops the tool loop immediately.

        We register a loop before calling _run_tool_loop so the steer() call
        targets the pre-registered context, then _run_tool_loop creates its own
        context and drains nothing — but the tool handler is never called because
        the first generate returns tool calls and the loop inherits the cancel.
        """
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

        # We need to steer after the tool loop creates its context.
        # Use a generate hook to steer during the first generate call.
        original_generate = provider.generate

        async def generate_with_cancel(context: AIContext) -> AIResponse:
            # On first call, steer cancel into the now-active loop
            if not hasattr(generate_with_cancel, "_called"):
                generate_with_cancel._called = True  # type: ignore[attr-defined]
                ch.steer(Cancel(reason="preemptive"))
            return await original_generate(context)

        provider.generate = generate_with_cancel  # type: ignore[assignment]

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        # Tool execution should be skipped because cancel was set
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
        """Cancel injected during first stream iteration stops loop after round 1."""
        generate_called_count = 0

        async def structured_stream(context):
            nonlocal generate_called_count
            generate_called_count += 1
            if generate_called_count == 1:
                yield StreamTextDelta(text="round1 ")
                yield StreamToolCall(id="tc1", name="search", arguments={"q": "x"})
            else:
                yield StreamTextDelta(text="should not appear")

        provider = MockAIProvider()
        provider.generate_structured_stream = structured_stream  # type: ignore[assignment]
        provider._supports_structured_streaming = True

        async def handler_that_cancels(name: str, args: dict) -> str:
            ch.steer(Cancel(reason="immediate"))
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

        # The cancel is injected during the tool handler of the first round,
        # so the loop exits after round 1 (either via cancel_event check or drain).
        assert generate_called_count <= 2


class TestToolLoopContextIsolation:
    async def test_loop_context_cleaned_up_after_run(self) -> None:
        """_active_loops is cleaned up after _run_tool_loop completes."""
        provider = MockAIProvider(ai_responses=[_final_response()])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(return_value="ok"),
            max_tool_rounds=5,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        assert len(ch._active_loops) == 0

    async def test_streaming_loop_context_cleaned_up(self) -> None:
        """_active_loops is cleaned up after _run_streaming_tool_loop completes."""

        async def structured_stream(context):
            yield StreamTextDelta(text="done")

        provider = MockAIProvider()
        provider.generate_structured_stream = structured_stream  # type: ignore[assignment]
        provider._supports_structured_streaming = True

        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=AsyncMock(return_value="ok"),
            max_tool_rounds=5,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        async for _ in ch._run_streaming_tool_loop(context):
            pass

        assert len(ch._active_loops) == 0
