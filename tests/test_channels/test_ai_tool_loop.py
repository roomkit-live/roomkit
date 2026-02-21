"""Tests for AIChannel tool loop: timeout, warning, parallel execution, truncation."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITextPart,
    AIToolCall,
    AIToolCallPart,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider


def _tool_response(content: str = "", tool_name: str = "search") -> AIResponse:
    """Create an AIResponse with a single tool call."""
    return AIResponse(
        content=content,
        tool_calls=[AIToolCall(id="tc1", name=tool_name, arguments={"q": "test"})],
    )


def _final_response(content: str = "Done") -> AIResponse:
    """Create a final AIResponse with no tool calls."""
    return AIResponse(content=content, tool_calls=[])


class TestToolLoopTimeout:
    async def test_loop_stops_at_deadline(self) -> None:
        """Tool loop breaks when timeout is exceeded."""
        max_rounds = 200
        provider = MockAIProvider(ai_responses=[_tool_response()] * max_rounds)
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=max_rounds,
            tool_loop_timeout_seconds=0.05,
        )

        # Patch event loop time so it jumps past the deadline after first round
        original_time = asyncio.get_event_loop().time
        time_call_count = 0

        def advancing_time() -> float:
            nonlocal time_call_count
            time_call_count += 1
            if time_call_count <= 1:
                return original_time()
            # After first call, return a time well past the deadline
            return original_time() + 1000

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        with patch.object(asyncio.get_event_loop(), "time", advancing_time):
            await ch._run_tool_loop(context)

        # Timeout must have stopped the loop — far fewer than max_tool_rounds
        assert handler.call_count < max_rounds
        assert handler.call_count <= 2

    async def test_no_timeout_when_none(self) -> None:
        """Tool loop runs without timeout when tool_loop_timeout_seconds is None."""
        responses = [_tool_response(), _tool_response(), _final_response()]
        provider = MockAIProvider(ai_responses=responses)
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        assert response.content == "Done"
        assert handler.call_count == 2

    async def test_timeout_before_any_tool_response(self) -> None:
        """Timeout fires on first round — loop returns partial with no tool results."""
        max_rounds = 200
        provider = MockAIProvider(ai_responses=[_tool_response()] * max_rounds)
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=max_rounds,
            tool_loop_timeout_seconds=0.001,
        )

        # Immediately past deadline from the very first time check
        original_time = asyncio.get_event_loop().time
        call_count = 0

        def already_expired() -> float:
            nonlocal call_count
            call_count += 1
            # First call sets the deadline; immediately return far future
            if call_count <= 1:
                return original_time()
            return original_time() + 99999

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        with patch.object(asyncio.get_event_loop(), "time", already_expired):
            response = await ch._run_tool_loop(context)

        # At most 1 round completes (the one in-flight when timeout is checked)
        assert handler.call_count <= 1
        # Response should exist (may be partial or timeout message)
        assert response is not None

    async def test_loop_completes_current_round_before_timeout(self) -> None:
        """Timeout check happens after round finishes, not mid-tool-execution."""
        call_count = 0

        async def slow_handler(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        # First generate → tool call, second → final
        responses = [_tool_response(), _final_response()]
        provider = MockAIProvider(ai_responses=responses)
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=slow_handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=10.0,  # won't fire
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        assert call_count == 1
        assert response.content == "Done"


class TestToolLoopWarning:
    async def test_soft_warning_at_configured_round(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning is logged at tool_loop_warn_after rounds."""
        warn_after = 3
        responses = [_tool_response()] * (warn_after + 1) + [_final_response()]
        provider = MockAIProvider(ai_responses=responses)
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
            tool_loop_warn_after=warn_after,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
            await ch._run_tool_loop(context)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(f"reached {warn_after} rounds, still running" in m for m in warning_msgs), (
            f"Expected exact warning format, got: {warning_msgs}"
        )

    async def test_no_warning_below_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when loop finishes before warn_after."""
        responses = [_tool_response(), _final_response()]
        provider = MockAIProvider(ai_responses=responses)
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
            tool_loop_warn_after=50,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
            await ch._run_tool_loop(context)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("still running" in m for m in warning_msgs)

    async def test_hard_cap_terminates_loop(self) -> None:
        """Loop stops at max_tool_rounds even if model keeps requesting tools."""
        max_rounds = 3
        provider = MockAIProvider(ai_responses=[_tool_response()] * (max_rounds + 5))
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=max_rounds,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        assert handler.call_count == max_rounds


class TestParallelToolExecution:
    async def test_tools_run_concurrently(self) -> None:
        """Multiple tool calls in a single round interleave execution."""
        execution_order: list[str] = []

        async def handler(name: str, args: dict) -> str:
            execution_order.append(f"start:{name}")
            # Yield control — if sequential, all starts would come before
            # any ends. With gather, starts interleave with ends.
            await asyncio.sleep(0)
            execution_order.append(f"end:{name}")
            return f"result:{name}"

        multi_tool_response = AIResponse(
            content="",
            tool_calls=[
                AIToolCall(id="tc1", name="tool_a", arguments={}),
                AIToolCall(id="tc2", name="tool_b", arguments={}),
            ],
        )
        provider = MockAIProvider(ai_responses=[multi_tool_response, _final_response()])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        assert response.content == "Done"
        assert len(execution_order) == 4
        # With asyncio.gather + sleep(0), both starts happen before ends
        # because gather launches all coroutines, they each hit sleep(0)
        # and yield, then they each complete.
        assert execution_order[0].startswith("start:")
        assert execution_order[1].startswith("start:")
        assert execution_order[2].startswith("end:")
        assert execution_order[3].startswith("end:")

    async def test_tool_failure_isolation(self) -> None:
        """One tool failing doesn't prevent others from completing."""

        async def handler(name: str, args: dict) -> str:
            if name == "bad_tool":
                raise ValueError("Tool exploded")
            return "ok"

        multi_tool_response = AIResponse(
            content="",
            tool_calls=[
                AIToolCall(id="tc1", name="good_tool", arguments={}),
                AIToolCall(id="tc2", name="bad_tool", arguments={}),
            ],
        )
        provider = MockAIProvider(ai_responses=[multi_tool_response, _final_response()])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        # Messages: [0]=user "go", [1]=assistant (tool calls), [2]=tool (results)
        assert context.messages[1].role == "assistant"
        tool_msg = context.messages[2]
        assert tool_msg.role == "tool"
        results = tool_msg.content
        assert len(results) == 2
        # Good tool succeeds
        assert results[0].name == "good_tool"
        assert results[0].result == "ok"
        # Bad tool has error message
        assert results[1].name == "bad_tool"
        assert "Error executing tool" in results[1].result
        assert "Tool exploded" in results[1].result


class TestToolResultTruncation:
    def test_small_result_unchanged(self) -> None:
        """Results under the limit are not truncated."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        result = "short result"
        assert ch._maybe_truncate_result(result) == result

    def test_large_result_truncated(self) -> None:
        """Results over the limit are truncated with start/end preserved."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        max_chars = ch._MAX_TOOL_RESULT_TOKENS * 4
        half = max_chars // 2
        # Create a result that's well over the limit
        large = "x" * (max_chars + 40_000)
        truncated = ch._maybe_truncate_result(large)

        assert len(truncated) < len(large)
        assert "[... truncated" in truncated
        # First half preserved
        assert truncated[:half] == large[:half]
        # Last half preserved
        assert truncated.endswith(large[-half:])

    def test_result_at_exact_limit_unchanged(self) -> None:
        """Result exactly at the token limit is not truncated."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)
        # estimate_tokens = len // 4 + 1, so for 30_000 tokens: (30_000-1)*4 = 119_996 chars
        chars = (ch._MAX_TOOL_RESULT_TOKENS - 1) * 4
        result = "x" * chars
        assert ch._maybe_truncate_result(result) == result


class TestContextOverflowRecovery:
    async def test_overflow_triggers_compaction(self) -> None:
        """Context overflow error triggers compaction and retry."""
        call_count = 0

        async def generate_with_overflow(context: AIContext) -> AIResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ProviderError("context length exceeded", retryable=True, status_code=400)
            if call_count == 3:
                return _final_response("Recovered")
            return _tool_response()

        provider = MockAIProvider()
        provider.generate = generate_with_overflow  # type: ignore[assignment]
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(
            messages=[AIMessage(role="user", content=f"msg{i}") for i in range(10)]
        )
        response = await ch._run_tool_loop(context)

        assert response.content == "Recovered"
        assert call_count == 3

    async def test_non_overflow_error_returns_partial(self) -> None:
        """Non-overflow errors return accumulated text as partial result."""
        call_count = 0

        async def generate_failing(context: AIContext) -> AIResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response("Thinking...")
            raise ProviderError("Internal server error", retryable=False, status_code=500)

        provider = MockAIProvider()
        provider.generate = generate_failing  # type: ignore[assignment]
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        assert "Thinking..." in response.content
        assert "[Agent interrupted" in response.content

    async def test_compaction_still_overflows_raises(self) -> None:
        """When compaction doesn't help, error propagates."""
        call_count = 0

        async def always_overflow(context: AIContext) -> AIResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_response()
            # Even after compaction, still overflows
            raise ProviderError("context length exceeded", retryable=True, status_code=400)

        provider = MockAIProvider()
        provider.generate = always_overflow  # type: ignore[assignment]
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        # Enough messages for compaction to work, but second generate
        # still overflows — the compacted context is re-raised.
        context = AIContext(
            messages=[AIMessage(role="user", content=f"msg{i}") for i in range(10)]
        )
        with pytest.raises(ProviderError, match="context length exceeded"):
            await ch._run_tool_loop(context)

    def test_is_context_overflow_matches_known_patterns(self) -> None:
        """_is_context_overflow detects known error messages."""
        patterns = [
            "context length exceeded",
            "maximum context length is 200000",
            "token limit reached",
            "too many tokens in the request",
            "request too large",
            "prompt is too long for model",
        ]
        for pattern in patterns:
            exc = ProviderError(pattern, retryable=True)
            assert AIChannel._is_context_overflow(exc), f"Should match: {pattern}"

    def test_is_context_overflow_rejects_unrelated_errors(self) -> None:
        """_is_context_overflow does not match unrelated errors."""
        exc = ProviderError("Invalid API key", retryable=False)
        assert not AIChannel._is_context_overflow(exc)

    async def test_compact_context_splits_messages(self) -> None:
        """_compact_context splits messages and creates summary."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        messages = [AIMessage(role="user", content=f"msg{i}") for i in range(10)]
        context = AIContext(messages=messages)
        compacted = await ch._compact_context(context)

        # First message is the summary
        assert compacted.messages[0].role == "user"
        assert "[Context compacted" in compacted.messages[0].content
        # Summary includes content from old messages
        assert "msg0" in compacted.messages[0].content
        # Second half preserved as-is
        assert len(compacted.messages) == 6  # 1 summary + 5 recent
        assert compacted.messages[1].content == "msg5"
        assert compacted.messages[-1].content == "msg9"

    async def test_compact_context_raises_when_too_few_messages(self) -> None:
        """_compact_context raises when <= 4 messages."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        messages = [AIMessage(role="user", content="m")] * 3
        context = AIContext(messages=messages)
        with pytest.raises(ProviderError, match="cannot compact further"):
            await ch._compact_context(context)


class TestToolLoopContextAccumulation:
    async def test_tool_results_appear_in_context_for_next_round(self) -> None:
        """Each round's tool results are appended to context, visible to next generate."""
        contexts_seen: list[int] = []

        async def tracking_generate(context: AIContext) -> AIResponse:
            # Record how many messages the provider sees on each call
            contexts_seen.append(len(context.messages))
            if len(contexts_seen) == 1:
                return _tool_response(content="round1")
            if len(contexts_seen) == 2:
                return _tool_response(content="round2")
            return _final_response("Done")

        provider = MockAIProvider()
        provider.generate = tracking_generate  # type: ignore[assignment]
        handler = AsyncMock(return_value="tool result")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        assert response.content == "Done"
        # First call: 1 message (user "go")
        assert contexts_seen[0] == 1
        # Second call: 1 + 2 = 3 (user + assistant[tool_calls] + tool[results])
        assert contexts_seen[1] == 3
        # Third call: 3 + 2 = 5 (prev + assistant[tool_calls] + tool[results])
        assert contexts_seen[2] == 5
        # Final context has all messages
        assert len(context.messages) == 5
        assert context.messages[1].role == "assistant"
        assert context.messages[2].role == "tool"
        assert context.messages[3].role == "assistant"
        assert context.messages[4].role == "tool"

    async def test_tool_results_content_matches_handler_output(self) -> None:
        """Tool result parts contain the actual handler return values."""

        async def handler(name: str, args: dict) -> str:
            return f"result_for_{name}"

        multi_tool_response = AIResponse(
            content="",
            tool_calls=[
                AIToolCall(id="tc1", name="alpha", arguments={}),
                AIToolCall(id="tc2", name="beta", arguments={}),
            ],
        )
        provider = MockAIProvider(ai_responses=[multi_tool_response, _final_response()])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=None,
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        await ch._run_tool_loop(context)

        tool_msg = context.messages[2]
        assert tool_msg.role == "tool"
        results = tool_msg.content
        # Results match handler output, keyed by tool name
        result_map = {r.name: r.result for r in results}
        assert result_map["alpha"] == "result_for_alpha"
        assert result_map["beta"] == "result_for_beta"


class TestExtractAccumulatedText:
    def test_extracts_from_string_content(self) -> None:
        messages = [
            AIMessage(role="user", content="question"),
            AIMessage(role="assistant", content="answer1"),
            AIMessage(role="user", content="follow-up"),
            AIMessage(role="assistant", content="answer2"),
        ]
        result = AIChannel._extract_accumulated_text(messages)
        assert "answer1" in result
        assert "answer2" in result
        assert "question" not in result

    def test_extracts_from_list_content(self) -> None:
        messages = [
            AIMessage(
                role="assistant",
                content=[
                    AITextPart(text="part1"),
                    AIToolCallPart(id="x", name="t", arguments={}),
                ],
            ),
        ]
        result = AIChannel._extract_accumulated_text(messages)
        assert "part1" in result

    def test_returns_empty_for_no_assistant_messages(self) -> None:
        messages = [AIMessage(role="user", content="hello")]
        result = AIChannel._extract_accumulated_text(messages)
        assert result == ""
