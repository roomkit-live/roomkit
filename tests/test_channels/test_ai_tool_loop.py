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
        # Provider always returns tool calls — only timeout can stop it
        provider = MockAIProvider(ai_responses=[_tool_response()] * 100)
        handler = AsyncMock(return_value="ok")
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            max_tool_rounds=200,
            tool_loop_timeout_seconds=0.05,  # very short timeout
        )

        # Patch event loop time so it jumps past the deadline after first round
        original_time = asyncio.get_event_loop().time
        call_count = 0

        def advancing_time() -> float:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return original_time()
            # After first call, return a time well past the deadline
            return original_time() + 1000

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        with patch.object(asyncio.get_event_loop(), "time", advancing_time):
            await ch._run_tool_loop(context)

        # Should have stopped early due to timeout (not all 200 rounds)
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
            tool_loop_timeout_seconds=None,  # no timeout
        )

        context = AIContext(messages=[AIMessage(role="user", content="go")])
        response = await ch._run_tool_loop(context)

        assert response.content == "Done"
        assert handler.call_count == 2

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
        # Build responses: warn_after tool rounds + 1 final
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
        assert any(f"reached {warn_after} rounds" in m for m in warning_msgs)

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
        assert not any("reached" in m and "rounds" in m for m in warning_msgs)

    async def test_hard_cap_terminates_loop(self) -> None:
        """Loop stops at max_tool_rounds even if model keeps requesting tools."""
        max_rounds = 3
        # All responses have tool calls — only cap stops the loop
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

        # max_tool_rounds tool executions + initial generate + max_tool_rounds re-generates
        assert handler.call_count == max_rounds


class TestParallelToolExecution:
    async def test_tools_run_concurrently(self) -> None:
        """Multiple tool calls in a single round execute concurrently."""
        execution_order: list[str] = []

        async def handler(name: str, args: dict) -> str:
            execution_order.append(f"start:{name}")
            await asyncio.sleep(0)  # yield control
            execution_order.append(f"end:{name}")
            return f"result:{name}"

        # Provider returns 2 tool calls in one round, then final
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

        # Both tools should have executed
        assert "start:tool_a" in execution_order
        assert "start:tool_b" in execution_order
        assert response.content == "Done"

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
        tool_msg = context.messages[2]
        assert tool_msg.role == "tool"
        results = tool_msg.content
        assert len(results) == 2
        # Good tool succeeds
        assert results[0].result == "ok"
        # Bad tool has error message
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
        # Create a very large result (>120K chars = >30K estimated tokens)
        large = "x" * 200_000
        truncated = ch._maybe_truncate_result(large)

        assert len(truncated) < len(large)
        assert "[... truncated" in truncated
        assert truncated.startswith("x" * 100)
        assert truncated.endswith("x" * 100)


class TestContextOverflowRecovery:
    async def test_overflow_triggers_compaction(self) -> None:
        """Context overflow error triggers compaction and retry."""
        call_count = 0

        async def generate_with_overflow(context: AIContext) -> AIResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Second call overflows
                raise ProviderError("context length exceeded", retryable=True, status_code=400)
            if call_count == 3:
                # After compaction, succeeds
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

        # Seed context with enough messages for compaction
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

        # First message should be the summary
        assert "[Context compacted" in compacted.messages[0].content
        # Should have summary + second half (5 messages)
        assert len(compacted.messages) == 6  # 1 summary + 5 recent

    async def test_compact_context_raises_when_too_few_messages(self) -> None:
        """_compact_context raises when <= 4 messages."""
        provider = MockAIProvider()
        ch = AIChannel("ai1", provider=provider)

        messages = [AIMessage(role="user", content="m")] * 3
        context = AIContext(messages=messages)
        with pytest.raises(ProviderError, match="cannot compact further"):
            await ch._compact_context(context)


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
                content=[AITextPart(text="part1"), AIToolCallPart(id="x", name="t", arguments={})],
            ),
        ]
        result = AIChannel._extract_accumulated_text(messages)
        assert "part1" in result

    def test_returns_empty_for_no_assistant_messages(self) -> None:
        messages = [AIMessage(role="user", content="hello")]
        result = AIChannel._extract_accumulated_text(messages)
        assert result == ""
