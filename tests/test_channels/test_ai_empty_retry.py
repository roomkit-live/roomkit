"""Bounded retry when a generation round ends empty *after* a tool call.

Small models sometimes run a tool, get the result, then return no text instead
of a final answer. The tool loop re-prompts once (bounded by ``max_empty_retries``)
for the final answer rather than ending empty. Covers both the non-streaming
(``_run_tool_loop``) and streaming (``_run_streaming_tool_loop``) paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels.ai import _EMPTY_RETRY_NUDGE, AIChannel
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider


def _tool(content: str = "") -> AIResponse:
    return AIResponse(
        content=content,
        tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
    )


def _final(content: str = "") -> AIResponse:
    return AIResponse(content=content, tool_calls=[])


def _ctx() -> AIContext:
    return AIContext(messages=[AIMessage(role="user", content="go")])


def _nudged(context: AIContext) -> int:
    return sum(1 for m in context.messages if m.content == _EMPTY_RETRY_NUDGE)


# ── non-streaming ───────────────────────────────────────────────────


async def test_retries_empty_after_tool_and_recovers() -> None:
    provider = MockAIProvider(ai_responses=[_tool(), _final(""), _final("Recovered")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    result = await ch._run_tool_loop(context)
    assert result.response.content == "Recovered"
    assert _nudged(context) == 1  # one corrective re-prompt injected


async def test_no_retry_when_budget_zero() -> None:
    provider = MockAIProvider(ai_responses=[_tool(), _final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=0,
    )
    context = _ctx()
    result = await ch._run_tool_loop(context)
    assert result.response.content == ""
    assert _nudged(context) == 0


async def test_no_retry_without_prior_tool() -> None:
    # A legitimately empty turn with no tools must NOT trigger a retry.
    provider = MockAIProvider(ai_responses=[_final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=2,
    )
    context = _ctx()
    await ch._run_tool_loop(context)
    assert _nudged(context) == 0
    assert len(provider.calls) == 1  # no extra generation


async def test_bounded_gives_up_when_still_empty() -> None:
    # Model stays empty: retry once (budget 1) then give up with the empty answer.
    provider = MockAIProvider(ai_responses=[_tool(), _final(""), _final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    result = await ch._run_tool_loop(context)
    assert result.response.content == ""
    assert _nudged(context) == 1  # exactly one retry, then give up


# ── streaming ───────────────────────────────────────────────────────


async def _collect_text(stream) -> str:
    return "".join([d for d in [x async for x in stream] if isinstance(d, str)])


async def test_streaming_retries_empty_after_tool_and_recovers() -> None:
    provider = MockAIProvider(
        streaming=True, ai_responses=[_tool(), _final(""), _final("Recovered")]
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    text = await _collect_text(ch._run_streaming_tool_loop(context))
    assert "Recovered" in text
    assert _nudged(context) == 1


async def test_streaming_no_retry_without_prior_tool() -> None:
    provider = MockAIProvider(streaming=True, ai_responses=[_final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=2,
    )
    context = _ctx()
    await _collect_text(ch._run_streaming_tool_loop(context))
    assert _nudged(context) == 0
