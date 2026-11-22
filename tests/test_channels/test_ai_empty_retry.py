"""Bounded retry when a generation round ends empty *after* a tool call.

Small models sometimes run a tool, get the result, then return no text instead
of a final answer. The tool loop re-prompts once (bounded by ``max_empty_retries``)
for the final answer rather than ending empty. Covers a provider that streams
and one read through its ``generate()``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import _EMPTY_RETRY_NUDGE, AIChannel
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AIToolCall,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.tool_calls import MALFORMED_CALL_NUDGE, malformed_call_nudge
from tests.tool_loop_modes import run_tool_loop


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


async def test_retries_empty_after_tool_and_recovers(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_tool(), _final(""), _final("Recovered")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    run = await run_tool_loop(ch, context)
    assert run.text == "Recovered"
    assert _nudged(context) == 1  # one corrective re-prompt injected


async def test_no_retry_when_budget_zero(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_tool(), _final("")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=0,
    )
    context = _ctx()
    run = await run_tool_loop(ch, context)
    assert run.text == ""
    assert _nudged(context) == 0


async def test_no_retry_without_prior_tool(streaming: bool) -> None:
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
    await run_tool_loop(ch, context)
    assert _nudged(context) == 0
    assert len(provider.calls) == 1  # no extra generation


async def test_bounded_gives_up_when_still_empty(streaming: bool) -> None:
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
    run = await run_tool_loop(ch, context)
    assert run.text == ""
    assert _nudged(context) == 1  # exactly one retry, then give up


def _malformed(finish_reason: str = "malformed_call") -> AIResponse:
    """A round that ended on a tool call its provider would not hand over."""
    return AIResponse(content="", tool_calls=[], finish_reason=finish_reason)


def _told_malformed(context: AIContext, nudge: str = MALFORMED_CALL_NUDGE) -> int:
    return sum(1 for m in context.messages if m.content == nudge)


@pytest.mark.parametrize("finish_reason", ["malformed_call", "unexpected_call"])
async def test_a_malformed_call_is_told_and_retried_on_the_first_round(
    streaming: bool, finish_reason: str
) -> None:
    """The model learns its call did not run and issues it again (RMK-314): a
    call its provider could not parse, or one to a tool the request did not
    enable (RMK-438)."""
    handler = AsyncMock(return_value="ok")
    provider = MockAIProvider(ai_responses=[_malformed(finish_reason), _tool(), _final("Done")])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=handler,
        tool_loop_timeout_seconds=None,
        max_empty_retries=1,
    )
    context = _ctx()
    run = await run_tool_loop(ch, context)
    assert run.text == "Done"
    assert _told_malformed(context, malformed_call_nudge(finish_reason)) == 1
    assert handler.await_count == 1


async def test_a_malformed_call_past_the_budget_ends_the_turn_empty(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_malformed()])
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        max_empty_retries=0,
    )
    run = await run_tool_loop(ch, _ctx())
    assert run.text == ""
    assert run.reason == "empty_response"


async def test_a_malformed_call_after_text_is_told_too(streaming: bool) -> None:
    """The model narrated, then its call failed to parse: it is still told, and
    what it said stays its turn in the context (RMK-314, RFC §6.4)."""
    narrated = AIResponse(content="Let me check.", tool_calls=[], finish_reason="malformed_call")
    handler = AsyncMock(return_value="ok")
    provider = MockAIProvider(ai_responses=[narrated, _tool(), _final("Done")])
    ch = AIChannel("ai1", provider=provider, tool_handler=handler, tool_loop_timeout_seconds=None)
    context = _ctx()
    run = await run_tool_loop(ch, context)
    assert run.text == "Done"
    assert handler.await_count == 1
    said = [m.content for m in context.messages]
    assert said.index("Let me check.") + 1 == said.index(MALFORMED_CALL_NUDGE)


class _FailsAfterFirst(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        if self.calls:
            raise ProviderError("invalid api key", retryable=False, status_code=401)
        return await super().generate(context)


async def test_a_failure_before_any_round_is_the_turn_s_own_error(streaming: bool) -> None:
    """The retry of a first-round malformed call ran no round: a provider failure
    is raised, as the streaming loop does, never an interrupted turn."""
    ch = AIChannel(
        "ai1",
        provider=_FailsAfterFirst(ai_responses=[_malformed()]),
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
    )
    with pytest.raises(ProviderError):
        await run_tool_loop(ch, _ctx())


async def test_a_schema_turn_asks_again_instead_of_failing_its_check(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[_malformed(), _final('{"city": "Paris"}')],
        response_schema=True,
        response_schema_with_tools=True,
    )
    ch = AIChannel(
        "ai1", provider=provider, tool_handler=AsyncMock(), tool_loop_timeout_seconds=None
    )
    context = _ctx().model_copy(
        update={
            "response_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            }
        }
    )
    run = await run_tool_loop(ch, context)
    assert run.text == '{"city": "Paris"}'
