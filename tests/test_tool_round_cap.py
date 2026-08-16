"""Per-round ceiling on the tool calls honoured from one generation.

The loop already bounds rounds, wall clock, identical repeats and result
size. A single round was the one unbounded axis: a model can spend its whole
output budget emitting tool calls, and the loop would execute every one of
them. Observed in production: a 27B local model emitted 164 calls in one
completion (154 byte-identical) until it hit max_tokens.

The cap is asserted on BOTH loops — that parity is the reason the rule lives
in ``_ai_loop_rules``; enforced in one path only, it would cover just the
providers that use that generation mode.

The tests pin the ceiling to a small explicit value rather than reading the
shipped constant: a test that sizes its flood from the constant it is
testing passes whatever that constant is, including "no bound at all".
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from roomkit.channels._ai_loop_rules import AIToolLoopRulesMixin
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

_TEST_CAP = 4
_FLOOD = 20

_SEARCH_TOOL = {
    "name": "search",
    "description": "Search for something.",
    "parameters": {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
}


@pytest.fixture
def small_cap(monkeypatch: pytest.MonkeyPatch) -> int:
    """Shrink the per-round ceiling so a flood is cheap to assert."""
    monkeypatch.setattr(AIToolLoopRulesMixin, "_MAX_TOOL_CALLS_PER_ROUND", _TEST_CAP)
    return _TEST_CAP


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_SEARCH_TOOL]},
    )


def _flood(count: int) -> AIResponse:
    """One generation asking for *count* distinct tool calls."""
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            AIToolCall(id=f"tc{i}", name="search", arguments={"q": f"query {i}"})
            for i in range(count)
        ],
    )


def _responses(count: int) -> list[AIResponse]:
    return [_flood(count), AIResponse(content="done", finish_reason="stop")]


def _assistant_tool_call_ids(context: Any) -> list[str]:
    return [
        part.id
        for message in context.messages
        if message.role == "assistant"
        for part in message.content or []
        if getattr(part, "id", None) and getattr(part, "name", None)
    ]


def _tool_result_ids(context: Any) -> list[str]:
    return [
        part.tool_call_id
        for message in context.messages
        if message.role == "tool"
        for part in message.content or []
    ]


async def _run(provider: MockAIProvider, handler: Any) -> Any:
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )
    if output.response_stream is not None:
        async for _ in output.response_stream:
            pass
    return output


def _recording_handler(executed: list[dict]) -> Any:
    async def handler(name: str, arguments: dict) -> str:
        executed.append(arguments)
        return json.dumps({"ok": True})

    return handler


def test_shipped_ceiling_is_a_real_bound() -> None:
    """The default must bound a degenerate run while clearing normal fan-out."""
    ceiling = AIToolLoopRulesMixin._MAX_TOOL_CALLS_PER_ROUND
    assert 8 <= ceiling <= 64


async def test_flooded_round_is_capped(small_cap: int) -> None:
    executed: list[dict] = []
    await _run(MockAIProvider(ai_responses=_responses(_FLOOD)), _recording_handler(executed))
    assert len(executed) == small_cap


async def test_round_under_the_ceiling_is_untouched(small_cap: int) -> None:
    executed: list[dict] = []
    await _run(MockAIProvider(ai_responses=_responses(small_cap)), _recording_handler(executed))
    assert len(executed) == small_cap


async def test_capped_round_leaves_no_orphan_tool_call(small_cap: int) -> None:
    """A dropped call must be absent from the assistant message too.

    An assistant message carrying a tool call with no matching result is a
    hard error on OpenAI-compatible providers, so truncating the execution
    without truncating the transcript would trade a loop for a 400 on the
    very next round.
    """
    provider = MockAIProvider(ai_responses=_responses(_FLOOD))
    await _run(provider, _recording_handler([]))

    final_context = provider.calls[-1]
    call_ids = _assistant_tool_call_ids(final_context)
    assert len(call_ids) == small_cap
    assert call_ids == _tool_result_ids(final_context)


async def test_streaming_flooded_round_is_capped(small_cap: int) -> None:
    executed: list[dict] = []
    await _run(
        MockAIProvider(ai_responses=_responses(_FLOOD), streaming=True),
        _recording_handler(executed),
    )
    assert len(executed) == small_cap


async def test_streaming_capped_round_leaves_no_orphan_tool_call(small_cap: int) -> None:
    provider = MockAIProvider(ai_responses=_responses(_FLOOD), streaming=True)
    await _run(provider, _recording_handler([]))

    final_context = provider.calls[-1]
    call_ids = _assistant_tool_call_ids(final_context)
    assert len(call_ids) == small_cap
    assert call_ids == _tool_result_ids(final_context)
