"""The tool-loop rules are single-definition: both loops must consume them.

The non-streaming (``_run_tool_loop``) and streaming
(``_run_streaming_tool_loop``) tool loops share their per-round business
rules via ``AIToolLoopRulesMixin`` (``channels/_ai_loop_rules.py``). These
tests patch a shared rule and assert BOTH paths reflect the patch — they
fail if a rule is re-inlined into one loop only, which would leave the
other generation mode without it.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import _EMPTY_RETRY_NUDGE, AIChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

_ECHO_TOOL = {
    "name": "echo",
    "description": "Echo a value back.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_ECHO_TOOL]},
    )


def _tool(i: int = 0) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"t{i}", name="echo", arguments={"value": str(i)})],
    )


async def _drain(output: ChannelOutput) -> None:
    if output.response_stream is not None:
        async for _ in output.response_stream:
            pass


@pytest.mark.parametrize("streaming", [False, True])
async def test_prepare_round_context_drives_both_paths(monkeypatch, streaming: bool) -> None:
    """Both loops consume the shared prepare-round rule and honor its result."""
    invocations = {"n": 0}
    original = AIChannel._prepare_round_context

    def stripping(self, context, loop_ctx, state, round_idx):
        invocations["n"] += 1
        prepared = original(self, context, loop_ctx, state, round_idx)
        return prepared.model_copy(update={"tools": []})

    monkeypatch.setattr(AIChannel, "_prepare_round_context", stripping)

    provider = MockAIProvider(
        ai_responses=[_tool(), AIResponse(content="done")], streaming=streaming
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value=json.dumps({"ok": True})),
    )
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )
    await _drain(output)

    # The loop consulted the shared rule at least once...
    assert invocations["n"] >= 1
    # ...and honored the context it returned: the generation following the
    # patched rule ran with tools stripped.
    assert not provider.calls[-1].tools
    if not streaming:
        # Sanity: the pre-loop generation (before any rule ran) had tools,
        # so the stripped last call proves the rule's effect.
        assert provider.calls[0].tools


@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_retry_rule_drives_both_paths(monkeypatch, streaming: bool) -> None:
    """Both loops consult the shared empty-retry rule, not a local copy."""
    monkeypatch.setattr(AIChannel, "_try_empty_retry", lambda self, *a, **k: False)

    provider = MockAIProvider(
        ai_responses=[_tool(), AIResponse(content=""), AIResponse(content="never")],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        max_empty_retries=1,
    )
    context = AIContext(messages=[AIMessage(role="user", content="go")])
    if streaming:
        async for _ in ch._run_streaming_tool_loop(context):
            pass
    else:
        await ch._run_tool_loop(context)

    # With the rule forced to "no retry", neither loop may re-generate or
    # nudge: a re-inlined local retry rule would produce a third provider
    # call and one nudge message here.
    assert len(provider.calls) == 2
    assert sum(1 for m in context.messages if m.content == _EMPTY_RETRY_NUDGE) == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_length_truncation_reaches_the_rule_on_both_paths(
    monkeypatch, streaming: bool
) -> None:
    """Both loops report the round's finish_reason to the empty-retry rule.

    A round truncated at the output cap arrives with empty content and
    ``finish_reason="length"``. A loop that drops finish_reason would hand the
    rule ``None``, making truncation indistinguishable from a silent model.
    """
    seen: list[str | None] = []
    original = AIChannel._try_empty_retry

    def recording(self, context, loop_ctx, state, **kwargs):
        seen.append(kwargs.get("finish_reason"))
        return original(self, context, loop_ctx, state, **kwargs)

    monkeypatch.setattr(AIChannel, "_try_empty_retry", recording)

    provider = MockAIProvider(
        ai_responses=[
            _tool(),
            AIResponse(content="", finish_reason="length"),
            AIResponse(content="never"),
        ],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        max_empty_retries=1,
    )
    context = AIContext(messages=[AIMessage(role="user", content="go")])
    if streaming:
        async for _ in ch._run_streaming_tool_loop(context):
            pass
    else:
        await ch._run_tool_loop(context)

    assert seen == ["length"]
    # Truncation is not silence: the nudge would be truncated again under the
    # same cap, so the rule must not spend a retry on it.
    assert len(provider.calls) == 2
    assert sum(1 for m in context.messages if m.content == _EMPTY_RETRY_NUDGE) == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reason", ["length", "max_tokens", "MAX_TOKENS"])
async def test_every_provider_spelling_of_truncation_is_understood(
    streaming: bool, reason: str
) -> None:
    """The rule reads truncation, not one provider's word for it.

    RoomKit forwards each provider's raw finish_reason: OpenAI-compatible
    servers and Ollama say ``length``, Anthropic's stop_reason says
    ``max_tokens``, Gemini's candidate says ``MAX_TOKENS``. A rule matching a
    single spelling would burn the retries it exists to save on the two
    providers whose reasoning budget is a headline feature.
    """
    provider = MockAIProvider(
        ai_responses=[
            _tool(),
            AIResponse(content="", finish_reason=reason),
            AIResponse(content="never"),
        ],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        max_empty_retries=1,
    )
    context = AIContext(messages=[AIMessage(role="user", content="go")])
    if streaming:
        async for _ in ch._run_streaming_tool_loop(context):
            pass
    else:
        await ch._run_tool_loop(context)

    assert len(provider.calls) == 2
    assert sum(1 for m in context.messages if m.content == _EMPTY_RETRY_NUDGE) == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_without_length_still_retries_on_both_paths(streaming: bool) -> None:
    """A genuinely silent round still gets its bounded nudge on both paths."""
    provider = MockAIProvider(
        ai_responses=[
            _tool(),
            AIResponse(content="", finish_reason="stop"),
            AIResponse(content="recovered"),
        ],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        max_empty_retries=1,
    )
    context = AIContext(messages=[AIMessage(role="user", content="go")])
    if streaming:
        async for _ in ch._run_streaming_tool_loop(context):
            pass
    else:
        await ch._run_tool_loop(context)

    assert len(provider.calls) == 3
    assert sum(1 for m in context.messages if m.content == _EMPTY_RETRY_NUDGE) == 1
