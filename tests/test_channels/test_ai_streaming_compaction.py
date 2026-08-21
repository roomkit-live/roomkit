"""Reactive compaction on the streaming tool loop.

A context-window overflow that surfaces *during* a streamed turn used to kill
it: ``_compact_context`` only had a call site in the non-streaming loop, and
every streaming-capable provider routes to the streaming one. The streaming
loop now compacts and replays a round — but only a round that has not spoken,
because text and thinking deltas were already handed to the consumer and a
replay would duplicate them.

The detection is a typed fact first: an envelope that classifies the failure
structurally sets ``ProviderError.context_overflow`` and may reword the
message freely; the English phrase list stays as the fallback for raw
provider errors.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.streaming import LoopEndMarker
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AIToolCall,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider


class _ScriptedStreamProvider(MockAIProvider):
    """Plays one scripted step per stream call.

    A step is an ``AIResponse`` (streamed as deltas), an exception (raised
    before any delta), or a ``(text, exception)`` pair (one delta, then the
    failure — a stream that already spoke).
    """

    def __init__(self, script: list[Any]) -> None:
        super().__init__(streaming=True)
        self.script = script
        # Snapshots taken at call time: the loop mutates contexts in place,
        # so a stored reference would reflect later rounds, not this call.
        self.seen_message_counts: list[int] = []
        self.seen_compacted: list[bool] = []

    async def generate_structured_stream(self, context: AIContext) -> Any:
        self.calls.append(context)
        self.seen_message_counts.append(len(context.messages))
        self.seen_compacted.append(
            any(
                isinstance(m.content, str) and m.content.startswith("[Context compacted")
                for m in context.messages
            )
        )
        step = self.script[len(self.calls) - 1]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, tuple):
            text, exc = step
            yield StreamTextDelta(text=text)
            raise exc
        events: list[StreamEvent] = []
        if step.content:
            events.append(StreamTextDelta(text=step.content))
        for tc in step.tool_calls:
            events.append(StreamToolCall(id=tc.id, name=tc.name, arguments=tc.arguments))
        events.append(StreamDone(finish_reason=step.finish_reason, usage=step.usage))
        for event in events:
            yield event


def _overflow_by_phrase() -> ProviderError:
    return ProviderError("prompt is too long: 210000 tokens > 200000 maximum", retryable=False)


def _overflow_by_flag() -> ProviderError:
    # Wording no phrase in the fallback list matches — the envelope's own
    # message after it rewrapped the provider's prose (a classifying host).
    return ProviderError(
        '{"code": "provider_context_overflow", "message": "window blown"}',
        retryable=False,
        context_overflow=True,
    )


def _tool() -> AIResponse:
    return AIResponse(
        content="", tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})]
    )


def _ctx() -> AIContext:
    # More than four messages, so _compact_context has something to cut.
    messages = []
    for i in range(3):
        messages.append(AIMessage(role="user", content=f"q{i}"))
        messages.append(AIMessage(role="assistant", content=f"a{i}"))
    messages.append(AIMessage(role="user", content="go"))
    return AIContext(messages=messages)


def _channel(provider: _ScriptedStreamProvider) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
    )


async def _drain(stream: Any) -> list[Any]:
    return [item async for item in stream]


async def test_overflow_mid_loop_compacts_and_the_turn_completes() -> None:
    provider = _ScriptedStreamProvider(
        [_tool(), _overflow_by_phrase(), AIResponse(content="Done", tool_calls=[])]
    )
    items = await _drain(_channel(provider)._run_streaming_tool_loop(_ctx()))

    text = "".join(x for x in items if isinstance(x, str))
    assert text.count("Done") == 1
    assert len(provider.calls) == 3  # tool round, refused round, compacted replay
    # The replay ran on a compacted context: smaller, summary in front.
    assert provider.seen_message_counts[2] < provider.seen_message_counts[1]
    assert provider.seen_compacted == [False, False, True]
    markers = [x for x in items if isinstance(x, LoopEndMarker)]
    assert markers and markers[-1].reason == "completed"


async def test_the_typed_flag_triggers_compaction_whatever_the_wording() -> None:
    provider = _ScriptedStreamProvider(
        [_tool(), _overflow_by_flag(), AIResponse(content="Done", tool_calls=[])]
    )
    items = await _drain(_channel(provider)._run_streaming_tool_loop(_ctx()))

    assert "".join(x for x in items if isinstance(x, str)).count("Done") == 1
    assert len(provider.calls) == 3
    assert provider.seen_compacted[2] is True


async def test_a_round_that_already_spoke_is_never_replayed() -> None:
    # The stream emits text, then dies on overflow. Replaying would duplicate
    # the emitted text in the room and in the persisted message — the error
    # must propagate instead, after exactly one provider call.
    provider = _ScriptedStreamProvider([("Partial", _overflow_by_phrase())])
    stream = _channel(provider)._run_streaming_tool_loop(_ctx())

    collected: list[Any] = []
    with pytest.raises(ProviderError):
        async for item in stream:
            collected.append(item)

    assert "".join(x for x in collected if isinstance(x, str)) == "Partial"
    assert len(provider.calls) == 1


async def test_one_compaction_per_round_then_the_error_propagates() -> None:
    provider = _ScriptedStreamProvider([_overflow_by_phrase(), _overflow_by_phrase()])
    with pytest.raises(ProviderError):
        await _drain(_channel(provider)._run_streaming_tool_loop(_ctx()))

    assert len(provider.calls) == 2  # the refused round + its one compacted replay


async def test_a_non_overflow_error_propagates_without_compaction() -> None:
    provider = _ScriptedStreamProvider([ProviderError("boom", retryable=False)])
    with pytest.raises(ProviderError):
        await _drain(_channel(provider)._run_streaming_tool_loop(_ctx()))

    assert len(provider.calls) == 1


def test_is_context_overflow_reads_the_typed_flag_first() -> None:
    assert AIChannel._is_context_overflow(_overflow_by_flag())
    assert AIChannel._is_context_overflow(_overflow_by_phrase())
    assert not AIChannel._is_context_overflow(ProviderError("boom"))
