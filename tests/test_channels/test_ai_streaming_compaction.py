"""Overflow recovery and replay safety live in the resilience wrappers.

Every generation path that goes through the retry wrappers (the streaming
tool loop, the no-tools streaming path, the non-streaming loop) compacts and
replays a context-window refusal once, before the retry budget or the
fallback provider see the oversized request. One guard rules every recovery,
and only the wrapper can enforce it: **a stream that has yielded anything is
never re-entered** — not by compaction, not by retry, not by fallback —
because the consumer already got the events and a replay would duplicate
them. A refusal that survives its compacted replay falls through to the
ordinary retry semantics, so an error that only *sounded* like an overflow
keeps its budget.

Detection is a tri-state typed fact first: a producer that classifies the
failure structurally sets ``ProviderError.context_overflow`` to ``True`` or
``False`` (the OpenAI-compatible family reads its error code; an envelope may
answer after measuring) and is believed in both directions; the shared phrase
list in ``is_context_overflow_message`` decides only when the flag is
``None``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import RetryPolicy
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
    is_context_overflow_message,
)
from roomkit.providers.ai.mock import MockAIProvider


class _ScriptedStreamProvider(MockAIProvider):
    """Plays one scripted step per stream call.

    A step is an ``AIResponse`` (streamed as deltas), an exception (raised
    before any delta), or a ``(events, exception)`` pair (the events, then
    the failure — a stream that already spoke).
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
        step = self.script[min(len(self.calls), len(self.script)) - 1]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, tuple):
            prelude, exc = step
            for event in prelude:
                yield event
            raise exc
        events: list[StreamEvent] = []
        if step.content:
            events.append(StreamTextDelta(text=step.content))
        for tc in step.tool_calls:
            events.append(StreamToolCall(id=tc.id, name=tc.name, arguments=tc.arguments))
        events.append(StreamDone(finish_reason=step.finish_reason, usage=step.usage))
        for event in events:
            yield event


def _overflow_by_phrase(*, retryable: bool = False) -> ProviderError:
    return ProviderError("prompt is too long: 210000 tokens > 200000 maximum", retryable=retryable)


def _overflow_by_flag() -> ProviderError:
    # Wording no phrase in the fallback list matches — the message of an
    # envelope that classified the failure structurally and rewrapped the
    # provider's prose.
    return ProviderError(
        '{"code": "refused", "detail": "window blown"}',
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


def _channel(
    provider: _ScriptedStreamProvider,
    *,
    retry_policy: RetryPolicy | None = None,
    fallback: Any | None = None,
) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=provider,
        tool_handler=AsyncMock(return_value="ok"),
        tool_loop_timeout_seconds=None,
        retry_policy=retry_policy,
        fallback_provider=fallback,
    )


@pytest.fixture(autouse=True)
def _no_backoff_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)


async def _drain(stream: Any) -> list[Any]:
    return [item async for item in stream]


def _texts(items: list[Any]) -> str:
    return "".join(x for x in items if isinstance(x, str))


# ── the streaming tool loop recovers ────────────────────────────────


async def test_overflow_mid_loop_compacts_and_the_turn_completes() -> None:
    provider = _ScriptedStreamProvider(
        [_tool(), _overflow_by_phrase(), AIResponse(content="Done", tool_calls=[])]
    )
    items = await _drain(_channel(provider)._run_streaming_tool_loop(_ctx()))

    assert _texts(items).count("Done") == 1
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

    assert _texts(items).count("Done") == 1
    assert len(provider.calls) == 3
    assert provider.seen_compacted[2] is True


async def test_a_round_that_already_spoke_is_never_replayed() -> None:
    # The stream emits text, then dies on overflow. Replaying would duplicate
    # the emitted text in the room and in the persisted message — the error
    # must propagate instead, after exactly one provider call.
    provider = _ScriptedStreamProvider(
        [([StreamTextDelta(text="Partial")], _overflow_by_phrase())]
    )
    stream = _channel(provider)._run_streaming_tool_loop(_ctx())

    collected: list[Any] = []
    with pytest.raises(ProviderError):
        async for item in stream:
            collected.append(item)

    assert _texts(collected) == "Partial"
    assert len(provider.calls) == 1


async def test_one_compaction_per_call_then_the_error_propagates() -> None:
    provider = _ScriptedStreamProvider([_overflow_by_phrase(), _overflow_by_phrase()])
    with pytest.raises(ProviderError):
        await _drain(_channel(provider)._run_streaming_tool_loop(_ctx()))

    assert len(provider.calls) == 2  # the refused call + its one compacted replay


async def test_a_non_overflow_error_propagates_without_compaction() -> None:
    provider = _ScriptedStreamProvider([ProviderError("boom", retryable=False)])
    with pytest.raises(ProviderError):
        await _drain(_channel(provider)._run_streaming_tool_loop(_ctx()))

    assert len(provider.calls) == 1


# ── the wrapper's guards, driven directly ───────────────────────────


async def test_a_stream_that_spoke_is_not_reentered_even_by_the_retry() -> None:
    # A retryable mid-stream failure must not re-enter the generator: the
    # round consuming it has already forwarded the delivered events.
    provider = _ScriptedStreamProvider(
        [([StreamTextDelta(text="Voici")], ProviderError("503", retryable=True))]
    )
    ch = _channel(provider, retry_policy=RetryPolicy(max_retries=3))

    collected: list[Any] = []
    with pytest.raises(ProviderError):
        async for event in ch._generate_stream_with_retry(_ctx()):
            collected.append(event)

    assert len(provider.calls) == 1
    assert sum(1 for e in collected if isinstance(e, StreamTextDelta)) == 1


async def test_stream_done_counts_as_emission() -> None:
    # Usage was already accumulated by the consumer; a replay would count it
    # twice. Emission is *any* yielded event, StreamDone included.
    provider = _ScriptedStreamProvider(
        [
            (
                [StreamDone(finish_reason="stop", usage={"input_tokens": 100})],
                _overflow_by_phrase(),
            )
        ]
    )
    with pytest.raises(ProviderError):
        await _drain(_channel(provider)._generate_stream_with_retry(_ctx()))

    assert len(provider.calls) == 1


_TPM_RATE_LIMIT = (
    "Request too large for gpt-4o in organization org-x on tokens per min (TPM): "
    "Limit 30000, Requested 31000."
)


async def test_a_rate_limit_worded_like_an_overflow_keeps_its_retries() -> None:
    # The property, not the incident: a retryable 429 reaches the provider
    # max_retries + 1 times whatever its prose says, and nobody compacts the
    # caller's context for a transient refusal.
    fallback = MockAIProvider(streaming=True, responses=["from fallback"])
    provider = _ScriptedStreamProvider([ProviderError(_TPM_RATE_LIMIT, retryable=True)])
    ch = _channel(provider, retry_policy=RetryPolicy(max_retries=3), fallback=fallback)

    items = await _drain(ch._generate_stream_with_retry(_ctx()))

    assert len(provider.calls) == 4  # every attempt the policy announces
    assert provider.seen_compacted == [False, False, False, False]
    assert len(set(provider.seen_message_counts)) == 1  # context untouched
    assert any(isinstance(e, StreamTextDelta) and e.text == "from fallback" for e in items)


async def test_an_explicit_no_beats_matching_prose() -> None:
    # An envelope that classified structurally and answered "not an overflow"
    # is believed, even when the wording matches the fallback list.
    provider = _ScriptedStreamProvider(
        [ProviderError("maximum context length exceeded", retryable=True, context_overflow=False)]
    )
    ch = _channel(provider, retry_policy=RetryPolicy(max_retries=2))

    with pytest.raises(ProviderError):
        await _drain(ch._generate_stream_with_retry(_ctx()))

    assert len(provider.calls) == 3  # retried, never compacted
    assert provider.seen_compacted == [False, False, False]


async def test_a_refusal_that_survives_compaction_falls_through_to_retry() -> None:
    # One compacted replay, then ordinary retry semantics: a retryable error
    # that still sounds like an overflow after compaction keeps its budget.
    provider = _ScriptedStreamProvider(
        [
            _overflow_by_phrase(retryable=True),
            _overflow_by_phrase(retryable=True),
            AIResponse(content="Done", tool_calls=[]),
        ]
    )
    ch = _channel(provider, retry_policy=RetryPolicy(max_retries=3))

    items = await _drain(ch._generate_stream_with_retry(_ctx()))

    assert len(provider.calls) == 3  # refusal, compacted replay, retry
    assert provider.seen_compacted == [False, True, True]
    assert any(isinstance(e, StreamTextDelta) and e.text == "Done" for e in items)


async def test_overflow_is_recovered_before_retry_and_fallback() -> None:
    # Replaying the same context is a deterministic refusal: the retry budget
    # and the fallback provider never see the oversized request, whatever the
    # error's retryable bit claims.
    fallback = MockAIProvider(streaming=True, responses=["from fallback"])
    provider = _ScriptedStreamProvider(
        [_overflow_by_phrase(retryable=True), AIResponse(content="Done", tool_calls=[])]
    )
    ch = _channel(
        provider,
        retry_policy=RetryPolicy(max_retries=3),
        fallback=fallback,
    )

    items = await _drain(ch._generate_stream_with_retry(_ctx()))

    assert len(provider.calls) == 2  # refusal + compacted replay, no retry burn
    assert fallback.calls == []
    assert any(isinstance(e, StreamTextDelta) and e.text == "Done" for e in items)


# ── the no-tools streaming path shares the recovery ─────────────────


async def test_the_no_tools_streaming_path_compacts_too() -> None:
    provider = _ScriptedStreamProvider(
        [_overflow_by_phrase(), AIResponse(content="Done", tool_calls=[])]
    )
    items = await _drain(_channel(provider)._stream_text_with_thinking(_ctx()))

    assert _texts(items).count("Done") == 1
    assert len(provider.calls) == 2
    assert provider.seen_compacted == [False, True]


async def test_the_no_tools_streaming_path_retries_and_falls_back() -> None:
    # The no-tools path draws retry and fallback from the wrapper like every
    # other generation path — the policy's word holds here too.
    fallback = MockAIProvider(streaming=True, responses=["from fallback"])
    provider = _ScriptedStreamProvider([ProviderError("503", retryable=True)])
    ch = _channel(
        provider,
        retry_policy=RetryPolicy(max_retries=2),
        fallback=fallback,
    )

    items = await _drain(ch._stream_text_with_thinking(_ctx()))

    assert len(provider.calls) == 3  # every attempt the policy announces
    assert _texts(items) == "from fallback"


# ── detection: the typed fact and the shared phrase list ────────────


def test_is_context_overflow_reads_the_typed_flag_first() -> None:
    assert AIChannel._is_context_overflow(_overflow_by_flag())
    assert AIChannel._is_context_overflow(_overflow_by_phrase())
    assert not AIChannel._is_context_overflow(ProviderError("boom"))


def test_the_shared_phrase_list_covers_both_packages_wordings() -> None:
    for text in (
        "This model's maximum context length is 128000 tokens",
        "error code: context_length_exceeded",
        "input exceeds the model capacity",
        "the request rode past the token limit",
        "prompt is too long: 210000 tokens > 200000 maximum",
        "Range of input length should be [1, 30720]",
    ):
        assert is_context_overflow_message(text), text
    assert not is_context_overflow_message("invalid api key")
    # OpenAI's tokens-per-minute rate limit — a 429 worth retrying, and the
    # wording that once cost a transient error its whole retry budget.
    assert not is_context_overflow_message(_TPM_RATE_LIMIT)


def test_openai_status_errors_carry_the_typed_fact() -> None:
    from roomkit.providers.openai.response import _overflow_fact

    class _FakeStatusError(Exception):
        def __init__(self, body: Any) -> None:
            self.body = body

    overflow = _FakeStatusError({"error": {"code": "context_length_exceeded"}})
    other = _FakeStatusError({"error": {"code": "invalid_request_error"}})
    assert _overflow_fact(overflow) is True
    # A miss is "nobody classified", never "no": the compatible vendors put
    # integers or generic strings in ``code`` and their overflows must stay
    # catchable by the phrase fallback.
    assert _overflow_fact(other) is None
    assert _overflow_fact(_FakeStatusError(None)) is None
