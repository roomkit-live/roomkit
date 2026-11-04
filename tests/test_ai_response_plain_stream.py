"""A turn without tools reports its completed response, usage and reasoning
included, through the one tool loop (RFC §6.4)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from roomkit import AIChannel
from roomkit.models.streaming import LoopEndMarker
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    ProviderError,
    StreamEvent,
)
from roomkit.providers.ai.mock import MockAIProvider


class TextOnlyProvider(AIProvider):
    """Streams text alone: no structured stream of its own, so the default
    reads a turn without tools through ``generate_stream``."""

    def __init__(self, *, ai_responses: list[AIResponse], streaming: bool = True) -> None:
        self._response = ai_responses[0]

    @property
    def model_name(self) -> str:
        return "text-only"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return self._response

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        if self._response.content:
            yield self._response.content


class BrokenProvider(MockAIProvider):
    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        async for event in super().generate_structured_stream(context):
            yield event
        raise ProviderError("broken stream", provider="mock", retryable=False)


@pytest.mark.parametrize("structured", [True, False])
@pytest.mark.parametrize("content", ["Hello there.", ""])
async def test_completed_plain_stream_reports_once(structured: bool, content: str) -> None:
    cls = MockAIProvider if structured else TextOnlyProvider
    provider = cls(
        streaming=True,
        ai_responses=[
            AIResponse(
                content=content,
                thinking="reasoning",
                finish_reason="stop",
                usage={"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3},
            )
        ],
    )
    channel = AIChannel("ai", provider)
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    channel._after_response_hook = observe
    async for _ in channel._run_streaming_tool_loop(AIContext()):
        pass
    assert channel.active_turns == 0
    assert len(seen) == 1
    event = seen[0]
    assert event.response_content == content
    assert event.segments == ([content] if content else [])
    assert event.streaming
    assert event.thinking == ("reasoning" if structured else "")
    # A text stream reports no usage: the counters read zero.
    assert event.usage == (
        {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3}
        if structured
        else {"input_tokens": 0, "output_tokens": 0}
    )
    assert event.tool_calls_count == event.round_count == 0
    assert event.loop_end_reason == "completed", "an exhausted stream ended on its own terms"


@pytest.mark.parametrize("close_early", [True, False])
async def test_unfinished_plain_stream_does_not_report_completion(close_early: bool) -> None:
    channel = AIChannel("ai", BrokenProvider(responses=["partial"], streaming=True))
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    channel._after_response_hook = observe
    stream = channel._run_streaming_tool_loop(AIContext())
    assert await anext(stream) == "partial"
    if close_early:
        await stream.aclose()
    else:
        with pytest.raises(ProviderError, match="broken stream"):
            async for _ in stream:
                pass
    assert not seen
    assert channel.active_turns == 0


async def test_plain_stream_hook_failure_does_not_break_delivery() -> None:
    channel = AIChannel("ai", MockAIProvider(responses=["ok"], streaming=True))

    async def observe(event: AIResponseEvent) -> None:
        raise RuntimeError("observer unavailable")

    channel._after_response_hook = observe
    items = [item async for item in channel._run_streaming_tool_loop(AIContext())]
    assert [item for item in items if isinstance(item, str)] == ["ok"]
    # The turn's record closes the structured stream (RMK-289)
    assert isinstance(items[-1], LoopEndMarker) and items[-1].reason == "completed"
    assert channel.active_turns == 0
