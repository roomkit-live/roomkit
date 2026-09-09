"""Plain streams report completed responses, including usage and reasoning."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from roomkit import AIChannel
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import AIContext, AIResponse, ProviderError, StreamEvent
from roomkit.providers.ai.mock import MockAIProvider


class TextOnlyProvider(MockAIProvider):
    @property
    def supports_structured_streaming(self) -> bool:
        return False


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
    async for _ in channel._stream_text_with_thinking(AIContext()):
        pass
    assert channel.active_turns == 0
    assert len(seen) == 1
    event = seen[0]
    assert event.response_content == content
    assert event.segments == ([content] if content else [])
    assert event.streaming
    assert event.thinking == ("reasoning" if structured else "")
    assert event.usage == (provider._ai_responses[0].usage if structured else {})
    assert event.tool_calls_count == event.round_count == 0
    assert event.loop_end_reason == "completed", "an exhausted stream ended on its own terms"


@pytest.mark.parametrize("close_early", [True, False])
async def test_unfinished_plain_stream_does_not_report_completion(close_early: bool) -> None:
    channel = AIChannel("ai", BrokenProvider(responses=["partial"], streaming=True))
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    channel._after_response_hook = observe
    stream = channel._stream_text_with_thinking(AIContext())
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
    assert [item async for item in channel._stream_text_with_thinking(AIContext())] == ["ok"]
    assert channel.active_turns == 0
