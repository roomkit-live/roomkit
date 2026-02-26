"""Mock AI provider for testing."""

from __future__ import annotations

from collections.abc import AsyncIterator

from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)


class MockAIProvider(AIProvider):
    """Round-robin response provider for tests."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        vision: bool = False,
        ai_responses: list[AIResponse] | None = None,
        streaming: bool = False,
    ) -> None:
        self.responses = responses or ["Hello from AI"]
        self._ai_responses = ai_responses
        self.calls: list[AIContext] = []
        self._index = 0
        self._vision = vision
        self._streaming = streaming

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def supports_vision(self) -> bool:
        return self._vision

    @property
    def supports_streaming(self) -> bool:
        return self._streaming

    @property
    def supports_structured_streaming(self) -> bool:
        return self._streaming

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if self._ai_responses:
            resp = self._ai_responses[self._index % len(self._ai_responses)]
            self._index += 1
            return resp
        content = self.responses[self._index % len(self.responses)]
        self._index += 1
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text from generate() as a single delta."""
        response = await self.generate(context)
        if response.content:
            yield response.content

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from generate() result."""
        response = await self.generate(context)
        if response.thinking:
            yield StreamThinkingDelta(thinking=response.thinking)
        if response.content:
            yield StreamTextDelta(text=response.content)
        for tc in response.tool_calls:
            yield StreamToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
        yield StreamDone(
            finish_reason=response.finish_reason,
            usage=response.usage,
            metadata=response.metadata,
        )
