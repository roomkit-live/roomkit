"""Mock AI provider for testing."""

from __future__ import annotations

from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse


class MockAIProvider(AIProvider):
    """Round-robin response provider for tests."""

    def __init__(self, responses: list[str] | None = None, *, vision: bool = False) -> None:
        self.responses = responses or ["Hello from AI"]
        self.calls: list[AIContext] = []
        self._index = 0
        self._vision = vision

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def supports_vision(self) -> bool:
        return self._vision

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        content = self.responses[self._index % len(self.responses)]
        self._index += 1
        return AIResponse(
            content=content,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
