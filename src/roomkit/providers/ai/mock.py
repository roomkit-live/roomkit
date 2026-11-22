"""Mock AI provider for testing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AIToolCall,
    ModelInfo,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    response_stream_events,
)
from roomkit.providers.ai.finish_reason import FinishReason
from roomkit.providers.ai.response_schema import (
    check_schema_answer,
    checked_stream,
    schema_for_generate,
)
from roomkit.providers.ai.tool_calls import is_malformed_call
from roomkit.providers.utils import _aclose_stream

_MOCK_MODELS = [
    ModelInfo(id="mock", display_name="Mock", context_window=8192, supports_vision=False),
    ModelInfo(
        id="mock-vision", display_name="Mock Vision", context_window=8192, supports_vision=True
    ),
]


class MockAIProvider(AIProvider):
    """Round-robin response provider for tests.

    With ``response_schema=True`` it honours :attr:`AIContext.response_schema`
    the way a real provider must (RFC §6.7), streaming included: the scripted
    answer has to be a JSON document satisfying it, and an :class:`AIResponse` scripted with
    ``finish_reason="refusal"``, or one that ran out of room (``"length"``),
    raises the matching
    :class:`~roomkit.providers.ai.response_schema.ResponseSchemaError`.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        vision: bool = False,
        ai_responses: list[AIResponse] | None = None,
        streaming: bool = False,
        tool_call_delta_chunks: int = 0,
        response_schema: bool = False,
        response_schema_with_tools: bool = False,
    ) -> None:
        self.responses = responses or ["Hello from AI"]
        self._ai_responses = ai_responses
        self.calls: list[AIContext] = []
        self._index = 0
        self._vision = vision
        self._streaming = streaming
        self._tool_call_delta_chunks = tool_call_delta_chunks
        self._response_schema = response_schema
        self._response_schema_with_tools = response_schema_with_tools

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

    @property
    def supports_response_schema(self) -> bool:
        return self._response_schema

    @property
    def supports_response_schema_with_tools(self) -> bool:
        return self._response_schema_with_tools

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Fixed two-entry catalog for exercising model-discovery code."""
        return list(_MOCK_MODELS)

    async def generate(self, context: AIContext) -> AIResponse:
        # Refused before the call is recorded: a real provider sends nothing.
        schema_for_generate(
            context,
            supported=self._response_schema,
            with_tools=self._response_schema_with_tools,
            provider="mock",
        )
        self.calls.append(context)
        response = self._next_response()
        # As a real provider: a call it could not parse is no answer (RFC §6.4).
        if (
            context.response_schema is not None
            and not response.tool_calls
            and not is_malformed_call(response.finish_reason)
        ):
            check_schema_answer(
                response.content,
                schema=context.response_schema,
                provider="mock",
                refusal="refusal" if response.finish_reason == "refusal" else None,
                finish_reason=response.finish_reason,
            )
        return response

    def _next_response(self) -> AIResponse:
        """The next scripted response, round-robin."""
        if self._ai_responses:
            resp = self._ai_responses[self._index % len(self._ai_responses)]
            self._index += 1
            return resp
        content = self.responses[self._index % len(self.responses)]
        self._index += 1
        return AIResponse(
            content=content,
            finish_reason=FinishReason.STOP,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield the scripted text as a single delta."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield the scripted response as events.

        A mock that does not stream is read the way every such provider is,
        through the default adapter over ``generate()``.
        """
        if not self._streaming:
            return super().generate_structured_stream(context)
        return self._stream(context)

    async def _stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The scripted response streamed the way a real provider streams it:
        text first, then a response schema's check before the done event."""
        schema_for_generate(
            context,
            supported=self._response_schema,
            with_tools=self._response_schema_with_tools,
            provider="mock",
        )
        stream = checked_stream(
            self._scripted_events(context),
            context,
            provider="mock",
            refusal=lambda done: "refusal" if done.finish_reason == "refusal" else None,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await _aclose_stream(stream)

    async def _scripted_events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        if context.response_schema is None:
            # Through generate(), so a test double that scripts generate() alone
            # drives the stream too.
            response = await self.generate(context)
        else:
            # Unchecked here: the text goes out first and checked_stream checks
            # it at the done event, as a real provider's stream does.
            self.calls.append(context)
            response = self._next_response()
        for event in response_stream_events(response, self._tool_call_deltas):
            yield event

    def _tool_call_deltas(self, tool_call: AIToolCall) -> list[StreamToolCallDelta]:
        """Split a call's arguments into ``tool_call_delta_chunks`` fragments.

        Real providers deliver a tool call's arguments fragment by fragment;
        this reproduces that for tests of the composition events. Zero chunks
        (the default) emits none, so a test that does not ask for them sees a
        plain complete call.
        """
        chunks = self._tool_call_delta_chunks
        if chunks <= 0:
            return []
        payload = json.dumps(tool_call.arguments)
        size = max(1, -(-len(payload) // chunks))
        fragments = [payload[i : i + size] for i in range(0, len(payload), size)] or [""]
        return [
            StreamToolCallDelta(id=tool_call.id, name=tool_call.name, arguments_delta=fragment)
            for fragment in fragments
        ]
