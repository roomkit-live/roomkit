"""Google Gemini AI provider — generates responses via the Google Generative AI API."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, cast

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AIThinkingPart,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.gemini.config import GeminiConfig


class GeminiAIProvider(AIProvider):
    """AI provider using the Google Gemini API."""

    def __init__(self, config: GeminiConfig) -> None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiAIProvider. "
                "Install it with: pip install roomkit[gemini]"
            ) from exc

        self._config = config
        self._genai = _genai
        self._types = _types
        self._client = _genai.Client(api_key=config.api_key.get_secret_value())

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        """All Gemini models support vision."""
        return True

    def _format_messages(self, messages: list[AIMessage]) -> list[Any]:
        """Convert AIMessage list to Gemini Content format."""
        contents = []
        for msg in messages:
            if isinstance(msg.content, list) and any(
                isinstance(p, AIToolCallPart) for p in msg.content
            ):
                # Model message with function calls
                parts = []
                for p in msg.content:
                    if isinstance(p, AITextPart):
                        parts.append(self._types.Part.from_text(text=p.text))
                    elif isinstance(p, AIToolCallPart):
                        parts.append(
                            self._types.Part.from_function_call(
                                name=p.name,
                                args=p.arguments,
                            )
                        )
                contents.append(self._types.Content(role="model", parts=parts))
            elif isinstance(msg.content, list) and any(
                isinstance(p, AIToolResultPart) for p in msg.content
            ):
                # Function responses
                parts = []
                for p in msg.content:
                    if isinstance(p, AIToolResultPart):
                        parts.append(
                            self._types.Part.from_function_response(
                                name=p.name,
                                response={"result": p.result},
                            )
                        )
                contents.append(self._types.Content(role="user", parts=parts))
            else:
                role = "model" if msg.role == "assistant" else "user"
                parts = self._format_content(msg.content)
                contents.append(self._types.Content(role=role, parts=parts))
        return contents

    def _format_content(
        self,
        content: (
            str
            | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
        ),
    ) -> list[Any]:
        """Convert content to Gemini Parts."""
        if isinstance(content, str):
            return [self._types.Part.from_text(text=content)]

        parts = []
        for item in content:
            if isinstance(item, AITextPart):
                parts.append(self._types.Part.from_text(text=item.text))
            elif isinstance(item, AIImagePart):
                parts.append(
                    self._types.Part.from_uri(
                        file_uri=item.url,
                        mime_type=item.mime_type or "image/jpeg",
                    )
                )
        return parts

    def _build_gen_config(self, context: AIContext) -> Any:
        """Build Gemini generation config from AIContext."""
        gen_config = self._types.GenerateContentConfig(
            temperature=context.temperature,
            max_output_tokens=context.max_tokens,
        )

        if context.system_prompt:
            gen_config.system_instruction = context.system_prompt

        if context.tools:
            func_decls = [
                self._types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    # Cast to Any: Gemini SDK accepts dict as Schema at runtime
                    parameters=cast(Any, t.parameters) if t.parameters else None,
                )
                for t in context.tools
            ]
            gen_config.tools = [self._types.Tool(function_declarations=func_decls)]

        return gen_config

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Wrap an SDK exception into a ProviderError."""
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        retryable = (
            status_code in (429, 500, 502, 503)
            if status_code
            else any(
                term in str(exc).lower() for term in ["rate", "limit", "429", "500", "502", "503"]
            )
        )
        return ProviderError(
            str(exc),
            retryable=retryable,
            provider="gemini",
            status_code=status_code,
        )

    def _record_ttfb(self, t0: float) -> None:
        """Record time-to-first-byte metric."""
        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.llm.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": "gemini", "model": self._config.model},
        )

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from the Gemini streaming API."""
        gen_config = self._build_gen_config(context)
        contents = self._format_messages(context.messages)

        t0 = time.monotonic()
        first_token = True
        tool_calls: list[StreamToolCall] = []
        usage: dict[str, int] = {}

        try:
            response_stream = await self._client.aio.models.generate_content_stream(
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
            async for chunk in response_stream:
                # Extract usage from each chunk (last one has the totals)
                if chunk.usage_metadata:
                    usage = {
                        "prompt_tokens": chunk.usage_metadata.prompt_token_count or 0,
                        "completion_tokens": chunk.usage_metadata.candidates_token_count or 0,
                    }

                if not chunk.candidates or not chunk.candidates[0].content:
                    continue

                parts = chunk.candidates[0].content.parts
                if not parts:
                    continue

                for part in parts:
                    if hasattr(part, "text") and part.text:
                        if first_token:
                            self._record_ttfb(t0)
                            first_token = False
                        yield StreamTextDelta(text=part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        fc_name: str = fc.name or ""
                        tool_calls.append(
                            StreamToolCall(
                                id=fc_name,
                                name=fc_name,
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )

            for tc in tool_calls:
                yield tc

            yield StreamDone(usage=usage, metadata={"model": self._config.model})

        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def generate(self, context: AIContext) -> AIResponse:
        """Generate by consuming the structured stream."""
        text_parts: list[str] = []
        tool_calls: list[AIToolCall] = []
        done_event: StreamDone | None = None

        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                text_parts.append(event.text)
            elif isinstance(event, StreamToolCall):
                tool_calls.append(
                    AIToolCall(id=event.id, name=event.name, arguments=event.arguments)
                )
            elif isinstance(event, StreamDone):
                done_event = event

        return AIResponse(
            content="".join(text_parts),
            usage=done_event.usage if done_event else {},
            tool_calls=tool_calls,
            metadata=done_event.metadata if done_event else {},
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the Gemini API."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def close(self) -> None:
        """Release the genai client reference."""
        self._client = None  # type: ignore[assignment]
