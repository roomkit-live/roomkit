"""Anthropic AI provider — generates responses via the Anthropic Messages API."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIProvider,
    AIResponse,
    AITextPart,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.anthropic.config import AnthropicConfig

# Claude models that support vision (Claude 3 and later)
_VISION_MODELS = (
    "claude-3",
    "claude-sonnet-4",
    "claude-opus-4",
)


class AnthropicAIProvider(AIProvider):
    """AI provider using the Anthropic Messages API."""

    def __init__(self, config: AnthropicConfig) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for AnthropicAIProvider. "
                "Install it with: pip install roomkit[anthropic]"
            ) from exc
        self._config = config
        self._api_status_error = _anthropic.APIStatusError
        self._client = _anthropic.AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
        )

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
        """Claude 3+ models support vision."""
        return any(self._config.model.startswith(prefix) for prefix in _VISION_MODELS)

    def _format_content(
        self, content: str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart]
    ) -> str | list[dict[str, Any]]:
        """Format message content for Anthropic API.

        Converts AITextPart/AIImagePart/AIToolCallPart/AIToolResultPart to
        Anthropic's content block format.
        """
        if isinstance(content, str):
            return content

        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, AITextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, AIImagePart):
                url = part.url
                if url.startswith("data:"):
                    # data:<media_type>;base64,<data>
                    header, _, b64data = url.partition(",")
                    media_type = header.split(":", 1)[1].split(";", 1)[0]
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64data,
                            },
                        }
                    )
                else:
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            },
                        }
                    )
            elif isinstance(part, AIToolCallPart):
                parts.append(
                    {
                        "type": "tool_use",
                        "id": part.id,
                        "name": part.name,
                        "input": part.arguments,
                    }
                )
            elif isinstance(part, AIToolResultPart):
                parts.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": part.tool_call_id,
                        "content": part.result,
                    }
                )
        return parts

    def _build_messages(
        self,
        messages: list[Any],
    ) -> list[dict[str, Any]]:
        """Build Anthropic-formatted messages, mapping tool roles to user."""
        result: list[dict[str, Any]] = []
        for m in messages:
            role = "user" if m.role == "tool" else m.role
            result.append({"role": role, "content": self._format_content(m.content)})
        return result

    def _build_kwargs(self, context: AIContext) -> dict[str, Any]:
        """Build kwargs dict shared by generate and streaming paths."""
        messages = self._build_messages(context.messages)
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": context.max_tokens or self._config.max_tokens,
            "messages": messages,
        }
        if context.system_prompt:
            kwargs["system"] = context.system_prompt
        if context.temperature is not None:
            kwargs["temperature"] = context.temperature
        if context.tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in context.tools
            ]
        return kwargs

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from the Anthropic Messages streaming API."""
        kwargs = self._build_kwargs(context)
        t0 = time.monotonic()
        first_token = True

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if first_token:
                        ttfb_ms = (time.monotonic() - t0) * 1000
                        from roomkit.telemetry.noop import NoopTelemetryProvider

                        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
                        telemetry.record_metric(
                            "roomkit.llm.ttfb_ms",
                            ttfb_ms,
                            unit="ms",
                            attributes={
                                "provider": "anthropic",
                                "model": self._config.model,
                            },
                        )
                        first_token = False
                    yield StreamTextDelta(text=text)

                # Extract final message for tool calls and usage
                final = await stream.get_final_message()

            # Yield tool calls from the final message
            for block in final.content:
                if block.type == "tool_use":
                    yield StreamToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )

            yield StreamDone(
                finish_reason=final.stop_reason,
                usage={
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                },
                metadata={"model": final.model},
            )
        except self._api_status_error as exc:
            retryable = exc.status_code in (429, 500, 502, 503, 529)
            raise ProviderError(
                str(exc),
                retryable=retryable,
                provider="anthropic",
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                str(exc),
                retryable=False,
                provider="anthropic",
                status_code=None,
            ) from exc

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
            finish_reason=done_event.finish_reason if done_event else None,
            usage=done_event.usage if done_event else {},
            metadata=done_event.metadata if done_event else {},
            tool_calls=tool_calls,
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the Anthropic Messages API."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text
