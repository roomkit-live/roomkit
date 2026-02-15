"""OpenAI AI provider — generates responses via the OpenAI Chat Completions API."""

from __future__ import annotations

import json
import time
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.openai.config import OpenAIConfig

# OpenAI models that support vision
_VISION_MODELS = (
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-vision",
    "o1",
    "o3",
)


class OpenAIAIProvider(AIProvider):
    """AI provider using the OpenAI Chat Completions API."""

    def __init__(self, config: OpenAIConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIAIProvider. "
                "Install it with: pip install roomkit[openai]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_vision(self) -> bool:
        """GPT-4o and GPT-4-turbo models support vision."""
        return any(self._config.model.startswith(prefix) for prefix in _VISION_MODELS)

    def _format_content(
        self,
        content: str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart],
    ) -> str | list[dict[str, Any]]:
        """Format message content for OpenAI API.

        Converts AITextPart/AIImagePart to OpenAI's content block format.
        """
        if isinstance(content, str):
            return content

        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, AITextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, AIImagePart):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part.url},
                    }
                )
        return parts

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-formatted messages with tool call/result support."""
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for m in messages:
            if isinstance(m.content, list) and any(
                isinstance(p, AIToolCallPart) for p in m.content
            ):
                # Assistant message with tool calls
                tool_calls = []
                content_text = ""
                for p in m.content:
                    if isinstance(p, AITextPart):
                        content_text = p.text
                    elif isinstance(p, AIToolCallPart):
                        tool_calls.append(
                            {
                                "id": p.id,
                                "type": "function",
                                "function": {
                                    "name": p.name,
                                    "arguments": json.dumps(p.arguments),
                                },
                            }
                        )
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_text or None,
                    "tool_calls": tool_calls,
                }
                result.append(msg)
            elif isinstance(m.content, list) and any(
                isinstance(p, AIToolResultPart) for p in m.content
            ):
                # Tool results → separate messages with role="tool"
                for p in m.content:
                    if isinstance(p, AIToolResultPart):
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": p.tool_call_id,
                                "content": p.result,
                            }
                        )
            else:
                result.append(
                    {
                        "role": m.role,
                        "content": self._format_content(m.content),
                    }
                )
        return result

    async def generate(self, context: AIContext) -> AIResponse:
        messages = self._build_messages(context.messages, context.system_prompt)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": context.max_tokens or self._config.max_tokens,
            "messages": messages,
        }
        if context.temperature is not None:
            kwargs["temperature"] = context.temperature

        # Add tools if provided
        if context.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in context.tools
            ]

        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except self._api_status_error as exc:
            retryable = exc.status_code in (429, 500, 502, 503)
            raise ProviderError(
                str(exc),
                retryable=retryable,
                provider="openai",
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                str(exc),
                retryable=False,
                provider="openai",
                status_code=None,
            ) from exc

        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.llm.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": "openai", "model": self._config.model},
        )

        choice = response.choices[0]
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        # Extract tool calls from response
        tool_calls: list[AIToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    AIToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return AIResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
            metadata={"model": response.model},
            tool_calls=tool_calls,
        )
