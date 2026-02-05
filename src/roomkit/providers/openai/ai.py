"""OpenAI AI provider — generates responses via the OpenAI Chat Completions API."""

from __future__ import annotations

import json
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIProvider,
    AIResponse,
    AITextPart,
    AIToolCall,
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
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_vision(self) -> bool:
        """GPT-4o and GPT-4-turbo models support vision."""
        return any(self._config.model.startswith(prefix) for prefix in _VISION_MODELS)

    def _format_content(
        self, content: str | list[AITextPart | AIImagePart]
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

    async def generate(self, context: AIContext) -> AIResponse:
        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        messages.extend(
            {"role": m.role, "content": self._format_content(m.content)} for m in context.messages
        )

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
