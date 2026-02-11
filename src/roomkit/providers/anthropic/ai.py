"""Anthropic AI provider — generates responses via the Anthropic Messages API."""

from __future__ import annotations

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
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": part.url,
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

    async def generate(self, context: AIContext) -> AIResponse:
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

        # Add tools if provided
        if context.tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in context.tools
            ]

        try:
            response = await self._client.messages.create(**kwargs)
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

        # Extract text content and tool calls from response
        text_content = ""
        tool_calls: list[AIToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    AIToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return AIResponse(
            content=text_content,
            finish_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            metadata={"model": response.model},
            tool_calls=tool_calls,
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the Anthropic Messages API."""
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

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
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
