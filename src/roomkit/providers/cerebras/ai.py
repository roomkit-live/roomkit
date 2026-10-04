"""Cerebras AI provider using the shared OpenAI-compatible transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    ModelInfo,
    StreamEvent,
    StreamToolCall,
)
from roomkit.providers.ai.chat_request import ChatDialect
from roomkit.providers.ai.reasoning import floored_effort, reasoning_floor
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.cerebras.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.utils import _aclose_stream


def _decode_arrays(value: Any, schema: dict[str, Any]) -> Any:
    """Undo Cerebras's JSON-string arrays only where the tool declares an array.

    Observed on qwen-3.8-27b's raw SSE output with an array-typed schema.
    This is transport repair, not coercive validation: scalars, malformed JSON,
    ambiguous schemas and unknown properties remain untouched for the guards.
    Never repeatedly decode a string or mutate the provider's shared context.
    """
    if schema.get("type") == "array" and isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return value
        if not isinstance(decoded, list):
            return value
        value = decoded
    if schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(value, list) and isinstance(items, dict):
            return [_decode_arrays(item, items) for item in value]
    if schema.get("type") in (None, "object") and isinstance(value, dict):
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            return {
                key: _decode_arrays(item, properties[key])
                if isinstance(properties.get(key), dict)
                else item
                for key, item in value.items()
            }
    return value


class CerebrasAIProvider(OpenAIAIProvider):
    """Cerebras chat, reasoning and tool calling, including streaming.

    Reuses OpenAI's async client and RoomKit's response decoder, error mapping,
    token accounting and live ``/v1/models`` discovery. Cerebras-specific
    request parameters and historical reasoning are shaped here.

    Example::

        provider = CerebrasAIProvider(
            CerebrasConfig(api_key="...", model="gpt-oss-120b")
        )
    """

    _config: CerebrasConfig
    _install_extra: ClassVar[str] = "cerebras"
    _chat_dialect: ClassVar[ChatDialect] = ChatDialect(thinking_field="reasoning")
    """Cerebras reads a model's earlier reasoning from a ``reasoning`` field
    beside ``content``, tool rounds included, rather than inline."""

    async def generate(self, context: AIContext) -> AIResponse:
        response = await super().generate(context)
        schemas = {tool.name: tool.parameters for tool in context.tools or []}
        return response.model_copy(
            update={
                "tool_calls": [
                    call.model_copy(
                        update={
                            "arguments": _decode_arrays(call.arguments, schemas.get(call.name, {}))
                        }
                    )
                    for call in response.tool_calls
                ]
            }
        )

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        schemas = {tool.name: tool.parameters for tool in context.tools or []}
        stream = super().generate_structured_stream(context)
        try:
            async for event in stream:
                if isinstance(event, StreamToolCall):
                    event = event.model_copy(
                        update={
                            "arguments": _decode_arrays(
                                event.arguments, schemas.get(event.name, {})
                            )
                        }
                    )
                yield event
        finally:
            await _aclose_stream(stream)

    @property
    def name(self) -> str:
        """Stable provider name in streaming and non-streaming telemetry."""
        return "cerebras"

    @property
    def _provider_name(self) -> str:
        return self.name

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Offline metadata; use ``list_models()`` for account availability."""
        return list(MODELS)

    @property
    def supports_vision(self) -> bool:
        """Report the configured model's capability; unknown ids default False."""
        entry = self.catalog_entry()
        return bool(entry and entry.supports_vision)

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Keep reasoning controls active during tool calls as well as text
        turns; a turn that states off is sent the model's floor (RFC §6.7)."""
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        floor = reasoning_floor(self.catalog_entry())
        effort = floored_effort(context, self._config.reasoning_effort, floor)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        for key in ("reasoning_format", "clear_thinking"):
            value = getattr(self._config, key)
            if value is not None:
                kwargs.setdefault("extra_body", {})[key] = value
