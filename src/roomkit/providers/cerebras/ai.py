"""Cerebras AI provider using the shared OpenAI-compatible transport."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, AIMessage, AIThinkingPart, ModelInfo
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.cerebras.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider


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
        """Keep reasoning controls active during tool calls as well as text turns."""
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        effort = context.reasoning_effort or self._config.reasoning_effort
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        for key in ("reasoning_format", "clear_thinking"):
            value = getattr(self._config, key)
            if value is not None:
                kwargs.setdefault("extra_body", {})[key] = value

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return assistant reasoning in Cerebras's separate ``reasoning`` field.

        The common builder embeds thinking in text tags for local servers.
        Cerebras expects a sibling of ``content``, including on tool turns.
        Copy the model before removing thinking so shared history is unchanged.
        """
        result = super()._build_messages([], system_prompt)
        for message in messages:
            reasoning = ""
            if message.role == "assistant" and isinstance(message.content, list):
                reasoning = "".join(
                    part.thinking for part in message.content if isinstance(part, AIThinkingPart)
                )
                content = [
                    part for part in message.content if not isinstance(part, AIThinkingPart)
                ]
                message = message.model_copy(update={"content": content or ""})
            converted = super()._build_messages([message])
            if reasoning:
                converted[0]["reasoning"] = reasoning
            result.extend(converted)
        return result
