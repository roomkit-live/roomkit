"""DeepSeek AI provider — generates responses via DeepSeek's OpenAI-compatible API."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.deepseek.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider


class DeepSeekAIProvider(OpenAIAIProvider):
    """AI provider using DeepSeek's OpenAI-compatible Chat Completions API.

    Subclasses :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` —
    DeepSeek speaks the Chat Completions wire format verbatim, so message
    building, tool handling, response parsing, streaming, ``/v1/models``
    discovery, and client construction are all inherited unchanged. Three
    things are genuinely DeepSeek's own: which models exist, how a thinking
    request is spelled, and how a cache hit is reported.

    Example::

        provider = DeepSeekAIProvider(
            DeepSeekConfig(api_key="sk-...", model="deepseek-v4-pro")
        )
    """

    _config: DeepSeekConfig
    _install_extra: ClassVar[str] = "deepseek"

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "deepseek"

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of DeepSeek chat models."""
        return list(MODELS)

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and DeepSeek's nested ``thinking`` object.

        Where the OpenAI parent sends a top-level ``reasoning_effort`` string,
        DeepSeek takes ``thinking: {"type": ..., "reasoning_effort": ...}``.
        The two are not interchangeable: a top-level field is silently ignored
        here, so inheriting the parent's version would leave every reasoning
        setting with no effect on the wire. It rides the SDK's ``extra_body``
        passthrough because the OpenAI schema has no such field.

        Unlike the parent, the object is sent on tool turns too — DeepSeek
        documents thinking and tool calls as compatible, and an agentic turn is
        exactly where the cost of reasoning is worth steering.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        thinking = self._resolve_thinking(context)
        if thinking is not None:
            kwargs.setdefault("extra_body", {})["thinking"] = thinking

    def _resolve_thinking(self, context: AIContext) -> dict[str, Any] | None:
        """Build DeepSeek's ``thinking`` object for this turn, or ``None`` to omit it.

        ``thinking_budget`` gates per-turn and outranks the config (mirrors the
        Mistral and OpenRouter providers): ``0`` disables thinking, any positive
        value enables it. The *size* of the budget is deliberately dropped —
        DeepSeek's API ignores token budgets, and translating one into an effort
        tier would invent a mapping the vendor does not publish. ``None`` falls
        back to ``enable_thinking``, and with neither set the request stays
        silent so the model's own default (thinking on) applies.
        """
        budget = context.thinking_budget
        enabled = self._config.enable_thinking if budget is None else budget > 0
        effort = self._config.reasoning_effort
        if enabled is False:
            return {"type": "disabled"}
        if enabled is None and effort is None:
            return None
        thinking: dict[str, Any] = {"type": "enabled"}
        if effort is not None:
            thinking["reasoning_effort"] = effort
        return thinking

    @staticmethod
    def _usage_from(raw: Any) -> dict[str, int]:
        """Map DeepSeek's usage object to roomkit's canonical counters.

        DeepSeek reports cache hits as ``prompt_cache_hit_tokens`` /
        ``prompt_cache_miss_tokens`` rather than OpenAI's
        ``prompt_tokens_details.cached_tokens``. The inherited implementation
        looks only at the OpenAI shape, so every cached token would be counted
        — and priced — as ordinary input: a 50x overcharge at DeepSeek's cache
        rate. An endpoint that reports neither shape (a proxy) falls back to the
        parent.
        """
        hit = getattr(raw, "prompt_cache_hit_tokens", None)
        if hit is None:
            return OpenAIAIProvider._usage_from(raw)
        miss = getattr(raw, "prompt_cache_miss_tokens", None)
        if miss is None:
            miss = max((raw.prompt_tokens or 0) - hit, 0)
        usage = {"input_tokens": miss, "output_tokens": raw.completion_tokens or 0}
        if hit:
            usage["cache_read_input_tokens"] = hit
        return usage
