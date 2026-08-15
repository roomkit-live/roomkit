"""Qwen AI provider — generates responses via Model Studio's OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.qwen.config import QwenConfig
from roomkit.providers.qwen.models import MODELS

_MODELS_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}


class QwenAIProvider(OpenAIAIProvider):
    """AI provider using Alibaba Model Studio's OpenAI-compatible API.

    Subclasses :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` — Model
    Studio speaks the Chat Completions wire format verbatim, so message
    building, tool handling, response parsing, streaming and client
    construction are all inherited unchanged. Four things are genuinely Qwen's
    own: which models exist, that there is no endpoint to ask, which models see
    images, and how a thinking request is spelled.

    Example::

        provider = QwenAIProvider(
            QwenConfig(api_key="sk-...", model="qwen3.7-max")
        )
    """

    _config: QwenConfig

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "qwen"

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Alibaba's hosted Qwen models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """Return the curated catalog — Model Studio publishes no models endpoint.

        The OpenAI-compatible deployment serves ``/chat/completions`` only, so
        the inherited ``models.list()`` would 404 against a live account. This
        is the one provider where the offline catalog *is* the discovery
        surface; it is also why that catalog carries every id a caller is
        expected to reach.
        """
        return self.available_models()

    @property
    def supports_vision(self) -> bool:
        """Whether the configured Qwen model accepts image input.

        The override is for the miss, not the hit: the OpenAI parent resolves
        known ids from the catalog — which for this class is Qwen's — but falls
        back to prefix-matching *its own* vision model names, and no ``qwen*``
        id can satisfy those. Inheriting that fallback would report an unknown
        Qwen model as text-only and silently drop images. Most of the current
        lineup reads images, so an id the catalog does not know defaults to
        ``True``: a multimodal model then works, and a text-only one answers
        with an error instead of quietly losing the attachment.
        """
        info = _MODELS_BY_ID.get(self._config.model)
        if info is None or info.supports_vision is None:
            return True
        return info.supports_vision

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and Qwen's ``enable_thinking`` / ``thinking_budget``.

        Where the OpenAI parent sends a top-level ``reasoning_effort`` string,
        Qwen takes a boolean switch and a token cap, both outside the OpenAI
        schema and so carried on the SDK's ``extra_body`` passthrough. The
        parent's field is deliberately not forwarded — Model Studio accepts it
        only for the third-party DeepSeek models it hosts, and sending it to a
        Qwen model risks a rejected request for a setting that would not apply.

        Unlike the parent, the switch is sent on tool turns too: nothing in
        Model Studio's API couples thinking to the absence of tools, and an
        agentic turn is exactly where the cost of reasoning is worth steering.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        thinking = self._resolve_thinking(context)
        if thinking:
            kwargs.setdefault("extra_body", {}).update(thinking)

    def _resolve_thinking(self, context: AIContext) -> dict[str, Any]:
        """Build Qwen's thinking fields for this turn, empty to leave them out.

        ``thinking_budget`` gates per-turn and outranks the config (mirrors the
        Mistral and OpenRouter providers): ``0`` disables thinking, any positive
        value enables it *and* caps the trace, which is the one place roomkit's
        budget maps straight onto a vendor parameter instead of being
        approximated. ``None`` falls back to ``enable_thinking``, and with
        neither set the request stays silent so the model's own default applies.
        """
        budget = context.thinking_budget
        if budget is None:
            enabled = self._config.enable_thinking
            return {} if enabled is None else {"enable_thinking": enabled}
        if budget <= 0:
            return {"enable_thinking": False}
        return {"enable_thinking": True, "thinking_budget": budget}
