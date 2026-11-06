"""xAI (Grok) AI provider — generates responses via xAI's OpenAI-compatible API."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.ai.reasoning import turn_setting
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.xai.config import XAIConfig
from roomkit.providers.xai.models import MODELS

_MODELS_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}


class XAIAIProvider(OpenAIAIProvider):
    """AI provider using xAI's OpenAI-compatible Chat Completions API.

    Subclasses :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` — xAI
    speaks the OpenAI Chat Completions API verbatim, so message building, tool
    handling, response parsing, streaming, ``/v1/models`` discovery, and client
    construction are all inherited unchanged. Only three things are genuinely
    xAI's own: which models exist, which of them see images, and when a
    reasoning request is legal.

    Distinct from :class:`~roomkit.providers.xai.realtime.XAIRealtimeProvider`,
    which speaks xAI's speech-to-speech WebSocket protocol.

    Example::

        provider = XAIAIProvider(XAIConfig(api_key="xai-...", model="grok-4.7"))
    """

    _config: XAIConfig
    _install_extra: ClassVar[str] = "xai"

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "xai"

    @staticmethod
    def _usage_from(raw: Any) -> dict[str, int]:
        """xAI reports reasoning beside ``completion_tokens`` and bills it as output.

        Its total is prompt + completion + reasoning, where OpenAI's reasoning
        sits inside completion. The reasoning joins ``output_tokens`` only when
        the total says it was reported apart, so a response shaped like
        OpenAI's is never counted twice (RFC §6, usage counters).
        """
        usage = OpenAIAIProvider._usage_from(raw)
        reasoning = usage.get("reasoning_tokens", 0)
        apart = (raw.prompt_tokens or 0) + (raw.completion_tokens or 0) + reasoning
        # An output the server left out stays unknown (RFC §6.4).
        if reasoning and "output_tokens" in usage and getattr(raw, "total_tokens", None) == apart:
            usage["output_tokens"] += reasoning
        return usage

    @property
    def supports_vision(self) -> bool:
        """Whether the configured Grok model accepts image input.

        The override is for the miss, not the hit: the OpenAI parent resolves
        known ids from the catalog — which for this class is Grok's — but falls
        back to prefix-matching *its own* vision model names, and no ``grok-*``
        id can satisfy those. Inheriting the fallback would report an unknown
        Grok model as text-only and silently drop images. Every Grok text model
        in the catalog is multimodal, so an id the catalog does not know (an
        alias like ``grok-latest``, or a model newer than the snapshot) defaults
        to ``True`` rather than losing a capability the family has.
        """
        info = _MODELS_BY_ID.get(self._config.model)
        if info is None or info.supports_vision is None:
            return True
        return info.supports_vision

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Grok chat/multimodal models."""
        return list(MODELS)

    def _supports_reasoning(self) -> bool:
        """Whether the configured model accepts a reasoning request.

        Read positively off the catalog's ``capabilities``: only a model the
        catalog knows *and* describes (non-empty tags) *without* ``"thinking"``
        is treated as refusing reasoning — today just
        ``grok-4.20-0309-non-reasoning``. An unknown id, or one with no tags,
        counts as capable, matching the rest of the Grok line.
        """
        info = _MODELS_BY_ID.get(self._config.model)
        if info is None or not info.capabilities:
            return True
        return "thinking" in info.capabilities

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and xAI's ``reasoning_effort`` to a request.

        On ``/v1/chat/completions`` xAI takes reasoning depth as a top-level
        ``reasoning_effort`` string, exactly like the OpenAI parent (the nested
        ``reasoning: {effort: ...}`` object belongs to ``/v1/responses``). Two
        things differ:

        * It is sent on **tool turns too**, whatever the model. The parent
          reads what an OpenAI model accepts there from its catalogue; xAI
          accepts the pair, and since Grok reasons unconditionally, effort is
          the only lever over the cost of an agentic turn — dropping it on
          exactly the turns that spend the most would defeat the setting.
        * It is withheld from a model the catalog marks as non-reasoning, which
          would reject it.

        The turn's effort outranks the configured one (RFC §6.7).
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        effort = turn_setting(context.reasoning_effort, self._config.reasoning_effort)
        if effort is not None and self._supports_reasoning():
            kwargs["reasoning_effort"] = effort
