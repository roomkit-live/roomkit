"""Meta Muse Spark chat provider — Chat Completions on the Meta Model API."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.ai.reasoning import thinking_switch, turn_setting
from roomkit.providers.meta.config import MetaConfig
from roomkit.providers.meta.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider

_MODELS_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}

# The chat models among what /v1/models lists next to them (Muse Image, Muse
# Voice Transcribe, SAM).
_CHAT_PREFIX = "muse-spark"


class MetaAIProvider(OpenAIAIProvider):
    """AI provider on Meta's Muse Spark, through its OpenAI-compatible Chat Completions.

    Subclasses :class:`~roomkit.providers.openai.ai.OpenAIAIProvider`: message
    building, tool calls, streaming, usage (reasoning and cached tokens
    included) and error mapping are inherited. What is Meta's own:

    * Muse Spark always reasons, so ``reasoning_effort`` rides every request,
      tool turns included — it is the only lever over the cost and latency of
      a turn. ``"none"`` is refused by the service and sent as ``"minimal"``
      (3.0 s against 4.6 s at ``"low"`` on a one-line answer, 2026-09-27).
    * ``/v1/models`` lists the account's image, speech and segmentation models
      too; :meth:`list_models` keeps the chat ones.

    Example::

        provider = MetaAIProvider(MetaConfig(api_key="...", reasoning_effort="low"))
    """

    _config: MetaConfig
    _install_extra: ClassVar[str] = "meta"

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "meta"

    @property
    def supports_vision(self) -> bool:
        """Every Muse Spark model reads images, a newer id than the catalog too."""
        info = _MODELS_BY_ID.get(self._config.model)
        if info is None or info.supports_vision is None:
            return True
        return info.supports_vision

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Muse Spark models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """The chat models the account can use, with the catalog's metadata."""
        return [m for m in await super().list_models() if m.id.startswith(_CHAT_PREFIX)]

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and Meta's ``reasoning_effort`` to a request.

        The turn's own effort outranks the configured one. It is sent on tool
        turns too (the OpenAI parent's rule is its own catalogue's), and
        ``"none"`` becomes ``"minimal"``: the service cannot turn reasoning off
        and answers ``"none"`` with a 400.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        effort = turn_setting(context.reasoning_effort, self._config.reasoning_effort)
        if thinking_switch(context) is False:
            effort = "none"  # a turn that switches reasoning off asks for the least of it
        if effort is not None:
            kwargs["reasoning_effort"] = "minimal" if effort == "none" else effort
