"""vLLM provider — an OpenAI-compatible provider pointed at a local server."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.vllm.config import VLLMConfig

__all__ = ["VLLMConfig", "create_vllm_provider"]


class _VLLMProvider(OpenAIAIProvider):
    """OpenAI-compatible provider whose model metadata is the server's, not OpenAI's.

    vLLM speaks the OpenAI Chat Completions API, so the whole wire format is
    inherited. What is *not* inherited is OpenAI's answer to "which models
    exist": a local server runs whatever weights someone loaded onto it, and
    that has no relationship to OpenAI's hosted lineup.

    Private because it adds nothing a caller configures — build it through
    :func:`create_vllm_provider`, which is the documented entry point.
    """

    _install_extra: ClassVar[str] = "vllm"

    @property
    def name(self) -> str:
        return "vllm"

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Empty: a local server's models are not knowable offline.

        Inheriting OpenAI's list would describe someone else's hosted models,
        and hand out one of their context windows for any id that happened to
        collide. Empty makes :attr:`context_window` ``None``, which is the
        honest answer for a model roomkit has never heard of. Call
        :meth:`list_models` for the real set — it queries ``/v1/models`` on the
        server, which does know.
        """
        return []

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature, then this turn's reasoning settings.

        vLLM renders the model's chat template server-side, so reasoning is
        steered through ``chat_template_kwargs`` rather than the top-level
        ``reasoning_effort`` the OpenAI parent sends — which a local template
        does not read. The parent's field is therefore not forwarded.

        Unlike the parent, the settings are sent on tool turns too: nothing on
        a local server couples reasoning to the absence of tools, and an
        agentic turn is exactly where its cost is worth steering.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        template_kwargs = self._resolve_template_kwargs(context)
        if template_kwargs:
            kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = template_kwargs

    def _resolve_template_kwargs(self, context: AIContext) -> dict[str, Any]:
        """Merge this turn's reasoning settings over the configured ones.

        The configured baseline lives in ``extra_body`` — put there by
        :func:`create_vllm_provider` from the config's own knobs, or written by
        hand for a template whose kwargs :class:`VLLMConfig` does not model.
        Merging rather than replacing means a per-turn switch cannot silently
        drop a configured effort it says nothing about. Empty when neither
        layer sets anything, so the request stays silent and the model's own
        default applies.
        """
        configured = (self._config.extra_body or {}).get("chat_template_kwargs", {})
        resolved: dict[str, Any] = dict(configured)
        if context.enable_thinking is not None:
            resolved["enable_thinking"] = context.enable_thinking
        if context.reasoning_effort is not None:
            resolved["reasoning_effort"] = context.reasoning_effort
        return resolved

    @property
    def supports_vision(self) -> bool:
        """True: whether the loaded weights read images is the server's call.

        With no offline catalog there is nothing to resolve the question
        against, and the parent's fallback prefixes are OpenAI's own model
        names — which no local model id matches, so inheriting it would report
        every vLLM deployment as text-only and drop images before they reached
        the wire. Passing them through lets a multimodal server work and a
        text-only one answer with an error, the same bargain Ollama and Mistral
        make.
        """
        return True


def create_vllm_provider(config: VLLMConfig) -> OpenAIAIProvider:
    """Create an OpenAI-compatible AI provider pointed at a local vLLM server.

    The returned provider is an :class:`OpenAIAIProvider` subclass: identical
    on the wire, but reporting model metadata for *your* server rather than
    OpenAI's hosted catalog. The ``openai`` SDK is imported lazily, when the
    provider is instantiated.

    Note:
        ``available_models()`` is empty by design — nothing offline can know
        what a local server loaded. Use
        :meth:`~OpenAIAIProvider.list_models`, which queries the server's
        ``/v1/models`` endpoint.

    Args:
        config: vLLM connection settings.

    Returns:
        A provider configured for the local vLLM server.
    """
    template_kwargs = config.chat_template_kwargs()
    sampling = config.sampling_body()
    explicit_body = dict(config.extra_body) if config.extra_body else {}
    # The typed sampling fields lay the floor and an explicit extra_body entry
    # wins over them, same rule as the template kwargs below: extra_body is the
    # escape hatch for a server this config does not model, so it must always
    # be able to override what the config decided.
    extra_body: dict[str, Any] | None = {**sampling, **explicit_body} or None
    if template_kwargs:
        extra_body = extra_body or {}
        extra_body["chat_template_kwargs"] = {
            **template_kwargs,
            **explicit_body.get("chat_template_kwargs", {}),
        }

    openai_config = OpenAIConfig(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        connect_timeout=config.connect_timeout,
        max_retries=config.max_retries,
        include_stream_usage=config.include_stream_usage,
        default_headers=config.headers,
        extra_body=extra_body,
    )
    return _VLLMProvider(openai_config)
