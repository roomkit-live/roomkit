"""vLLM provider — an OpenAI-compatible provider pointed at a local server."""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo
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
    openai_config = OpenAIConfig(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        max_retries=config.max_retries,
        include_stream_usage=config.include_stream_usage,
        default_headers=config.headers,
        extra_body=config.extra_body,
    )
    return _VLLMProvider(openai_config)
