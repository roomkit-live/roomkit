"""vLLM provider — thin wrapper that returns an OpenAI-compatible provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.providers.vllm.config import VLLMConfig

if TYPE_CHECKING:
    from roomkit.providers.openai.ai import OpenAIAIProvider

__all__ = ["VLLMConfig", "create_vllm_provider"]


def create_vllm_provider(config: VLLMConfig) -> OpenAIAIProvider:
    """Create an OpenAI-compatible AI provider pointed at a local vLLM server.

    This is a factory function — no new subclass is needed because vLLM
    implements the OpenAI Chat Completions API.  The ``openai`` SDK is
    imported lazily when :class:`OpenAIAIProvider` is instantiated.

    Note:
        The returned provider inherits :meth:`OpenAIAIProvider.available_models`,
        whose curated catalog lists OpenAI's *hosted* models — not whatever a
        local vLLM server serves. For a vLLM deployment, call
        :meth:`~OpenAIAIProvider.list_models` instead: it queries the server's
        ``/v1/models`` endpoint and returns the models actually loaded there.

    Args:
        config: vLLM connection settings.

    Returns:
        An :class:`OpenAIAIProvider` configured for the local vLLM server.
    """
    from roomkit.providers.openai.ai import OpenAIAIProvider
    from roomkit.providers.openai.config import OpenAIConfig

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
    return OpenAIAIProvider(openai_config)
