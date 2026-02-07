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

    Args:
        config: vLLM connection settings.

    Returns:
        An :class:`OpenAIAIProvider` configured for the local vLLM server.
    """
    from pydantic import SecretStr

    from roomkit.providers.openai.ai import OpenAIAIProvider
    from roomkit.providers.openai.config import OpenAIConfig

    openai_config = OpenAIConfig(
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    return OpenAIAIProvider(openai_config)
