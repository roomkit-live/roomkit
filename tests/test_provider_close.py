"""Every text provider's ``close()`` releases the HTTP client its SDK opened
(RMK-651).

A ``close()`` guarded on an attribute the SDK does not have closes nothing and
says nothing: ``MistralAIProvider`` asked for a ``close()`` the mistralai client
never had. Each provider is built over its real SDK, no network, and the HTTP
client behind it must report closed afterwards. PolarGrid is absent on purpose:
its SDK opens a client per request and holds none between them.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig

Built = tuple[AIProvider, Callable[[], bool]]
"""A provider, and whether the HTTP client its SDK opened is closed."""


def _anthropic() -> Built:
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    return provider, provider._client.is_closed


def _openai() -> Built:
    provider = OpenAIAIProvider(OpenAIConfig(api_key="k", model="gpt-4.1"))
    return provider, provider._client.is_closed


def _gemini() -> Built:
    provider = GeminiAIProvider(GeminiConfig(api_key="k"))
    http = provider._http
    assert http is not None
    return provider, lambda: http.is_closed


def _mistral() -> Built:
    provider = MistralAIProvider(MistralConfig(api_key="k", model="mistral-large-latest"))
    http = provider._client.sdk_configuration.async_client
    return provider, lambda: http.is_closed


def _ollama() -> Built:
    provider = OllamaAIProvider(OllamaConfig(model="qwen3:8b"))
    http = provider._client._client
    return provider, lambda: http.is_closed


@pytest.mark.parametrize(
    "build",
    [_anthropic, _openai, _gemini, _mistral, _ollama],
    ids=["anthropic", "openai", "gemini", "mistral", "ollama"],
)
async def test_close_releases_the_sdk_http_client(build: Callable[[], Built]) -> None:
    provider, is_closed = build()
    assert not is_closed()

    await provider.close()

    assert is_closed()
