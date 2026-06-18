"""Discover the models each AI provider supports.

Every ``AIProvider`` exposes two model-discovery entry points:

- ``available_models()`` — a curated, *offline* catalog (a classmethod, so it
  needs no API key, no network, and no provider SDK). Call it to learn which
  models you can configure before wiring anything up.
- ``list_models()`` — a *live* query against the provider's API for the models
  the account/server actually exposes right now. It falls back to the curated
  catalog for providers without a models endpoint.

Run with:
    uv run python examples/list_models.py

Set OPENAI_API_KEY to also see a live ``list_models()`` call against OpenAI, or
OPENROUTER_API_KEY to list every model OpenRouter exposes (300+).
"""

from __future__ import annotations

import asyncio
import os

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.polargrid.ai import PolarGridAIProvider

CURATED_PROVIDERS = {
    "Anthropic": AnthropicAIProvider,
    "OpenAI": OpenAIAIProvider,
    "OpenRouter": OpenRouterAIProvider,
    "Gemini": GeminiAIProvider,
    "Mistral": MistralAIProvider,
    "Ollama": OllamaAIProvider,
    "PolarGrid": PolarGridAIProvider,
}


def _format(model: ModelInfo) -> str:
    ctx = f"{model.context_window:,}" if model.context_window else "?"
    vision = "👁 " if model.supports_vision else "   "
    flag = " (deprecated)" if model.deprecated else ""
    return f"  {vision}{model.id:<32} ctx={ctx:<12}{flag}"


def show_curated_catalogs() -> None:
    """Print the offline catalog for every provider — no key required."""
    for label, provider_cls in CURATED_PROVIDERS.items():
        models = provider_cls.available_models()
        print(f"\n{label} — {len(models)} curated models")
        for model in models:
            print(_format(model))


async def show_live_openai() -> None:
    """Query OpenAI's live /v1/models when an API key is available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n(set OPENAI_API_KEY to see a live list_models() call)")
        return

    from roomkit.providers.openai.config import OpenAIConfig

    provider = OpenAIAIProvider(OpenAIConfig(api_key=api_key))
    try:
        live = await provider.list_models()
        print(f"\nOpenAI live — {len(live)} models reported by the API")
        for model in live:
            print(_format(model))
    finally:
        await provider.close()


async def show_live_openrouter() -> None:
    """Query OpenRouter's live /models — its full catalog, with metadata."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\n(set OPENROUTER_API_KEY to list every model OpenRouter exposes)")
        return

    from roomkit.providers.openrouter.config import OpenRouterConfig

    provider = OpenRouterAIProvider(OpenRouterConfig(api_key=api_key, model="openai/gpt-5.5"))
    try:
        live = await provider.list_models()
        print(f"\nOpenRouter live — {len(live)} models reported by the API")
        for model in live:
            print(_format(model))
    finally:
        await provider.close()


async def main() -> None:
    show_curated_catalogs()
    await show_live_openai()
    await show_live_openrouter()


if __name__ == "__main__":
    asyncio.run(main())
