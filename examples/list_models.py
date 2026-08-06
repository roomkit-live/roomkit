"""Discover the models each AI provider supports.

Every ``AIProvider`` exposes two entry points, and they answer different
questions:

- ``list_models()`` — **discovery**. A live query against the provider's API,
  so it is always current and reflects your own account: entitlements,
  regional availability, whichever weights someone pulled onto a local server.
  Falls back to the offline list for providers without a models endpoint.
- ``available_models()`` — what RoomKit knows *without* a network call (a
  classmethod: no API key, no network, no provider SDK). It exists because
  ``context_window`` is a sync property — history trimming needs a number
  before any request goes out and cannot await one. A lineup turns over faster
  than a release cycle, so treat this as offline metadata, never as the
  authoritative list of what a provider offers.

Each entry also carries the vendor's list price (``ModelInfo.pricing``), so
pricing a turn needs no second table: ``pricing.cost_for(response.usage)``
answers directly. Keeping the rates beside the ids is what stops the two
drifting — a rate sheet maintained elsewhere silently bills a newly added
model at zero.

Run with:
    uv run python examples/list_models.py

Set OPENAI_API_KEY to also see a live ``list_models()`` call against OpenAI, or
OPENROUTER_API_KEY to list every model OpenRouter exposes (300+).
"""

from __future__ import annotations

import asyncio
import os

from roomkit.providers.ai.base import ModelInfo, ModelPricing
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
    return f"  {vision}{model.id:<32} ctx={ctx:<12}{_format_price(model.pricing):<26}{flag}"


def _format_price(pricing: ModelPricing | None) -> str:
    """One model's rates, or why there are none.

    Ollama serves weights pulled onto your own hardware and PolarGrid runs on
    private edges: neither publishes a per-token price, so those catalogs say
    so rather than claiming zero.
    """
    if pricing is None:
        return "no published rate"
    rates = f"${pricing.input_per_million:g}/${pricing.output_per_million:g} per M"
    if pricing.cache_read_per_million is not None:
        rates += f" (cached in ${pricing.cache_read_per_million:g})"
    return rates


def show_curated_catalogs() -> None:
    """Print the offline catalog for every provider — no key required."""
    for label, provider_cls in CURATED_PROVIDERS.items():
        models = provider_cls.available_models()
        print(f"\n{label} — {len(models)} curated models")
        for model in models:
            print(_format(model))


def show_what_a_turn_costs() -> None:
    """Price a real turn's usage from the catalog — no key, no network.

    The counters are the ones a provider reports in ``AIResponse.usage``, and
    the cached prefix is deliberately separate from fresh input: at Anthropic's
    rates it is a tenth the price, so folding the two together overstates the
    bill by an order of magnitude on a long conversation.
    """
    usage = {
        "input_tokens": 12_693,
        "output_tokens": 20,
        "cache_read_input_tokens": 48_000,
    }
    print(f"\nWhat one turn costs — usage {usage}")
    for provider_cls, model_id in (
        (GeminiAIProvider, "gemini-3.6-flash"),
        (AnthropicAIProvider, "claude-opus-5"),
        (OpenAIAIProvider, "gpt-5.6-sol"),
    ):
        entry = next(m for m in provider_cls.available_models() if m.id == model_id)
        if entry.pricing is None:
            continue
        cost = entry.pricing.cost_for(usage)
        print(
            f"  {model_id:<32} ${cost:.6f} {entry.pricing.currency}"
            f"   (rates verified {entry.pricing.verified})"
        )


async def show_live_openai() -> None:
    """Query OpenAI's live /v1/models when an API key is available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n(set OPENAI_API_KEY to see a live list_models() call)")
        return

    from roomkit.providers.openai.config import OpenAIConfig

    provider = OpenAIAIProvider(OpenAIConfig(api_key=api_key, model="gpt-5.6-sol"))
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
    show_what_a_turn_costs()
    await show_live_openai()
    await show_live_openrouter()


if __name__ == "__main__":
    asyncio.run(main())
