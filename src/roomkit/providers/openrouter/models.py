"""Offline metadata for a slice of OpenRouter model slugs.

Hand-maintained snapshot returned by ``OpenRouterAIProvider.available_models``.
OpenRouter aggregates 300+ models across many upstream providers, so an offline
list can only ever be a small, representative slice of current flagships — it
is not, and cannot be, the discovery surface. Call
``OpenRouterAIProvider.list_models()`` for that: OpenRouter's ``/api/v1/models``
is public and returns the whole set with live metadata.

Ids, context windows, and modalities were read from that endpoint on
2026-08-05.

``supports_vision`` reflects the ``image`` input modality the endpoint reports,
and is left ``None`` ("unknown") for text-only models rather than set to False,
matching how :class:`~roomkit.providers.ai.base.ModelInfo` treats an unknown
capability.

Prices are OpenRouter's own, from the same endpoint on the same date — here
the aggregator *is* the seller, so its ``pricing`` object is the rate card,
not a mirror of someone else's. It can differ from the upstream vendor's own
list price, and does: ``openai/gpt-5.6-terra`` resells at $1/$6 where OpenAI
charges $2/$12. Both are right for whoever bills.

One field needs normalization. OpenRouter reports only the five-minute storage
premium in ``input_cache_write`` for explicit Gemini caching, while billing a
write as ordinary input plus that premium. RoomKit's canonical cache-write
counter is disjoint from ordinary input, so the Google entries carry the full
sum. Anthropic and OpenAI already report the full per-token write charge.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 8, 5)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="anthropic/claude-opus-5",
        display_name="Claude Opus 5",
        context_window=1_000_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="anthropic/claude-opus-4.8",
        display_name="Claude Opus 4.8",
        context_window=1_000_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-5",
        display_name="Claude Sonnet 5",
        context_window=1_000_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=10.0,
            cache_read_per_million=0.2,
            cache_write_per_million=2.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="openai/gpt-5.6-sol",
        display_name="GPT-5.6 Sol",
        context_window=1_050_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=30.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="openai/gpt-5.6-terra",
        display_name="GPT-5.6 Terra",
        context_window=1_050_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.0,
            output_per_million=6.0,
            cache_read_per_million=0.1,
            cache_write_per_million=1.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="google/gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        context_window=1_048_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=7.5,
            cache_read_per_million=0.15,
            cache_write_per_million=1.5833333,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="google/gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=1_048_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=9.0,
            cache_read_per_million=0.15,
            cache_write_per_million=1.5833333,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="x-ai/grok-4.5",
        display_name="Grok 4.5",
        context_window=500_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=6.0,
            cache_read_per_million=0.3,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="x-ai/grok-4.20",
        display_name="Grok 4.20",
        context_window=2_000_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=2.5,
            cache_read_per_million=0.2,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="mistralai/mistral-medium-3-5",
        display_name="Mistral Medium 3.5",
        context_window=262_144,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=7.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="deepseek/deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        context_window=1_048_576,
        pricing=ModelPricing(
            input_per_million=0.435,
            output_per_million=0.87,
            cache_read_per_million=0.003625,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="qwen/qwen3.8-max",
        display_name="Qwen3.8 Max",
        context_window=1_000_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=6.0,
            cache_read_per_million=0.25,
            cache_write_per_million=2.5,
            verified=_VERIFIED,
        ),
    ),
]
