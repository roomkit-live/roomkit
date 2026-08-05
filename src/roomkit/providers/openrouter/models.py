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
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="anthropic/claude-opus-5",
        display_name="Claude Opus 5",
        context_window=1_000_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="anthropic/claude-opus-4.8",
        display_name="Claude Opus 4.8",
        context_window=1_000_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-5",
        display_name="Claude Sonnet 5",
        context_window=1_000_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="openai/gpt-5.6-sol",
        display_name="GPT-5.6 Sol",
        context_window=1_050_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="openai/gpt-5.6-terra",
        display_name="GPT-5.6 Terra",
        context_window=1_050_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="google/gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        context_window=1_048_576,
        supports_vision=True,
    ),
    ModelInfo(
        id="google/gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=1_048_576,
        supports_vision=True,
    ),
    ModelInfo(
        id="x-ai/grok-4.5",
        display_name="Grok 4.5",
        context_window=500_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="x-ai/grok-4.20",
        display_name="Grok 4.20",
        context_window=2_000_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="mistralai/mistral-medium-3-5",
        display_name="Mistral Medium 3.5",
        context_window=262_144,
        supports_vision=True,
    ),
    ModelInfo(
        id="deepseek/deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        context_window=1_048_576,
    ),
    ModelInfo(
        id="qwen/qwen3.8-max",
        display_name="Qwen3.8 Max",
        context_window=1_000_000,
        supports_vision=True,
    ),
]
