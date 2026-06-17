"""Curated catalog of popular OpenRouter model slugs.

Hand-maintained snapshot returned by ``OpenRouterAIProvider.available_models``.
OpenRouter aggregates 300+ models across many upstream providers, so this is a
deliberately small, representative slice of current flagships — *not* an
exhaustive list. The ids and context windows were sourced from OpenRouter's
live ``/api/v1/models`` endpoint (June 2026); the lineup moves fast, so call
``OpenRouterAIProvider.list_models()`` for the authoritative, always-current
set with live metadata.

``supports_vision`` is set only where the model is unambiguously multimodal
(Anthropic / OpenAI / Google flagships); it is left ``None`` ("unknown")
elsewhere rather than guessed.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="anthropic/claude-opus-4.8",
        display_name="Claude Opus 4.8",
        context_window=1_000_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-4.5",
        display_name="Claude Sonnet 4.5",
        context_window=1_000_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="openai/gpt-5.5",
        display_name="GPT-5.5",
        context_window=1_050_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="openai/gpt-5.5-pro",
        display_name="GPT-5.5 Pro",
        context_window=1_050_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="google/gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=1_048_576,
        supports_vision=True,
    ),
    ModelInfo(
        id="x-ai/grok-4.20",
        display_name="Grok 4.20",
        context_window=2_000_000,
    ),
    ModelInfo(
        id="deepseek/deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        context_window=1_048_576,
    ),
    ModelInfo(
        id="mistralai/mistral-medium-3.5",
        display_name="Mistral Medium 3.5",
        context_window=262_144,
    ),
    ModelInfo(
        id="qwen/qwen3.7-max",
        display_name="Qwen3.7 Max",
        context_window=1_000_000,
    ),
]
