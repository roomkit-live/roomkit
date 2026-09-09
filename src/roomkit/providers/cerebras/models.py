"""Offline metadata for models listed by Cerebras, verified 2026-09-09.

Sources: https://api.cerebras.ai/public/v1/models (ids, capabilities, limits,
GPT OSS and Gemma prices) and
https://inference-docs.cerebras.ai/models/qwen-3.8-27b (Qwen prices).
The public API reports zero Qwen prices, while its model card quotes the
Developer rates below. Qwen's conservative 65,536-token public limit fits the
trial tier too; the card advertises 128k for paid tiers. Gemma is listed by the
public API but documented for dedicated endpoints. Account availability is
answered by ``list_models()``, not this snapshot.

Cache reads cost the ordinary input rate, per
https://inference-docs.cerebras.ai/capabilities/prompt-caching.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 9, 9)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-oss-120b",
        display_name="GPT OSS 120B",
        context_window=131_072,
        supports_vision=False,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=0.35,
            output_per_million=0.75,
            cache_read_per_million=0.35,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="qwen-3.8-27b",
        display_name="Qwen 3.8 27B",
        context_window=65_536,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=0.99,
            output_per_million=1.49,
            cache_read_per_million=0.99,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemma-4-31b",
        display_name="Gemma 4 31B",
        context_window=131_072,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=0.99,
            output_per_million=1.49,
            cache_read_per_million=0.99,
            verified=_VERIFIED,
        ),
    ),
]
