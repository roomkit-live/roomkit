"""Offline metadata for OpenAI chat/multimodal models.

Hand-maintained list returned by ``OpenAIAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what OpenAI currently offers. Call ``OpenAIAIProvider.list_models()``
for that; it queries the account's ``/v1/models``.

Sourced from the OpenAI models and deprecations docs
(developers.openai.com/api/docs/models, .../deprecations), verified 2026-08-05.

Scope is the chat/responses-capable text + multimodal models; embeddings,
audio (whisper/tts), and image-generation models are intentionally omitted.

``deprecated=True`` marks a model OpenAI has given a shutdown date. Models
already past their shutdown are removed outright rather than flagged — a dead
id is a 404, and keeping it here only invites one.

The GPT-5.6 tier reaches its "pro" depth through ``reasoning.mode: "pro"`` on
the base model rather than through a separate id, which is why — unlike 5.4
and 5.5 — it has no ``-pro`` entries.

Prices are the standard synchronous rates from OpenAI's pricing page
(developers.openai.com/api/docs/pricing), read 2026-08-05 — not the Batch
column, which is half of them, and not Flex. ``cache_read`` is OpenAI's
cached-input rate, applied automatically to a repeated prefix; the ``pro``
tiers publish none because they do not cache. GPT-5.6 also publishes an
explicit cache-write rate and higher prices beyond 272k input tokens; both
are represented in its entries. Earlier models leave ``cache_write`` unset.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_CTX_1M = 1_050_000
_VERIFIED = date(2026, 8, 5)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-5.6-sol",
        display_name="GPT-5.6 Sol",
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=30.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.6-terra",
        display_name="GPT-5.6 Terra",
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=12.0,
            cache_read_per_million=0.2,
            cache_write_per_million=2.5,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.6-luna",
        display_name="GPT-5.6 Luna",
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.2,
            output_per_million=1.2,
            cache_read_per_million=0.02,
            cache_write_per_million=0.25,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.5",
        display_name="GPT-5.5",
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=30.0,
            cache_read_per_million=0.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.5-pro",
        display_name="GPT-5.5 Pro",
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=30.0,
            output_per_million=180.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4",
        display_name="GPT-5.4",
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.5,
            output_per_million=15.0,
            cache_read_per_million=0.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4-pro",
        display_name="GPT-5.4 Pro",
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=30.0,
            output_per_million=180.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        display_name="GPT-5.4 mini",
        context_window=400_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.75,
            output_per_million=4.5,
            cache_read_per_million=0.075,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4-nano",
        display_name="GPT-5.4 nano",
        context_window=400_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.2,
            output_per_million=1.25,
            cache_read_per_million=0.02,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.1",
        display_name="GPT-5.1",
        context_window=400_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=10.0,
            cache_read_per_million=0.125,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4.1",
        display_name="GPT-4.1",
        context_window=1_047_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=8.0,
            cache_read_per_million=0.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        context_window=1_047_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.4,
            output_per_million=1.6,
            cache_read_per_million=0.1,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4.1-nano",
        display_name="GPT-4.1 nano",
        context_window=1_047_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.1,
            output_per_million=0.4,
            cache_read_per_million=0.025,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4o",
        display_name="GPT-4o",
        context_window=128_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.5,
            output_per_million=10.0,
            cache_read_per_million=1.25,
            verified=_VERIFIED,
        ),
    ),
    # --- Deprecated: OpenAI has announced a shutdown date -----------------
    ModelInfo(
        id="gpt-5",
        display_name="GPT-5",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=10.0,
            cache_read_per_million=0.125,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5-mini",
        display_name="GPT-5 mini",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=0.25,
            output_per_million=2.0,
            cache_read_per_million=0.025,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5-nano",
        display_name="GPT-5 nano",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=0.05,
            output_per_million=0.4,
            cache_read_per_million=0.005,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.2",
        display_name="GPT-5.2",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=1.75,
            output_per_million=14.0,
            cache_read_per_million=0.175,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="o3",
        display_name="o3",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=8.0,
            cache_read_per_million=0.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="o3-pro",
        display_name="o3-pro",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=20.0,
            output_per_million=80.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="o4-mini",
        display_name="o4-mini",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=1.1,
            output_per_million=4.4,
            cache_read_per_million=0.275,
            verified=_VERIFIED,
        ),
    ),
]
