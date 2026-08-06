"""Offline metadata for Mistral chat/multimodal models.

Hand-maintained list returned by ``MistralAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what Mistral currently offers. Call ``MistralAIProvider.list_models()``
for that; it queries the account's models endpoint.

Sourced from the Mistral model cards (docs.mistral.ai/models), verified
2026-08-05.

Scope is general chat/multimodal models; embeddings, moderation, OCR, audio
(Voxtral), and code-completion (Codestral/Devstral) models are omitted. The whole
current Mistral 3 family is multimodal, and every member carries a 256k window
except Ministral 3 3B, which is half that. The deprecated 128k-tier models leave
``context_window`` as ``None`` — Mistral documents them only as "128k" without a
firm token integer.

Both the dated ids and their ``-latest`` aliases are listed so either form
resolves here.

Prices come from Mistral's API pricing table (mistral.ai/pricing/api), read
2026-08-05. Prompt-cache reads on current models cost 10% of ordinary input;
cache population carries no separate per-token write rate.
``pixtral-large-latest`` carries no price at all: it reached its retirement
date (2026-05-31) and Mistral no longer quotes it — an absent price is the
honest answer, an invented one is not.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_CTX = 262_144
_VERIFIED = date(2026, 8, 5)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="mistral-large-latest",
        display_name="Mistral Large 3",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.5,
            output_per_million=1.5,
            cache_read_per_million=0.05,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="mistral-large-2512",
        display_name="Mistral Large 3",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.5,
            output_per_million=1.5,
            cache_read_per_million=0.05,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="mistral-medium-latest",
        display_name="Mistral Medium 3.5",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=7.5,
            cache_read_per_million=0.15,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="mistral-medium-3-5",
        display_name="Mistral Medium 3.5",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=7.5,
            cache_read_per_million=0.15,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="mistral-small-latest",
        display_name="Mistral Small 4",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.15,
            output_per_million=0.6,
            cache_read_per_million=0.015,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="mistral-small-2603",
        display_name="Mistral Small 4",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.15,
            output_per_million=0.6,
            cache_read_per_million=0.015,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="ministral-14b-latest",
        display_name="Ministral 3 14B",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.2,
            output_per_million=0.2,
            cache_read_per_million=0.02,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="ministral-8b-latest",
        display_name="Ministral 3 8B",
        context_window=_CTX,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.15,
            output_per_million=0.15,
            cache_read_per_million=0.015,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="ministral-3b-latest",
        display_name="Ministral 3 3B",
        context_window=131_072,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.1,
            output_per_million=0.1,
            cache_read_per_million=0.01,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="magistral-medium-latest",
        display_name="Magistral Medium 1.2",
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=5.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="magistral-small-latest",
        display_name="Magistral Small 1.2",
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=0.5,
            output_per_million=1.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="pixtral-large-latest",
        display_name="Pixtral Large",
        supports_vision=True,
        deprecated=True,
    ),
]
