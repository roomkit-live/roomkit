"""Offline metadata for OpenAI image-generation models.

Hand-maintained list returned by ``OpenAIImageProvider.available_models`` — a
counterpart to ``openai/models.py``, kept apart from it because the two are
disjoint sets: no id here converses, and no id there draws (RFC §25.6).

Sourced from the OpenAI images guide and the ``ImageModel`` literal shipped by
the ``openai`` SDK (2.48.0), verified 2026-08-07.

Scope is the GPT image models on ``/v1/images``. ``dall-e-2`` and ``dall-e-3``
are omitted: they are billed a flat amount per image rather than per token, so
the rates here could not describe them without inventing a unit. Dated
snapshots (``gpt-image-2-2026-04-21``) are omitted for the same reason the chat
catalog omits most of them — the undated id is what a caller configures.

Prices are the standard synchronous rates from OpenAI's pricing page
(developers.openai.com/api/docs/pricing), read 2026-08-07 — not the Batch
column, which is half of them. OpenAI quotes six numbers per model: text and
image input, their cached-input rates, and text and image output. Four are
represented. The cached *image* input rate is not: prompt caching applies to
the Responses API's image tool, not to the generation endpoint this provider
calls, so roomkit never reports a counter it would price. A model with no text
output rate ("-" in OpenAI's table) generates images only, and carries
``output_per_million=0`` rather than a guess.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing
from roomkit.providers.image.base import IMAGE_GEN_CAPABILITY

_VERIFIED = date(2026, 8, 7)
_CAPS = [IMAGE_GEN_CAPABILITY, "edit"]

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-image-2",
        display_name="GPT Image 2",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=0.0,
            cache_read_per_million=1.25,
            image_input_per_million=8.0,
            image_output_per_million=30.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-image-1.5",
        display_name="GPT Image 1.5",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=10.0,
            cache_read_per_million=1.25,
            image_input_per_million=8.0,
            image_output_per_million=32.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="chatgpt-image-latest",
        display_name="ChatGPT Image (latest)",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=10.0,
            cache_read_per_million=1.25,
            image_input_per_million=8.0,
            image_output_per_million=32.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-image-1",
        display_name="GPT Image 1",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=0.0,
            cache_read_per_million=1.25,
            image_input_per_million=10.0,
            image_output_per_million=40.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-image-1-mini",
        display_name="GPT Image 1 mini",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=0.0,
            cache_read_per_million=0.2,
            image_input_per_million=2.5,
            image_output_per_million=8.0,
            verified=_VERIFIED,
        ),
    ),
]
