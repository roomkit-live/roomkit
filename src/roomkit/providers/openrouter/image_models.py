"""Offline metadata for a slice of OpenRouter image model slugs.

Hand-maintained snapshot returned by ``OpenRouterImageProvider.available_models``
— a counterpart to ``openrouter/models.py``, kept apart from it because the two
are disjoint sets: no slug here converses, and no slug there draws (RFC §25.6).
OpenRouter's Image API aggregates 40+ models across a dozen vendors, so an
offline list can only ever be a small, representative slice of current
flagships; the discovery surface is OpenRouter's public
``GET /api/v1/images/models``, which returns the whole set with each model's
live parameter constraints.

Ids and reference-image support were read from that endpoint on 2026-08-21.

No entry carries a ``pricing``, deliberately. The Image API quotes its rate
cards per endpoint (``GET /api/v1/images/models/{id}/endpoints``), mostly in
per-image units that vary by quality and resolution — a shape
:class:`~roomkit.providers.ai.base.ModelPricing`'s per-token rates cannot state
without inventing a unit. What a call actually cost is authoritative anyway:
OpenRouter reports the billed amount on every response, and the provider
surfaces it as ``ImageResult.usage["cost"]``.

``edit`` is tagged on every entry because every model in this slice accepts
``input_references``; the live endpoint states each model's maximum.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.image.base import IMAGE_GEN_CAPABILITY

_CAPS = [IMAGE_GEN_CAPABILITY, "edit"]

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="google/gemini-3.1-flash-image",
        display_name="Gemini 3.1 Flash Image (Nano Banana 2)",
        supports_vision=True,
        capabilities=_CAPS,
    ),
    ModelInfo(
        id="google/gemini-3-pro-image",
        display_name="Gemini 3 Pro Image (Nano Banana Pro)",
        supports_vision=True,
        capabilities=_CAPS,
    ),
    ModelInfo(
        id="openai/gpt-image-2",
        display_name="GPT Image 2",
        supports_vision=True,
        capabilities=_CAPS,
    ),
    ModelInfo(
        id="x-ai/grok-imagine-image-2.0",
        display_name="Grok Imagine Image 2.0",
        supports_vision=True,
        capabilities=_CAPS,
    ),
    ModelInfo(
        id="bytedance-seed/seedream-5-0-pro",
        display_name="Seedream 5.0 Pro",
        supports_vision=True,
        capabilities=_CAPS,
    ),
    ModelInfo(
        id="black-forest-labs/flux.2-pro",
        display_name="FLUX.2 Pro",
        supports_vision=True,
        capabilities=_CAPS,
    ),
]
