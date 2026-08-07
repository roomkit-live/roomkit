"""Offline metadata for Google Gemini image-generation models.

Hand-maintained list returned by ``GeminiImageProvider.available_models`` — a
counterpart to ``gemini/models.py``, kept apart from it because the two are
disjoint sets: no id here converses, and no id there draws (RFC §25.6). These
are the ``*-image`` models the chat catalog's scope note explicitly excludes.

Sourced from the Gemini API models and pricing docs (ai.google.dev), verified
2026-08-07. Ids carry no ``models/`` prefix, matching the form
``GeminiImageConfig.model`` and the generate-content calls use.

Prices are the paid-tier **standard** rates (not Batch, not Flex, not
Priority), per million tokens. Google quotes one input rate covering text and
image alike — hence the same value on ``input_per_million`` and
``image_input_per_million`` — a text-and-thinking output rate, and a separate,
far higher rate for the pixels. The per-image figures Google also advertises
are that image rate times the token count of a given resolution
(1120 tokens at 1K/2K), so they are the same price stated twice, not a second
unit.

Context windows are omitted deliberately: an image model's published limit
describes a prompt these providers never trim against, and an unknown window
is safer than one nobody reconciles.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing
from roomkit.providers.image.base import IMAGE_GEN_CAPABILITY

_VERIFIED = date(2026, 8, 7)
_CAPS = [IMAGE_GEN_CAPABILITY, "edit"]

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3-pro-image",
        display_name="Gemini 3 Pro Image (Nano Banana Pro)",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=12.0,
            cache_read_per_million=0.2,
            image_input_per_million=2.0,
            image_output_per_million=120.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.1-flash-image",
        display_name="Gemini 3.1 Flash Image (Nano Banana 2)",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=0.5,
            output_per_million=3.0,
            image_input_per_million=0.5,
            image_output_per_million=60.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.1-flash-lite-image",
        display_name="Gemini 3.1 Flash Lite Image (Nano Banana 2 Lite)",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=0.25,
            output_per_million=1.5,
            image_input_per_million=0.25,
            image_output_per_million=30.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-2.5-flash-image",
        display_name="Gemini 2.5 Flash Image (Nano Banana)",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=0.3,
            output_per_million=2.5,
            cache_read_per_million=0.03,
            image_input_per_million=0.3,
            image_output_per_million=30.0,
            verified=_VERIFIED,
        ),
    ),
]
