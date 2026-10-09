"""Offline metadata for Google Gemini image-generation models.

Hand-maintained list returned by ``GeminiImageProvider.available_models`` — a
counterpart to ``gemini/models.py``, kept apart from it because the two are
disjoint sets: no id here converses, and no id there draws (RFC §25.6). These
are the ``*-image`` models the chat catalog's scope note explicitly excludes.

Sourced from the Gemini API models and pricing docs (ai.google.dev), verified
2026-08-07 (Nano Banana 2.1 and the 2.5 Flash Image shutdown date: 2026-10-09).
Ids carry no ``models/`` prefix, matching the form ``GeminiImageConfig.model``
and the generate-content calls use.

Prices are the paid-tier **standard** rates (not Batch, not Flex, not
Priority), per million tokens. Google quotes one input rate covering text and
image alike — hence the same value on ``input_per_million`` and
``image_input_per_million`` — a text-and-thinking output rate, and a separate,
far higher rate for the pixels. The per-image figures Google also advertises
are that image rate times the token count of a given resolution (1120
tokens at 1K, 1680 at 2K on Flash and Nano Banana 2.1, 1120 at 2K on Pro), so
they are the same price stated twice, not a second unit.

Context windows are omitted deliberately: an image model's published limit
describes a prompt these providers never trim against, and an unknown window
is safer than one nobody reconciles. Cache writes are omitted for the reason
the chat catalog gives: Google bills cache storage by token-hour, and
restating an hourly rate as a per-token one would be a wrong number rather
than a missing one.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing
from roomkit.providers.image.base import IMAGE_GEN_CAPABILITY
from roomkit.providers.image.options import ImageCapabilities, ImageModelInfo

_VERIFIED = date(2026, 8, 7)
_CAPS = [IMAGE_GEN_CAPABILITY, "edit"]

_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
_PRO = ImageCapabilities(
    options=[
        "aspect_ratio",
        "image_size",
        "output_format",
        "previous_interaction_id",
        "store",
        "search_types",
    ],
    aspect_ratios=_RATIOS,
    image_sizes=["1K", "2K", "4K"],
    formats=["png", "jpeg"],
    max_references=14,
    max_images=None,
    continuity=True,
    search_types=["web_search"],
    verified=date(2026, 9, 15),
)
_FLASH = _PRO.model_copy(
    update={
        "options": [*_PRO.options, "thinking_level"],
        "thinking_levels": ["minimal", "high"],
        "aspect_ratios": [*_RATIOS, "1:4", "4:1", "1:8", "8:1"],
        "image_sizes": ["512", "1K", "2K", "4K"],
        "search_types": ["web_search", "image_search"],
    }
)
_LITE = _PRO.model_copy(
    update={
        "image_sizes": ["1K"],
        "search_types": [],
        "options": [option for option in _PRO.options if option != "search_types"]
        + ["thinking_level"],
        "thinking_levels": ["minimal", "high"],
    }
)
# Nano Banana 2.1 keeps 3.1 Flash Image's controls but drops the 512 size,
# returns JPEG only (``image/png`` is a 400, measured 2026-10-09), and adds
# ``medium``, its default thinking level (RMK-654).
_NB21 = _FLASH.model_copy(
    update={
        "image_sizes": ["1K", "2K", "4K"],
        "formats": ["jpeg"],
        "thinking_levels": ["minimal", "medium", "high"],
        "verified": date(2026, 10, 9),
    }
)
_V25 = _LITE.model_copy(
    update={
        "max_references": 3,
        # Google moved the shutdown from 2026-10-02 to 2027-03-15, its earliest
        # date (deprecations page, 2026-10-09).
        "retirement_date": date(2027, 3, 15),
        "thinking_levels": [],
        "options": [option for option in _LITE.options if option != "thinking_level"],
    }
)

MODELS: list[ModelInfo] = [
    ImageModelInfo(
        id="gemini-nano-banana-2.1",
        image=_NB21,
        display_name="Gemini Nano Banana 2.1",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=7.5,
            image_input_per_million=1.5,
            image_output_per_million=30.0,
            verified=date(2026, 10, 9),
        ),
    ),
    ImageModelInfo(
        id="gemini-3-pro-image",
        image=_PRO,
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
    ImageModelInfo(
        id="gemini-3.1-flash-image",
        image=_FLASH,
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
    ImageModelInfo(
        id="gemini-3.1-flash-lite-image",
        image=_LITE,
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
    ImageModelInfo(
        id="gemini-2.5-flash-image",
        image=_V25,
        deprecated=True,
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
