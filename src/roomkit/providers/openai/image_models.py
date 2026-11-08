"""Offline metadata for OpenAI image-generation models.

Hand-maintained list returned by ``OpenAIImageProvider.available_models`` — a
counterpart to ``openai/models.py``, kept apart from it because the two are
disjoint sets: no id here converses, and no id there draws (RFC §25.6).

Sourced from the OpenAI images guide and the ``ImageModel`` literal shipped by
the ``openai`` SDK (2.48.0), verified 2026-08-07. The GPT Image 2.5 pair is the
exception: announced 2026-09-08, both carry their own model page under
developers.openai.com/api/docs/models but neither had reached the SDK's
``ImageModel`` literal by 2.54.0, so the ids come from those pages. So do the
GPT Image 2.6 pair's, read 2026-11-08; their SVG and GIF outputs and their
``complexity`` request field are not offered here yet.

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
from roomkit.providers.image.options import ImageCapabilities, ImageModelInfo

_VERIFIED = date(2026, 8, 7)
_CAPS = [IMAGE_GEN_CAPABILITY, "edit"]

_STANDARD = ImageCapabilities(
    options=[
        "quality",
        "background",
        "output_format",
        "output_compression",
        "moderation",
        "input_fidelity",
        "partial_images",
    ],
    qualities=["auto", "low", "medium", "high"],
    formats=["png", "jpeg", "webp"],
    backgrounds=["auto", "opaque", "transparent"],
    sizes=["auto", "1024x1024", "1536x1024", "1024x1536"],
    max_references=16,
    mask=True,
    streaming=True,
    verified=date(2026, 9, 15),
)
_FLEXIBLE = _STANDARD.model_copy(
    update={
        "flexible_size": True,
        "sizes": ["auto"],
        "options": [option for option in _STANDARD.options if option != "input_fidelity"],
    }
)
_V25 = _FLEXIBLE.model_copy(
    update={"qualities": ["auto", "low", "medium", "high", "xhigh", "max"]}
)

MODELS: list[ModelInfo] = [
    ImageModelInfo(
        id="gpt-image-2.6-sunburst",
        image=_V25,
        display_name="GPT Image 2.6 Sunburst",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=0.0,
            cache_read_per_million=1.25,
            image_input_per_million=8.0,
            image_output_per_million=30.0,
            verified=date(2026, 11, 8),
        ),
    ),
    ImageModelInfo(
        id="gpt-image-2.6-flare",
        image=_V25,
        display_name="GPT Image 2.6 Flare",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=0.0,
            cache_read_per_million=1.25,
            image_input_per_million=8.0,
            image_output_per_million=30.0,
            verified=date(2026, 11, 8),
        ),
    ),
    ImageModelInfo(
        id="gpt-image-2.5-sunburst",
        image=_V25,
        aliases=["gpt-image-2.5-sunburst-2026-09-08"],
        display_name="GPT Image 2.5 Sunburst",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=0.0,
            cache_read_per_million=1.25,
            image_input_per_million=8.0,
            image_output_per_million=30.0,
            verified=date(2026, 9, 9),
        ),
    ),
    ImageModelInfo(
        id="gpt-image-2.5-flare",
        image=_V25,
        aliases=["gpt-image-2.5-flare-2026-09-08"],
        display_name="GPT Image 2.5 Flare",
        supports_vision=True,
        capabilities=_CAPS,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=0.0,
            cache_read_per_million=1.25,
            image_input_per_million=8.0,
            image_output_per_million=30.0,
            verified=date(2026, 9, 9),
        ),
    ),
    ImageModelInfo(
        id="gpt-image-2",
        image=_FLEXIBLE,
        aliases=["gpt-image-2-2026-04-21"],
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
    ImageModelInfo(
        id="gpt-image-1.5",
        image=_STANDARD,
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
    ImageModelInfo(
        id="chatgpt-image-latest",
        image=_STANDARD,
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
    ImageModelInfo(
        id="gpt-image-1",
        image=_STANDARD,
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
    ImageModelInfo(
        id="gpt-image-1-mini",
        image=_STANDARD.model_copy(
            update={"options": [o for o in _STANDARD.options if o != "input_fidelity"]}
        ),
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
