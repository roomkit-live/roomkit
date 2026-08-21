"""Offline metadata for xAI (Grok Imagine) image-generation models.

Hand-maintained list returned by ``XAIImageProvider.available_models`` — a
counterpart to ``xai/models.py``, kept apart from it because the two are
disjoint sets: no id here converses, and no id there draws (RFC §25.6).

Sourced from xAI's models and image-capability docs (docs.x.ai), verified
2026-08-21. ``grok-2-image`` is gone from the vendor's model page and is not
carried here.

No entry carries a ``pricing``: xAI bills this lineup a flat amount per
generated image (varying by model, and for ``grok-imagine-image-2.0`` by
quality and resolution), plus a flat amount per reference image on an edit.
:class:`~roomkit.providers.ai.base.ModelPricing` states per-token rates, and a
per-image charge restated per token would be a wrong number rather than a
missing one — the same reason the OpenAI image catalog omits the DALL·E
models. Read the current per-image amounts from xAI's own pricing page.

``edit`` is tagged where reference images are documented to be accepted:
xAI's editing docs name ``grok-imagine-image-2.0``, and its quality-mode
sibling accepts references through the same endpoint. The base
``grok-imagine-image`` documents no editing support and carries no tag.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.image.base import IMAGE_GEN_CAPABILITY

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="grok-imagine-image-2.0",
        display_name="Grok Imagine Image 2.0",
        supports_vision=True,
        capabilities=[IMAGE_GEN_CAPABILITY, "edit"],
    ),
    ModelInfo(
        id="grok-imagine-image-quality",
        display_name="Grok Imagine Image (Quality)",
        supports_vision=True,
        capabilities=[IMAGE_GEN_CAPABILITY, "edit"],
    ),
    ModelInfo(
        id="grok-imagine-image",
        display_name="Grok Imagine Image",
        supports_vision=False,
        capabilities=[IMAGE_GEN_CAPABILITY],
    ),
]
