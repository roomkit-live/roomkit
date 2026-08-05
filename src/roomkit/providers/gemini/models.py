"""Offline metadata for Google Gemini models.

Hand-maintained list returned by ``GeminiAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what the Gemini API currently offers. Call
``GeminiAIProvider.list_models()`` for that; it reports ``inputTokenLimit``
per model straight from the API.

Sourced from the Gemini API models docs (ai.google.dev/gemini-api/docs/models),
verified 2026-08-05.

Ids carry no ``models/`` prefix, matching the form ``GeminiConfig.model`` and the
generate-content calls use. Current text/multimodal models all report a
1,048,576-token input window and accept image input.

Scope is the text/multimodal generate-content models. The image-generation
(``*-image``, Nano Banana), TTS, Live, embedding, Veo, Lyria, robotics and
Deep Research variants are out of scope for this provider. Google lists
``gemini-3-pro-preview`` and ``gemini-3.1-flash-lite-preview`` as shut down;
they are absent rather than flagged, since a retired id is a 404.

Prices are the paid-tier rates from Google's pricing page
(ai.google.dev/gemini-api/docs/pricing), read 2026-08-05. Two things they
flatten. Google charges more per token for prompts over 200k on the Pro
tiers (``gemini-3.1-pro-preview``, ``gemini-2.5-pro``: 2x input, 1.5x
output) and more for audio input on the Flash tiers; the entries carry the
text rate at the standard prompt size, which is what a text conversation
pays. And context caching is billed by *storage time* ($1.00–$4.50 per
million tokens per hour), not per token written — so ``cache_write`` stays
unset here rather than restating an hourly rate as a per-token one.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_CTX = 1_048_576
_VERIFIED = date(2026, 8, 5)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=7.5,
            cache_read_per_million=0.15,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=1.5,
            output_per_million=9.0,
            cache_read_per_million=0.15,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash-Lite",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=0.3,
            output_per_million=2.5,
            cache_read_per_million=0.03,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro (Preview)",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=12.0,
            cache_read_per_million=0.2,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=0.25,
            output_per_million=1.5,
            cache_read_per_million=0.025,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=10.0,
            cache_read_per_million=0.125,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=0.3,
            output_per_million=2.5,
            cache_read_per_million=0.03,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=0.1,
            output_per_million=0.4,
            cache_read_per_million=0.01,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash (Preview)",
        context_window=_CTX,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=0.5,
            output_per_million=3.0,
            cache_read_per_million=0.05,
            verified=_VERIFIED,
        ),
    ),
]
