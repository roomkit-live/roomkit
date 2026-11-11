"""Offline metadata for Google Gemini models.

Hand-maintained list returned by ``GeminiAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what the Gemini API currently offers. Call
``GeminiAIProvider.list_models()`` for that; it reports ``inputTokenLimit``
per model straight from the API.

Sourced from the Gemini API models docs (ai.google.dev/gemini-api/docs/models),
verified 2026-09-03.

Ids carry no ``models/`` prefix, matching the form ``GeminiConfig.model`` and the
generate-content calls use. Current text/multimodal models all report a
1,048,576-token input window and accept image input.

Scope is the text/multimodal generate-content models. The image-generation
(``*-image``, Nano Banana), TTS, Live, embedding, Veo, Lyria, robotics and
Deep Research variants are out of scope for this provider. Google lists
``gemini-3-pro-preview`` and ``gemini-3.1-flash-lite-preview`` as shut down;
they are absent rather than flagged, since a retired id is a 404.

Prices are the paid-tier standard rates from Google's pricing page
(ai.google.dev/gemini-api/docs/pricing), read 2026-09-03. The Pro entries
represent Google's higher rates beyond 200k input tokens (2x input, 1.5x
output). Audio input on Flash has a separate rate that this text-token catalog
does not represent. Batch, Flex and Priority tiers are not represented either:
roomkit calls the synchronous API, so the standard column is the one that bills.

Three entries carry a launch discount rather than a settled rate. Google prices
``gemini-3.8-flash``, ``gemini-3.7-flash`` and ``gemini-3.6-flash`` at half the
published figure "through December 31, 2026", doubling on January 1, 2027 to
$1.50/$7.50/$0.15. What a call costs today is what belongs here, so the
discounted rates are the ones stated — and those three need a re-read in
January, when the increase makes them silently half of what a consumer is
billed.

``gemini-4-argon`` (read 2026-10-13) and ``gemini-4-carbon`` (read 2026-11-11)
have no long-context tier and a scheduled increase instead of a discount:
$2.00/$10.00/$0.10 through November 30, 2026, then $4.00/$20.00/$0.20 from
December 1, 2026. The current rates are stated, and both entries need a re-read
in December. Their caching is implicit only. Google's thinking page lists both
at ``low``/``medium``/``high``; Argon's refusal of ``minimal`` and its taking a
budget of 0 were measured, Carbon's are assumed from Argon's (its account had
no credit left to probe on 2026-11-11).

Context caching is billed by *storage time* ($1.00–$4.50 per million tokens per
hour), not per token written — so ``cache_write`` stays unset here rather than
restating an hourly rate as a per-token one.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_CTX = 1_048_576
_VERIFIED = date(2026, 9, 3)

# What a model takes of thinking (RFC §6.7), measured on 2026-09-30: every
# Gemini 3 model takes the levels ``low``, ``medium`` and ``high``, and some
# ``minimal`` too; a Gemini 2.5 model refuses a level and takes a token budget
# only. Two cannot stop reasoning: a budget of 0 answers 400.
TAKES_LEVELS = "thinking_level"
TAKES_MINIMAL = "thinking_level_minimal"
CANNOT_STOP = "thinking_required"
_LEVELS = [TAKES_LEVELS]
_LEVELS_MINIMAL = [TAKES_LEVELS, TAKES_MINIMAL]

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-4-carbon",
        display_name="Gemini 4 Carbon",
        context_window=_CTX,
        supports_vision=True,
        # As Argon: no ``minimal`` on Google's thinking page (2026-11-11).
        capabilities=["thinking", "audio", "video", *_LEVELS],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=10.0,
            cache_read_per_million=0.1,
            verified=date(2026, 11, 11),
        ),
    ),
    ModelInfo(
        id="gemini-4-argon",
        display_name="Gemini 4 Argon",
        context_window=_CTX,
        supports_vision=True,
        # Refuses ``minimal`` with a 400 and takes a budget of 0 (2026-10-13).
        capabilities=["thinking", "audio", "video", *_LEVELS],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=10.0,
            cache_read_per_million=0.1,
            verified=date(2026, 10, 13),
        ),
    ),
    ModelInfo(
        id="gemini-3.8-flash",
        display_name="Gemini 3.8 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video", *_LEVELS],
        pricing=ModelPricing(
            input_per_million=0.75,
            output_per_million=3.75,
            cache_read_per_million=0.075,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.7-flash",
        display_name="Gemini 3.7 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video", *_LEVELS],
        pricing=ModelPricing(
            input_per_million=0.75,
            output_per_million=3.75,
            cache_read_per_million=0.075,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video", *_LEVELS_MINIMAL],
        pricing=ModelPricing(
            input_per_million=0.75,
            output_per_million=3.75,
            cache_read_per_million=0.075,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video", *_LEVELS_MINIMAL],
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
        capabilities=["thinking", "audio", "video", *_LEVELS_MINIMAL, CANNOT_STOP],
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
        capabilities=["thinking", "audio", "video", *_LEVELS, CANNOT_STOP],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=12.0,
            cache_read_per_million=0.2,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video", *_LEVELS_MINIMAL],
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
        # 404 "no longer available to new users" (2026-09-30).
        deprecated=True,
        capabilities=["thinking", "audio", "video"],
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=10.0,
            cache_read_per_million=0.125,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
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
        # 404 "no longer available to new users" (2026-09-30).
        deprecated=True,
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
        capabilities=["thinking", *_LEVELS_MINIMAL],
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=0.5,
            output_per_million=3.0,
            cache_read_per_million=0.05,
            verified=_VERIFIED,
        ),
    ),
]
