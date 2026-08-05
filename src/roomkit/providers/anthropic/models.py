"""Offline metadata for Anthropic Claude models.

Hand-maintained list returned by ``AnthropicAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what Anthropic currently offers. Call
``AnthropicAIProvider.list_models()`` for that; it asks the account's API.

Sourced from the Anthropic models overview
(platform.claude.com/docs/en/about-claude/models), verified 2026-08-05.

All current Claude models accept image input; context windows are 1M for the
4.6+/Opus 5/Fable/Mythos tier on the Claude API and 200K for the rest. Dated
snapshot ids and their dateless aliases are both listed so either form
resolves here.

Prices are the first-party Claude API rates from Anthropic's pricing page
(platform.claude.com/docs/en/about-claude/pricing), read 2026-08-05.
``cache_write`` is the 5-minute write (1.25x input) because that is the TTL
``AnthropicAIProvider`` asks for — its markers are ``{"type": "ephemeral"}``,
never the 1-hour variant, which costs 2x. Modifiers that are per-request
rather than per-model are absent by construction: the Batch API's 50%, fast
mode's 2x, and the 1.1x for ``inference_geo: "us"``.

One rate here has an expiry: Claude Sonnet 5 is $2/$10 under introductory
pricing through 2026-08-31 and $3/$15 from 2026-09-01. The entry states what
Anthropic charges on ``verified``, not a forecast — which is exactly why the
date travels with the price.

The retired ids keep the price Anthropic still publishes for them (they
remain callable on Bedrock and Google Cloud), so a bill from before the
retirement can still be reconciled against this catalog.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 8, 5)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="claude-opus-5",
        display_name="Claude Opus 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-fable-5",
        display_name="Claude Fable 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=10.0,
            output_per_million=50.0,
            cache_read_per_million=1.0,
            cache_write_per_million=12.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-mythos-5",
        display_name="Claude Mythos 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=10.0,
            output_per_million=50.0,
            cache_read_per_million=1.0,
            cache_write_per_million=12.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-8",
        display_name="Claude Opus 4.8",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=10.0,
            cache_read_per_million=0.2,
            cache_write_per_million=2.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache_read_per_million=0.3,
            cache_write_per_million=3.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=1.0,
            output_per_million=5.0,
            cache_read_per_million=0.1,
            cache_write_per_million=1.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=1.0,
            output_per_million=5.0,
            cache_read_per_million=0.1,
            cache_write_per_million=1.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-7",
        display_name="Claude Opus 4.7",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-6",
        display_name="Claude Opus 4.6",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache_read_per_million=0.3,
            cache_write_per_million=3.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache_read_per_million=0.3,
            cache_write_per_million=3.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-5-20251101",
        display_name="Claude Opus 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-5",
        display_name="Claude Opus 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-1-20250805",
        display_name="Claude Opus 4.1",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=15.0,
            output_per_million=75.0,
            cache_read_per_million=1.5,
            cache_write_per_million=18.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-1",
        display_name="Claude Opus 4.1",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=15.0,
            output_per_million=75.0,
            cache_read_per_million=1.5,
            cache_write_per_million=18.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache_read_per_million=0.3,
            cache_write_per_million=3.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-sonnet-4-0",
        display_name="Claude Sonnet 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache_read_per_million=0.3,
            cache_write_per_million=3.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=15.0,
            output_per_million=75.0,
            cache_read_per_million=1.5,
            cache_write_per_million=18.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-opus-4-0",
        display_name="Claude Opus 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=15.0,
            output_per_million=75.0,
            cache_read_per_million=1.5,
            cache_write_per_million=18.75,
            verified=_VERIFIED,
        ),
    ),
]
