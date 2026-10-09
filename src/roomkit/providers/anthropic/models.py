"""Offline metadata for Anthropic Claude models.

Hand-maintained list returned by ``AnthropicAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what Anthropic currently offers. Call
``AnthropicAIProvider.list_models()`` for that; it asks the account's API.

Sourced from the Anthropic models overview
(platform.claude.com/docs/en/about-claude/models), verified 2026-09-03.

All current Claude models accept image input; context windows are 1M for the
4.6+/Opus 5/Fable/Mythos tier and Haiku 5.5 on the Claude API and 200K for the
rest. Dated
snapshot ids and their dateless aliases are both listed so either form
resolves here.

Prices are the first-party Claude API rates from Anthropic's pricing page
(platform.claude.com/docs/en/about-claude/pricing), read 2026-09-03.
``cache_write`` is the 5-minute write (1.25x input) because that is the TTL
``AnthropicAIProvider`` asks for — its markers are ``{"type": "ephemeral"}``,
never the 1-hour variant, which costs 2x. ``cache_read`` is 0.1x input on
every model but four: Claude Fable 5.1 and Claude Mythos 5.1, where a hit
bills 0.025x ($0.25 per million), and Claude Opus 5.5 and Claude Sonnet 5.5,
where it bills 0.05x ($0.20 and $0.10 per million; pricing page, read
2026-09-22 and 2026-10-09). Modifiers that are per-request rather than
per-model are absent by construction: the Batch API's 50%, fast mode's 2x,
and the 1.1x for ``inference_geo: "us"``. Claude Haiku 5.5 is the one model
priced by prompt length: $0.10 / $0.50 per million up to 100,000 input tokens,
five times that above (pricing page, read 2026-10-07), which its
``long_context_*`` fields carry.

One rate here outlived its expiry: Claude Sonnet 5 launched at $2/$10 as
introductory pricing through 2026-08-31, with $3/$15 scheduled from
2026-09-01; Anthropic made $2/$10 the standard price instead (pricing page,
2026-09-03). The entry states what Anthropic charges on ``verified``, not a
forecast — which is exactly why the date travels with the price.

The retired ids keep the price Anthropic still publishes for them (they
remain callable on Bedrock and Google Cloud), so a bill from before the
retirement can still be reconciled against this catalog.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 9, 3)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="claude-opus-5-5",
        display_name="Claude Opus 5.5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=4.0,
            output_per_million=20.0,
            cache_read_per_million=0.2,
            cache_write_per_million=5.0,
            verified=date(2026, 9, 22),
        ),
    ),
    ModelInfo(
        id="claude-opus-5",
        display_name="Claude Opus 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-fable-5-1",
        display_name="Claude Fable 5.1",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=10.0,
            output_per_million=50.0,
            cache_read_per_million=0.25,
            cache_write_per_million=12.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-fable-5",
        display_name="Claude Fable 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=10.0,
            output_per_million=50.0,
            cache_read_per_million=1.0,
            cache_write_per_million=12.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-mythos-5-1",
        display_name="Claude Mythos 5.1",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=10.0,
            output_per_million=50.0,
            cache_read_per_million=0.25,
            cache_write_per_million=12.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-mythos-5",
        display_name="Claude Mythos 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
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
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    # Claude Sonnet 5.5 added 2026-09-30 at Sonnet 5's rates; its
    # ``deferred_tools`` checked on the wire that day (a ``defer_loading``
    # tool, then called after a ``tool_reference`` result). A cache hit bills
    # 0.05x input, not Sonnet 5's 0.1x (pricing page, 2026-10-09).
    ModelInfo(
        id="claude-sonnet-5-5",
        display_name="Claude Sonnet 5.5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=10.0,
            cache_read_per_million=0.1,
            cache_write_per_million=2.5,
            verified=date(2026, 10, 9),
        ),
    ),
    ModelInfo(
        id="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
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
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache_read_per_million=0.3,
            cache_write_per_million=3.75,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="claude-haiku-5-5",
        display_name="Claude Haiku 5.5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
        # Priced by prompt length: up to 100,000 input tokens at the rates
        # below, five times them above (pricing page, read 2026-10-07).
        pricing=ModelPricing(
            input_per_million=0.10,
            output_per_million=0.50,
            cache_read_per_million=0.01,
            cache_write_per_million=0.125,
            long_context_threshold_tokens=100_000,
            long_context_input_multiplier=5.0,
            long_context_output_multiplier=5.0,
            verified=date(2026, 10, 7),
        ),
    ),
    ModelInfo(
        id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking", "deferred_tools"],
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
        capabilities=["thinking", "deferred_tools"],
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
        capabilities=["thinking", "deferred_tools"],
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
        capabilities=["thinking", "deferred_tools"],
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=25.0,
            cache_read_per_million=0.5,
            cache_write_per_million=6.25,
            verified=_VERIFIED,
        ),
    ),
    # Claude Sonnet 4.5 deprecated 2026-09-30, retiring 2026-11-30 on the
    # Claude API, replaced by Claude Sonnet 5.5 (model deprecations page,
    # 2026-10-09); anthropic 1.11 warns on every request that names it.
    ModelInfo(
        id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        capabilities=["thinking", "deferred_tools"],
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
        deprecated=True,
        capabilities=["thinking", "deferred_tools"],
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
        capabilities=["thinking", "deferred_tools"],
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
        capabilities=["thinking", "deferred_tools"],
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
