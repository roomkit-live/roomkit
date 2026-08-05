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
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="claude-opus-5",
        display_name="Claude Opus 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-fable-5",
        display_name="Claude Fable 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-mythos-5",
        display_name="Claude Mythos 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-opus-4-8",
        display_name="Claude Opus 4.8",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-opus-4-7",
        display_name="Claude Opus 4.7",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-opus-4-6",
        display_name="Claude Opus 4.6",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-opus-4-5-20251101",
        display_name="Claude Opus 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-opus-4-5",
        display_name="Claude Opus 4.5",
        context_window=200_000,
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="claude-opus-4-1-20250805",
        display_name="Claude Opus 4.1",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="claude-opus-4-1",
        display_name="Claude Opus 4.1",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="claude-sonnet-4-0",
        display_name="Claude Sonnet 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="claude-opus-4-0",
        display_name="Claude Opus 4",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
]
