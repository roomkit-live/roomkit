"""Offline catalog of OpenAI Realtime speech-to-speech models.

Hand-maintained list returned by ``OpenAIRealtimeProvider.available_models`` —
a counterpart to ``openai/models.py`` kept apart from it for the reason the
image catalog gives (RFC §25.6): the sets are disjoint. No id here answers a
Chat Completions call, and no chat id opens a Realtime WebSocket, so merging
them would only oblige every consumer of the conversational catalog to filter
out a class of models it can never use.

Sourced from OpenAI's Realtime API model docs (platform.openai.com), verified
2026-08-07. No public aggregator mirrors this lineup — the ids live on the
Realtime WebSocket endpoint, and the mirror's ``gpt-audio``/``gpt-audio-mini``
are the chat-completions audio models, not these — so
``scripts/check_models.py`` names this catalog in ``UNMIRRORED_CATALOGS``
rather than comparing it against a slice that cannot contain it.

Context windows are omitted deliberately: the realtime channel never trims
history against a window (the provider holds the conversation server-side),
and an unknown number is safer than one nobody reconciles. Pricing is omitted
for a unit reason, not a laziness one: Realtime models bill *audio* tokens at
rates :class:`~roomkit.providers.ai.base.ModelPricing` does not model, and
restating the text rates alone would price the wrong unit.

``capabilities`` states reasoning support positively, as the chat catalog
does: ``"thinking"`` marks the ``gpt-realtime-2``\\ + models whose sessions
accept ``reasoning_effort``. ``supports_vision`` follows the provider's
documented cut: ``gpt-realtime-2.1`` and later read images via
``inject_image``; earlier models do not.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-realtime-2.1",
        display_name="GPT Realtime 2.1",
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="gpt-realtime-2.1-mini",
        display_name="GPT Realtime 2.1 Mini",
        supports_vision=True,
        capabilities=["thinking"],
    ),
    ModelInfo(
        id="gpt-realtime-1.5",
        display_name="GPT Realtime 1.5",
        supports_vision=False,
    ),
    ModelInfo(
        id="gpt-4o-realtime-preview",
        display_name="GPT-4o Realtime (preview)",
        supports_vision=False,
        deprecated=True,
    ),
    ModelInfo(
        id="gpt-4o-mini-realtime-preview",
        display_name="GPT-4o Mini Realtime (preview)",
        supports_vision=False,
        deprecated=True,
    ),
]
