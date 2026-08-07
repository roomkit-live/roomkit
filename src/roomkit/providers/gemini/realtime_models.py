"""Offline catalog of Google Gemini Live speech-to-speech models.

Hand-maintained list returned by ``GeminiLiveProvider.available_models`` — a
counterpart to ``gemini/models.py`` kept apart from it for the reason the
image catalog gives (RFC §25.6): the sets are disjoint. No id here answers a
generate-content call, and no chat id opens a Live session.

Sourced from Google's Live API docs (ai.google.dev), verified 2026-08-07. No
public aggregator mirrors the Live lineup, so ``scripts/check_models.py``
names this catalog in ``UNMIRRORED_CATALOGS`` rather than comparing it
against a slice that cannot contain it.

Context windows are omitted deliberately: the realtime channel never trims
history against a window, and an unknown number is safer than one nobody
reconciles. Pricing is omitted for a unit reason: Live sessions bill audio
tokens at rates :class:`~roomkit.providers.ai.base.ModelPricing` does not
model, and restating text rates alone would price the wrong unit.

Every Live model accepts image frames (``inject_image`` rides the same
``send_realtime_input`` the video path uses), hence ``supports_vision=True``
throughout — documented behaviour, not a guess.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3.1-flash-live-preview",
        display_name="Gemini 3.1 Flash Live (preview)",
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-2.5-flash-native-audio-preview-12-2025",
        display_name="Gemini 2.5 Flash Native Audio (preview)",
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-2.0-flash-live-001",
        display_name="Gemini 2.0 Flash Live",
        supports_vision=True,
    ),
]
