"""Offline catalog of xAI Grok realtime speech-to-speech models.

Hand-maintained list returned by ``XAIRealtimeProvider.available_models`` — a
counterpart to ``xai/models.py``, whose scope note already sends realtime ids
here. The sets are disjoint (RFC §25.6): ``grok-2-audio`` answers no chat
completion, and no chat Grok opens a Realtime WebSocket.

Sourced from the xAI audio API capability docs, verified 2026-08-07. The
public aggregator mirrors xAI's chat models only, so
``scripts/check_models.py`` names this catalog in ``UNMIRRORED_CATALOGS``.

Context window and pricing are omitted for the reasons the OpenAI realtime
catalog gives: the realtime channel never trims by window, and audio-token
rates are a unit :class:`~roomkit.providers.ai.base.ModelPricing` does not
model. ``supports_vision=False`` restates a deliberate 0.43.0 decision: the
xAI provider does not inherit the OpenAI image path, its models not having
been verified to accept one.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="grok-2-audio",
        display_name="Grok 2 Audio",
        supports_vision=False,
    ),
]
