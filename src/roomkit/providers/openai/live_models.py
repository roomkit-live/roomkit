"""Offline catalog of OpenAI GPT-Live speech-to-speech models.

Hand-maintained list returned by ``OpenAILiveProvider.available_models``. Kept
apart from ``realtime_models.py`` for the same reason that one is kept apart
from the chat catalog (RFC §25.6): the sets are disjoint. A ``gpt-live-*`` id
opens a Live API session (``/v1/live/sessions``) and nothing else — it answers
no Realtime WebSocket, no chat completion.

Sourced from OpenAI's Live API model docs (developers.openai.com), verified
2026-09-11. No public aggregator mirrors the lineup, so
``scripts/check_models.py`` names this catalog in ``UNMIRRORED_CATALOGS``.

Context windows and pricing are omitted for the reasons the Realtime catalog
gives, with one more: the live model bills *session seconds*, a unit
:class:`~roomkit.providers.ai.base.ModelPricing` does not model, and the
backend it delegates to is billed separately under its own catalog entry.

``supports_vision`` is ``False`` throughout: the Live endpoint takes audio and
text in, audio and text out, and no image.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-live-1",
        display_name="GPT-Live 1",
        supports_vision=False,
        capabilities=["full_duplex", "reasoning_delegation"],
    ),
]
