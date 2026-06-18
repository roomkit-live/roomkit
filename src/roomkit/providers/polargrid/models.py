"""Curated catalog of PolarGrid chat models.

Hand-maintained, offline snapshot returned by
``PolarGridAIProvider.available_models``. Sourced from PolarGrid's model
availability guide (https://polargrid.mintlify.app/guides/model-availability,
verified 2026-06-11). Only the chat / LLM models usable through this
provider's ``generate()`` are listed here; the live
``PolarGridAIProvider.list_models()`` queries the connected edge and also
surfaces the STT / TTS models (``whisper-large-v3-turbo``,
``cohere-transcribe-03-2026``, ``kokoro-82m``, ``tada-3b-ml``).

Availability is **regional** — the catalog ids are not loaded on every
edge:

- ``qwen-3.5-27b`` — yto-01, yul-01, yvr-02, nyc-01/02, sfo-01, dfw-01/02
- ``qwen-3.6-35b-a3b`` — **yul-02 only** (Montreal serves it in place of
  the standard ``qwen-3.5-27b``)

PolarGrid's guide does not publish context windows, so they are left
unset (``None`` = unknown) rather than guessed.
"""

from __future__ import annotations

from pydantic import BaseModel

from roomkit.providers.ai.base import ModelInfo


class PolarGridRegion(BaseModel):
    """The PolarGrid edge a provider is routed to.

    PolarGrid exposes no live list of all regions over the edge API, so
    this describes only the *connected* edge. Both fields come straight
    from the SDK client and may be ``None`` before the client is built.

    Attributes:
        id: Edge id (e.g. ``"yul-02"``, ``"yvr-01"``).
        name: Human-readable edge name (e.g. ``"Montreal 02"``).
    """

    id: str | None = None
    name: str | None = None


MODELS: list[ModelInfo] = [
    ModelInfo(
        id="qwen-3.5-27b",
        display_name="Qwen 3.5 27B",
        supports_vision=False,
        capabilities=["completion", "tools"],
    ),
    ModelInfo(
        id="qwen-3.6-35b-a3b",
        display_name="Qwen 3.6 35B-A3B",
        supports_vision=False,
        # enable_thinking validated end-to-end on yul-02.
        capabilities=["completion", "tools", "thinking"],
    ),
]
