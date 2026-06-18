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
    """A PolarGrid edge.

    Returned both by the curated catalog (:func:`available_regions`) and
    by the live :meth:`~roomkit.providers.polargrid.PolarGridAIProvider.connected_region`.
    PolarGrid exposes no live list of all regions over the edge API, so the
    connected-edge query reports only the routed edge.

    Attributes:
        id: Edge id (e.g. ``"yul-02"``, ``"yvr-02"``).
        name: Human-readable edge name (e.g. ``"Montreal 02"``).
        location: Geographic placement (e.g. ``"Canada East"``, ``"US West"``)
            — Canadian edges (``location`` starts with ``"Canada"``) are the
            ones that keep data on Canadian soil.
    """

    id: str | None = None
    name: str | None = None
    location: str | None = None


# Authoritative edge list from PolarGrid's regions guide
# (https://polargrid.mintlify.app/guides/regions, verified 2026-06-11). The
# Canada/US split is the data-residency signal (Law 25 / PIPEDA).
REGIONS: list[PolarGridRegion] = [
    PolarGridRegion(id="yto-01", name="Toronto", location="Canada Central"),
    PolarGridRegion(id="yul-01", name="Montreal", location="Canada East"),
    PolarGridRegion(id="yul-02", name="Montreal 02", location="Canada East"),
    PolarGridRegion(id="yvr-02", name="Vancouver", location="Canada West"),
    PolarGridRegion(id="nyc-01", name="New York", location="US East"),
    PolarGridRegion(id="nyc-02", name="New York 02", location="US East"),
    PolarGridRegion(id="dfw-01", name="Dallas", location="US Central"),
    PolarGridRegion(id="dfw-02", name="Dallas 02", location="US Central"),
    PolarGridRegion(id="sfo-01", name="San Francisco", location="US West"),
]


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
