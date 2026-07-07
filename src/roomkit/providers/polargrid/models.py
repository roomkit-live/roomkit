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

_REGION_IDS: frozenset[str] = frozenset(r.id for r in REGIONS if r.id)

# Friendly region aliases → canonical edge id. Mirrors the PolarGrid SDK's
# own resolution table (``polargrid.client.REGION_ALIASES``, verified against
# polargrid-sdk 0.8.5 on 2026-07-06) so a region roomkit accepts is one the
# SDK can actually route. The SDK publishes no live region list — hence this
# offline mirror, same rationale as REGIONS above. If the SDK adds an alias,
# add it here too.
REGION_ALIASES: dict[str, str] = {
    "toronto": "yto-01",
    "yto": "yto-01",
    "vancouver": "yvr-02",
    "yvr": "yvr-02",
    "montreal": "yul-01",
    "yul": "yul-01",
    "new-york": "nyc-01",
    "newyork": "nyc-01",
    "nyc": "nyc-01",
    "dallas": "dfw-01",
    "dfw": "dfw-01",
    "san-francisco": "sfo-01",
    "sanfrancisco": "sfo-01",
    "sf": "sfo-01",
    "sfo": "sfo-01",
}


def resolve_region_id(region: str) -> str | None:
    """Resolve a pinned region string (edge id or friendly alias) to a canonical
    edge id, or ``None`` if it is neither.

    Case-insensitive, mirroring the SDK. Lets callers reject a typo like
    ``"yul-2"`` up front instead of letting the SDK build an unroutable host
    (``https://api.yul-2.edge.polargrid.ai``) that fails later with an opaque
    DNS error.
    """
    normalized = region.lower()
    resolved = REGION_ALIASES.get(normalized, normalized)
    return resolved if resolved in _REGION_IDS else None


def region_choices() -> str:
    """Human-readable list of accepted region ids and aliases, for error text."""
    ids = ", ".join(r.id for r in REGIONS if r.id)
    aliases = ", ".join(sorted(REGION_ALIASES))
    return f"ids ({ids}) or aliases ({aliases})"


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
