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
edge. Live sweep of the LLM-serving edges on 2026-08-19, mid-rollout of
qwen 3.8:

- ``qwen-3.8-27b`` — yto-01, yvr-02, nyc-02, sfo-01, dfw-02 (it *replaced*
  ``qwen-3.5-27b`` on each; the two are never co-loaded)
- ``qwen-3.5-27b`` — **yul-01 only** now; nyc-01 and dfw-01, which also
  carried it, no longer resolved in DNS at the sweep
- ``qwen-3.6-35b-a3b`` — **yul-02 only** (unreachable at the 2026-08-19
  sweep; entry kept from the 2026-07-09 live verification)

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


# Offline mirror of the SDK's own ``polargrid.client.POLARGRID_REGIONS``
# (re-read from polargrid-sdk 0.10.0 on 2026-08-13 — 16 edges and 15 aliases,
# matching this table exactly both ways), which is what actually routes
# a request — the regions guide agrees, but the shipped table is the thing that
# decides. Mirrored rather than imported because this module must load without
# the optional SDK installed. An edge missing here is refused by
# ``resolve_region_id`` even though the SDK could route it, so the floor in
# pyproject.toml is what keeps the mirror truthful: chi/lax/phx/sea/was landed
# in 0.9.0, mia-01 in 0.9.1, sfo-03 in 0.9.2. Bump both together.
#
# The Canada/US split is the data-residency signal (Law 25 / PIPEDA).
REGIONS: list[PolarGridRegion] = [
    PolarGridRegion(id="yto-01", name="Toronto", location="Canada Central"),
    PolarGridRegion(id="yul-01", name="Montreal", location="Canada East"),
    PolarGridRegion(id="yul-02", name="Montreal 02", location="Canada East"),
    PolarGridRegion(id="yvr-02", name="Vancouver", location="Canada West"),
    PolarGridRegion(id="nyc-01", name="New York", location="US East"),
    PolarGridRegion(id="nyc-02", name="New York 02", location="US East"),
    PolarGridRegion(id="was-01", name="Washington DC", location="US East"),
    PolarGridRegion(id="mia-01", name="Miami", location="US East"),
    PolarGridRegion(id="dfw-01", name="Dallas", location="US Central"),
    PolarGridRegion(id="dfw-02", name="Dallas 02", location="US Central"),
    PolarGridRegion(id="chi-01", name="Chicago", location="US Central"),
    PolarGridRegion(id="sfo-01", name="San Francisco", location="US West"),
    PolarGridRegion(id="sfo-03", name="San Francisco 03", location="US West"),
    PolarGridRegion(id="lax-01", name="Los Angeles", location="US West"),
    PolarGridRegion(id="sea-01", name="Seattle", location="US West"),
    PolarGridRegion(id="phx-01", name="Phoenix", location="US West"),
]

_REGION_IDS: frozenset[str] = frozenset(r.id for r in REGIONS if r.id)

# Friendly region aliases → canonical edge id. Mirrors the PolarGrid SDK's
# own resolution table (``polargrid.client.REGION_ALIASES``, unchanged in
# polargrid-sdk 0.10.0, re-verified 2026-08-19) so a region roomkit accepts is
# one the SDK can actually route. Deliberately not extended past what the SDK
# carries: the newer edges have no alias upstream, and inventing one here would
# make this a table roomkit maintains rather than a mirror it tracks. Their
# canonical ids work. If the SDK adds an alias, add it here too.
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


# Vision: PolarGrid rolled out multimodal chat with polargrid-sdk 0.9.0 (the
# chat endpoint now accepts OpenAI-shaped image_url content), but vision is the
# deployed model's capability, not the SDK's. Only qwen-3.6-35b-a3b (yul-02)
# actually reads the image — verified live 2026-07-09; qwen-3.5-27b accepts the
# request but answers as if no image was sent, so it stays text-only.
# qwen-3.8-27b is refused server-side ("model 'qwen-3.8-27b' does not support
# image input", verified live 2026-08-19) — a clean error rather than 3.5's
# silent ignore, but text-only all the same.
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
        supports_vision=True,
        # enable_thinking + vision validated end-to-end on yul-02.
        capabilities=["completion", "tools", "thinking", "vision"],
    ),
    ModelInfo(
        id="qwen-3.8-27b",
        display_name="Qwen 3.8 27B",
        supports_vision=False,
        # completion + tools + enable_thinking validated live on yto-01
        # (2026-08-19); image input refused server-side.
        capabilities=["completion", "tools", "thinking"],
    ),
]

# Model id → catalog entry, for the model-driven ``supports_vision`` lookup.
MODELS_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}
