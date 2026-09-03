"""Curated catalog of PolarGrid chat models.

Hand-maintained, offline snapshot returned by
``PolarGridAIProvider.available_models``. Sources, all read 2026-09-02:
PolarGrid's model pages (https://polargrid.mintlify.app/models), its
model-availability guide (``/guides/model-availability``) and its regions
guide (``/guides/regions``), cross-checked live against the autorouter, which
answers ``/v1/route?model=<id>`` with 404 when no edge serves the id, and a
sweep of every edge's ``/health``. Only the chat / LLM models usable through
this provider's ``generate()`` are listed; the live
``PolarGridAIProvider.list_models()`` queries the connected edge and also
surfaces the STT / TTS models (``whisper-large-v3-turbo``,
``cohere-transcribe-03-2026``, ``kokoro-82m``, ``tada-3b-ml``).

The public LLM lineup is one model:

- ``qwen-3.8-27b`` — fleet-wide: every public edge in the vendor's matrix,
  and the id the autorouter routes.
- ``qwen-3.5-27b`` — **retired 2026-08-20**. No edge serves it (the vendor's
  page says so; the autorouter answers 404 for it) and a request fails with
  ``404 model_not_loaded``. Removed rather than flagged, as the OpenAI
  catalog does with an id past its shutdown: a dead id here only invites
  the 404.
- ``qwen-3.6-35b-a3b`` — **customer pilot**, served from no public edge (its
  Montreal pilot node, yul-02, left the public fleet; the autorouter answers
  404 for it). Kept in :data:`PILOT_MODELS` rather than :data:`MODELS`: a
  pilot customer's ``supports_vision`` still resolves through
  :data:`MODELS_BY_ID`, and nobody else is offered a model that 404s.

Context windows come from the model pages (256K for 3.8; 8192 as *served*
for 3.6, capped below its native window to bound KV-cache VRAM), the 3.8
list price from the models page.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from roomkit.providers.ai.base import ModelInfo, ModelPricing


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
# (re-read from polargrid-sdk 0.10.0 on 2026-09-02 — 16 edges and 15 aliases,
# matching this table exactly both ways), which is what actually routes
# a request — the shipped table is the thing that decides. Mirrored rather
# than imported because this module must load without the optional SDK
# installed. An edge missing here is refused by ``resolve_region_id`` even
# though the SDK could route it, so the floor in pyproject.toml is what keeps
# the mirror truthful: chi/lax/phx/sea/was landed in 0.9.0, mia-01 in 0.9.1,
# sfo-03 in 0.9.2. Bump both together.
#
# The vendor's regions guide publishes 15 of these: ``yul-02`` (Montreal 02)
# left the public fleet with the qwen-3.6 customer pilot and is no longer
# listed, but the SDK still routes it and the edge still answers ``/health``
# (sweep of 2026-09-02), so a pilot customer pinning it is not refused. That
# sweep found every other edge healthy — nyc-01 and dfw-01 back in DNS after
# the 2026-08-19 gap — except yto-01, which resolved but accepted no TCP
# connection at the time: an outage, not a catalog fact.
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
# polargrid-sdk 0.10.0, re-verified 2026-09-02) so a region roomkit accepts is
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
# chat endpoint accepts OpenAI-shaped image_url content), but vision is the
# deployed model's capability, not the SDK's. Only qwen-3.6-35b-a3b actually
# reads the image — verified live on yul-02 on 2026-07-09, while that edge was
# public. qwen-3.8-27b is refused server-side ("model 'qwen-3.8-27b' does not
# support image input", verified live 2026-08-19): a clean error, text-only all
# the same.
_VERIFIED = date(2026, 9, 2)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="qwen-3.8-27b",
        display_name="Qwen 3.8 27B",
        context_window=262_144,
        supports_vision=False,
        # completion + tools + enable_thinking validated live on yto-01
        # (2026-08-19); image input refused server-side.
        capabilities=["completion", "tools", "thinking"],
        # The vendor's list price (models page, read 2026-09-02). It publishes
        # no cache or long-context tier. A private edge bills what its
        # contract says, which is not this catalog's business.
        pricing=ModelPricing(input_per_million=0.20, output_per_million=0.75, verified=_VERIFIED),
    ),
]
"""Chat models served on PolarGrid's public edges: what ``available_models``
advertises, and what the priced-catalog guard covers."""

PILOT_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="qwen-3.6-35b-a3b",
        display_name="Qwen 3.6 35B-A3B",
        context_window=8192,
        supports_vision=True,
        # enable_thinking + vision validated end-to-end on yul-02 (2026-07-09).
        capabilities=["completion", "tools", "thinking", "vision"],
    ),
]
"""Chat models in limited availability (customer pilot), served from no public
edge and quoted at no list price. Not advertised, still recognised: a pilot
customer's ``supports_vision`` and ``list_models`` backfill resolve here."""

# Model id → catalog entry, for the model-driven ``supports_vision`` lookup and
# the ``list_models`` backfill — public and pilot alike, because both are
# models the configured edge may serve.
MODELS_BY_ID: dict[str, ModelInfo] = {m.id: m for m in (*MODELS, *PILOT_MODELS)}
