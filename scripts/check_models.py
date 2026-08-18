#!/usr/bin/env python3
"""Check the offline model catalogs against a live upstream source.

The catalogs in ``providers/*/models.py`` and ``providers/*/image_models.py``
exist so a provider can answer offline — a context window, an image model's
rates. That makes them useful and stale-able at the same time: nothing in a
test suite notices when a vendor ships a new flagship or changes a window,
because the catalog is self-consistent either way. This script is what notices,
so a release does not quietly ship last quarter's lineup.

The reference is OpenRouter's ``/api/v1/models``, which is public, needs no key,
and republishes ids, context windows, modalities, and rates for every major
vendor — one request covers Anthropic, OpenAI, Google, Mistral, and xAI. It is a
mirror, not the vendor, so treat a finding as "go read the vendor's docs", never
as a value to paste in blind.

A catalog is compared against the upstream slice matching its output modality
(:func:`belongs_to`), so the chat and image lineups a vendor publishes under one
namespace never contaminate each other.

What has no mirror is named rather than skipped: Ollama (a local pull registry)
and PolarGrid (private edges) carry their own verification dates, and
``UNMIRRORED_CATALOGS`` lists the rest with the reason each was not found. A
catalog nobody checks is the one that goes stale, so silence about it would read
as coverage.

Reported:

  DRIFT    a catalog id whose context window disagrees with upstream — an
           actively wrong number, the one failure that silently mistrims
           conversation history.
  PRICE    a rate the catalog states and upstream quotes differently — the
           failure that bills the wrong amount. A non-zero upstream rate that
           the catalog leaves unset is also reported, unless the units are
           explicitly known to be incomparable.
  MISSING  an upstream model newer than everything the catalog knows in that
           same family — i.e. the vendor moved on.
  GONE     a non-deprecated catalog id upstream no longer lists — a candidate
           retirement, reported as a warning because a mirror lagging is at
           least as likely as a real removal.

Whether every model *has* a price is not checked here: that one needs neither
network nor release, so it lives in the test suite
(``tests/test_providers/test_ai_models.py``) and fails on the commit that adds
an unpriced model.

Exit codes: 0 clean, 1 findings, 2 upstream unreachable (network, not drift —
callers should warn and continue rather than block on someone else's outage).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from roomkit.providers.ai.base import ModelInfo

UPSTREAM_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class Catalog:
    """One curated catalog and the upstream slice it is compared against.

    Attributes:
        label: Name used in findings.
        module: Import path exposing a ``MODELS`` list.
        prefix: Upstream vendor namespace this catalog's ids live under.
        prefixed: Whether the catalog's own ids already carry that namespace
            (OpenRouter's do), so nothing is stripped.
        track_new: Whether to report upstream models newer than everything the
            catalog knows in the same family.
        modality: Upstream output modality this catalog covers. Keeps the
            text and image slices from contaminating each other — a vendor
            publishes both under one namespace, and comparing a catalog
            against the wrong half reports every id as retired.
    """

    label: str
    module: str
    prefix: str
    prefixed: bool = False
    track_new: bool = True
    modality: str = "text"


CATALOGS: list[Catalog] = [
    Catalog("anthropic", "roomkit.providers.anthropic.models", "anthropic/"),
    Catalog("openai", "roomkit.providers.openai.models", "openai/"),
    Catalog("gemini", "roomkit.providers.gemini.models", "google/"),
    Catalog("mistral", "roomkit.providers.mistral.models", "mistralai/"),
    Catalog("xai", "roomkit.providers.xai.models", "x-ai/"),
    # Both namespaces upstream are dominated by open-weight checkpoints and
    # legacy lines the vendors' own hosted APIs no longer answer to — 13
    # deepseek ids for a two-model lineup, 47 qwen ids for five hosted ones — so
    # "something newer exists in this family" is always true and would be noise
    # rather than signal. Retirements and price drift still report.
    Catalog("deepseek", "roomkit.providers.deepseek.models", "deepseek/", track_new=False),
    Catalog("qwen", "roomkit.providers.qwen.models", "qwen/", track_new=False),
    # An intentionally partial slice of 300+ aggregated models, so "something
    # newer exists" is always true and would be noise rather than signal.
    Catalog(
        "openrouter", "roomkit.providers.openrouter.models", "", prefixed=True, track_new=False
    ),
    Catalog("gemini-image", "roomkit.providers.gemini.image_models", "google/", modality="image"),
]

# Catalogs with no upstream to compare against, and why. Printed on every run:
# a catalog nobody checks is the one that goes stale, and silence about it
# reads as coverage. Each entry is a claim that someone looked for a mirror.
UNMIRRORED_CATALOGS: dict[str, str] = {
    "openai-image": (
        "the mirror republishes chat models; OpenAI's gpt-image-* live on the images "
        "endpoint and are absent from it (checked 2026-08-07). Rates come from OpenAI's "
        "own pricing page and are only as fresh as the `verified` date they carry."
    ),
    "openai-realtime": (
        "the mirror routes chat completions; the gpt-realtime-* lineup lives on the "
        "Realtime WebSocket API and is absent from it — its gpt-audio/gpt-audio-mini "
        "are the chat-completions audio models, not these (checked 2026-08-07). Ids "
        "come from OpenAI's Realtime API docs."
    ),
    "gemini-realtime": (
        "no public aggregator mirrors the Live API lineup (checked 2026-08-07). Ids "
        "come from Google's Live API docs and are the ones this repo's examples run."
    ),
    "xai-realtime": (
        "the mirror lists xAI's chat models only; grok-2-audio is a Realtime "
        "WebSocket id absent from it (checked 2026-08-07). Sourced from the xAI "
        "audio capability docs."
    ),
}

# Upstream slugs that are routing lanes rather than distinct vendor models.
# Flagging them as "missing from the catalog" would ask for ids the vendor's
# own API rejects.
_ROUTING_SUFFIXES = ("-fast",)

# Upstream slugs the mirror invents or resolves differently, so their absence
# from a catalog is correct. Case-by-case, because the same suffix can be real
# on one tier and not another: `gpt-5.5-pro` and `gpt-5.4-pro` are genuine
# OpenAI ids, while the 5.6 tier ships no `-pro` id at all.
MIRROR_ONLY: dict[str, str] = {
    "openai/o4-mini-high": (
        "reasoning-effort route; OpenAI's API catalog lists o4-mini and its dated snapshot, "
        "not a separate -high model id (official model page, 2026-08-05)"
    ),
    "openai/gpt-5.6-sol-pro": "no -pro id in OpenAI's 5.6 tier (pricing page, 2026-08-05)",
    "openai/gpt-5.6-terra-pro": "no -pro id in OpenAI's 5.6 tier (pricing page, 2026-08-05)",
    "openai/gpt-5.6-luna-pro": "no -pro id in OpenAI's 5.6 tier (pricing page, 2026-08-05)",
}

# Catalog ids where roomkit's value is deliberately not the mirror's. Each one
# is a decision, so it carries its reason — an entry here is a claim that
# someone checked the vendor, not that the finding was inconvenient.
DELIBERATE: dict[str, str] = {
    # The mirror advertises the ceiling reachable with Anthropic's 1M-context
    # beta header. roomkit sends no beta headers of its own, so the window a
    # caller actually gets is the default 200K.
    "claude-sonnet-4-5": "200K is the no-beta default; mirror reports the 1M beta ceiling",
    "claude-sonnet-4-5-20250929": "200K is the no-beta default; mirror reports the beta ceiling",
    # OpenRouter currently mirrors the Sol/Terra ceiling onto the whole 5.6
    # family. OpenAI's migration guide gives Luna its smaller live limit.
    "gpt-5.6-luna": "OpenAI documents 400K; mirror reports the 1.05M Sol/Terra ceiling",
    # Project Glasswing only — never published on a public aggregator.
    "claude-mythos-5": "Project Glasswing access only, absent from public mirrors",
    # xAI's dated variant ids; the mirror republishes only the undated forms.
    "grok-4.20-0309-reasoning": "mirror carries the undated grok-4.20 alias instead",
    "grok-4.20-0309-non-reasoning": "mirror carries the undated grok-4.20 alias instead",
    "grok-4.20-multi-agent-0309": "mirror carries the undated alias instead",
    # Alibaba's hosted VL model. The mirror republishes the open-weight
    # qwen3-vl-* checkpoints (235b, 30b, 32b, 8b) that anyone can self-host,
    # not the `-plus` id Model Studio answers to (billing page, 2026-08-14).
    "qwen3-vl-plus": "hosted-only id; mirror carries the open-weight qwen3-vl checkpoints",
}

# Catalog ids whose *rates* deliberately differ from the mirror's, same rule as
# DELIBERATE: an entry is a claim that someone read the vendor's own price list.
PRICE_DELIBERATE: dict[str, str] = {
    # The mirror resells these two at exactly OpenAI's Batch column ($1/$6 and
    # $0.10/$0.60). roomkit calls the synchronous API, so the catalog carries
    # the standard rate from OpenAI's pricing page (2026-08-05).
    "gpt-5.6-terra": "mirror quotes the Batch rate; roomkit bills the standard one",
    "gpt-5.6-luna": "mirror quotes the Batch rate; roomkit bills the standard one",
    # Same divergence, different cause: the mirror is running a 50% promotion on
    # its own `openai/gpt-5.6-sol` slug, which lands on the Batch numbers by
    # coincidence. OpenAI's standard rate is $5/$30 with cached input at 10% and
    # cache writes at 1.25x (model docs, 2026-08-18) — a discount the aggregator
    # funds is not a price cut by the vendor, and the openrouter catalog carries
    # the resold rate in its own entry.
    "gpt-5.6-sol": "mirror runs a promo on its own slug; roomkit bills OpenAI's list rate",
    # Same shape, Google's side: the mirror resells 3.7 Flash at exactly the
    # Batch/Flex column ($0.375/$1.875/$0.0375) while Google's synchronous rate
    # is double that (pricing page, 2026-08-13). Its own `openrouter` entry
    # carries the resold rate, because there the mirror is the seller.
    "gemini-3.7-flash": "mirror resells at the Batch rate; roomkit bills Google's standard one",
    # No seller to agree with. Eighteen hosts serve this open-weights model
    # through OpenRouter between $0.42 and $1.74 per million input, and the
    # top-level quote follows whichever endpoint routing prefers: $0.435 on
    # 2026-08-05, $0.63168 on the 11th, $1.168 on the 13th. Refreshing the
    # catalog to match is a treadmill that lands on a different number every
    # release, so the entry states DeepSeek's own endpoint and stays put.
    "deepseek/deepseek-v4-pro": "multi-host model; upstream quotes the routed endpoint",
    # Same fact from the other side. RoomKit's deepseek provider calls
    # DeepSeek's own endpoint, so its catalog carries DeepSeek's own rates
    # (api-docs.deepseek.com pricing, 2026-08-14) while the mirror quotes
    # whichever of its eighteen hosts routing picked — above the vendor for
    # pro, below it for flash, and a different number every week. Specifically
    # the catalog carries the peak column of the peak/off-peak schedule
    # effective 2026-08-16 16:00 UTC, which is the undiscounted rate; an
    # off-peak call bills half. Note what these two entries cost: a suppressed
    # model is invisible to the price gate in *both* directions, so a genuine
    # DeepSeek repricing will not be reported here either.
    "deepseek-v4-pro": "catalog states DeepSeek's own endpoint; upstream quotes a routed host",
    "deepseek-v4-flash": "catalog states DeepSeek's own endpoint; upstream quotes a routed host",
    # Alibaba runs near-permanent limited-time promotions and the mirror resells
    # at a discounted rate (qwen3.7-plus at exactly the 20% off price, the other
    # two at rates matching no published column). ModelPricing carries one list
    # rate, so the catalog states Alibaba's international list price
    # (billing-for-model-studio, 2026-08-14) and the promotion lands as an
    # under-charge. The same entries also cover cache write, which upstream
    # quotes at Alibaba's 125% explicit-cache-creation rate: this provider never
    # creates an explicit cache, so the catalog leaves that rate unset.
    "qwen3.7-max": "mirror resells at a discounted rate; catalog carries Alibaba's list price",
    "qwen3.7-plus": "mirror resells at the 20%-off promotion; catalog carries the list price",
    "qwen3.6-flash": "mirror resells at a discounted rate; catalog carries Alibaba's list price",
}

# Per-field normalization where the upstream pricing object quotes only one
# component but RoomKit's disjoint usage counter must carry the complete charge.
PRICE_FIELD_DELIBERATE: dict[tuple[str, str, str], str] = {
    (
        "openrouter",
        "google/gemini-3.6-flash",
        "input_cache_write",
    ): "upstream quotes the 5-minute storage premium; RoomKit adds ordinary input",
    (
        "openrouter",
        "google/gemini-3.5-flash",
        "input_cache_write",
    ): "upstream quotes the 5-minute storage premium; RoomKit adds ordinary input",
    (
        "openrouter",
        "google/gemini-3.7-flash",
        "input_cache_write",
    ): "upstream quotes the 5-minute storage premium; RoomKit adds ordinary input",
}

# Upstream sometimes squeezes a charge with different units into its token
# pricing object. Those values cannot populate ModelPricing without producing
# a dimensionally wrong cost. Keep exceptions narrow: provider + exact field.
UNCOMPARABLE_PRICE_FIELDS: dict[tuple[str, str], str] = {
    ("gemini", "input_cache_write"): "Google bills cache storage by token-hour, not per write",
    # Same Google billing fact, and the exception is keyed per catalog so the
    # image lineup needs its own entry rather than inheriting the chat one.
    ("gemini-image", "input_cache_write"): (
        "Google bills cache storage by token-hour, not per write"
    ),
}

# (ModelPricing attribute, upstream pricing key, label) — upstream quotes USD
# per token, the catalog quotes USD per million.
_TEXT_PRICE_FIELDS = (
    ("input_per_million", "prompt", "input"),
    ("output_per_million", "completion", "output"),
    ("cache_read_per_million", "input_cache_read", "cache read"),
    ("cache_write_per_million", "input_cache_write", "cache write"),
)

# An image catalog carries the text rates too — an image model charges for the
# prompt that describes the picture. `image` is the vendor's image-input rate;
# for every model mirrored here it is quoted per token and equals `prompt`
# exactly, so it compares on the same scale as the rest.
_IMAGE_PRICE_FIELDS = (
    *_TEXT_PRICE_FIELDS,
    ("image_input_per_million", "image", "image input"),
    ("image_output_per_million", "image_output", "image output"),
)

_PRICE_FIELDS_BY_MODALITY = {
    "text": _TEXT_PRICE_FIELDS,
    "image": _IMAGE_PRICE_FIELDS,
}

# A dateless alias (`mistral-large-latest`) or a dated snapshot
# (`claude-haiku-4-5-20251001`) is a form the vendor accepts but a mirror does
# not republish, so absence upstream says nothing about it.
_SNAPSHOT_RE = re.compile(r"-\d{8}$")


@dataclass
class Findings:
    """Everything one catalog turned up, split by severity."""

    drift: list[str] = field(default_factory=list)
    price: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)
    expected: int = 0
    """Divergences suppressed by ``DELIBERATE`` / ``MIRROR_ONLY`` /
    ``PRICE_DELIBERATE`` / the field-level price exceptions. Counted so a
    growing pile of exceptions stays visible instead of hiding in the source."""

    @property
    def errors(self) -> list[str]:
        return self.drift + self.price + self.missing


def normalize(model_id: str) -> str:
    """Fold the punctuation vendors and mirrors disagree on.

    Anthropic writes ``claude-opus-4-8`` where OpenRouter writes
    ``claude-opus-4.8``; the two name one model.
    """
    return model_id.lower().replace(".", "-")


def is_alias(model_id: str) -> bool:
    """Whether an id is an alias or dated snapshot rather than a base id."""
    return model_id.endswith("-latest") or bool(_SNAPSHOT_RE.search(model_id))


def family(model_id: str) -> str:
    """The leading family token — ``gemini`` in ``gemini-3.6-flash``.

    Used to keep a vendor's open-weights lines out of its hosted-API catalog:
    Google publishes ``gemma-*`` under the same upstream namespace as
    ``gemini-*``, and only one of those belongs here.
    """
    return re.split(r"[-.]", model_id, maxsplit=1)[0]


def price_findings(
    label: str, model: ModelInfo, upstream: dict[str, Any], modality: str = "text"
) -> list[str]:
    """Compare the rates a catalog entry states against upstream's quote.

    A positive upstream quote that the catalog leaves unset is a finding. The
    only exceptions are explicit provider/field pairs whose units cannot be
    represented by :class:`ModelPricing`.
    """
    if model.pricing is None:
        return []
    quoted = upstream.get("pricing") or {}
    findings: list[str] = []
    for attribute, key, name in _PRICE_FIELDS_BY_MODALITY[modality]:
        if (label, model.id, key) in PRICE_FIELD_DELIBERATE:
            continue
        ours = getattr(model.pricing, attribute)
        theirs = quoted.get(key)
        if theirs in (None, ""):
            continue
        upstream_rate = float(theirs) * 1_000_000
        if ours is None:
            if upstream_rate == 0 or (label, key) in UNCOMPARABLE_PRICE_FIELDS:
                continue
            findings.append(
                f"{label}: {model.id} has no {name} rate but upstream quotes ${upstream_rate:g}/M"
            )
            continue
        if not math.isclose(ours, upstream_rate, rel_tol=1e-6):
            findings.append(
                f"{label}: {model.id} {name} ${ours:g}/M but upstream quotes ${upstream_rate:g}/M"
            )
    return findings


def expected_price_divergences(
    label: str, model: ModelInfo, upstream: dict[str, Any], modality: str = "text"
) -> int:
    """Count explicit field-level suppressions that apply to this quote."""
    if model.pricing is None:
        return 0
    quoted = upstream.get("pricing") or {}
    expected = 0
    for attribute, key, _name in _PRICE_FIELDS_BY_MODALITY[modality]:
        theirs = quoted.get(key)
        if theirs in (None, ""):
            continue
        if (label, model.id, key) in PRICE_FIELD_DELIBERATE or (
            getattr(model.pricing, attribute) is None
            and float(theirs) != 0
            and (label, key) in UNCOMPARABLE_PRICE_FIELDS
        ):
            expected += 1
    return expected


def belongs_to(item: dict[str, Any], modality: str) -> bool:
    """Whether an upstream model belongs to the slice a catalog compares against.

    The two slices are not symmetric, because roomkit's catalogs are not. An
    image model advertises ``["image", "text"]`` — it narrates what it drew —
    so the image slice is "can output an image", while the text slice is "text
    is *all* it outputs". Reading both as "contains the modality" would file
    every image model in the chat catalog too, and report it missing there.
    """
    modalities = (item.get("architecture") or {}).get("output_modalities") or []
    if modality == "text":
        return modalities == ["text"]
    return modality in modalities


def fetch_upstream(url: str = UPSTREAM_URL, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Return every upstream model that declares an output modality.

    Both slices this script compares against — text and image — come from the
    one request; :func:`belongs_to` splits them. Raises on any fetch failure.
    """
    request = urllib.request.Request(  # noqa: S310 - constant https URL
        url, headers={"User-Agent": "roomkit-check-models"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{url} returned no model data")
    return [item for item in data if (item.get("architecture") or {}).get("output_modalities")]


def check_catalog(catalog: Catalog, upstream: list[dict[str, Any]]) -> Findings:
    """Compare one curated catalog against the upstream slice for its vendor."""
    label, prefix = catalog.label, catalog.prefix
    curated = list(import_module(catalog.module).MODELS)
    found = Findings()

    # Upstream models for this vendor and modality, keyed by the normalized
    # bare id.
    scope = {}
    for item in upstream:
        raw = str(item.get("id", ""))
        if ":" in raw or not raw.startswith(prefix):
            continue
        if not belongs_to(item, catalog.modality):
            continue
        bare = raw if catalog.prefixed else raw[len(prefix) :]
        scope[normalize(bare)] = item

    newest_known_by_family: dict[str, int] = {}

    for model in curated:
        match = scope.get(normalize(model.id))
        if match is None:
            if model.id in DELIBERATE:
                found.expected += 1
            elif not model.deprecated and not is_alias(model.id):
                found.gone.append(f"{label}: {model.id} not listed upstream — retired?")
            continue
        model_family = family(normalize(model.id))
        newest_known_by_family[model_family] = max(
            newest_known_by_family.get(model_family, 0),
            int(match.get("created") or 0),
        )
        window = match.get("context_length")
        if model.id in DELIBERATE:
            found.expected += 1
        elif model.context_window is not None and window and model.context_window != window:
            found.drift.append(
                f"{label}: {model.id} context_window={model.context_window:,} "
                f"but upstream reports {window:,}"
            )
        # Rates are checked even where the window is a known divergence: the
        # two disagree for unrelated reasons, and a suppressed window should
        # not silence a wrong price.
        if model.id in PRICE_DELIBERATE:
            found.expected += 1
        else:
            found.expected += expected_price_divergences(label, model, match, catalog.modality)
            found.price.extend(price_findings(label, model, match, catalog.modality))

    if not (catalog.track_new and newest_known_by_family):
        return found

    curated_ids = {normalize(m.id) for m in curated}
    for key, item in sorted(scope.items(), key=lambda kv: -int(kv[1].get("created") or 0)):
        item_family = family(key)
        newest_known = newest_known_by_family.get(item_family)
        if key in curated_ids or newest_known is None:
            continue
        if key.endswith(_ROUTING_SUFFIXES):
            continue
        if item["id"] in MIRROR_ONLY:
            found.expected += 1
            continue
        if int(item.get("created") or 0) > newest_known:
            found.missing.append(
                f"{label}: {item['id']} is newer than the catalogued {item_family} family "
                f"(context {item.get('context_length'):,})"
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=UPSTREAM_URL, help="upstream models endpoint")
    args = parser.parse_args()

    try:
        upstream = fetch_upstream(args.url)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"could not reach {args.url}: {exc}", file=sys.stderr)
        print("model catalogs NOT verified", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []
    expected = 0
    for catalog in CATALOGS:
        found = check_catalog(catalog, upstream)
        errors.extend(found.errors)
        warnings.extend(found.gone)
        expected += found.expected

    for label, reason in UNMIRRORED_CATALOGS.items():
        print(f"unmirrored  {label}: {reason}")
    for warning in warnings:
        print(f"warning  {warning}")
    for error in errors:
        print(f"STALE    {error}")

    noted = f", {expected} known divergence(s) allowed" if expected else ""
    if errors:
        print(
            f"\n{len(errors)} finding(s){noted}. Confirm each against the vendor's own docs — "
            "this reads a mirror — then update providers/*/models.py, or record the "
            "divergence in scripts/check_models.py with its reason.",
            file=sys.stderr,
        )
        return 1

    unmirrored = (
        f", {len(UNMIRRORED_CATALOGS)} catalog(s) with no mirror" if UNMIRRORED_CATALOGS else ""
    )
    print(
        f"{len(CATALOGS)} model catalogs match upstream "
        f"({len(upstream)} models fetched{noted}{unmirrored})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
