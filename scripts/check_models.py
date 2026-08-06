#!/usr/bin/env python3
"""Check the offline model catalogs against a live upstream source.

The catalogs in ``providers/*/models.py`` exist so ``AIProvider.context_window``
can answer without a network call. That makes them useful and stale-able at the
same time: nothing in a test suite notices when a vendor ships a new flagship or
changes a window, because the catalog is self-consistent either way. This script
is what notices, so a release does not quietly ship last quarter's lineup.

The reference is OpenRouter's ``/api/v1/models``, which is public, needs no key,
and republishes ids, context windows, and modalities for every major vendor —
one request covers Anthropic, OpenAI, Google, Mistral, and xAI. It is a mirror,
not the vendor, so treat a finding as "go read the vendor's docs", never as a
value to paste in blind.

Ollama (a local pull registry) and PolarGrid (private edges) are not mirrored
there and are not checked; they carry their own verification dates.

Reported:

  DRIFT    a catalog id whose context window disagrees with upstream — an
           actively wrong number, the one failure that silently mistrims
           conversation history.
  PRICE    a rate the catalog states and upstream quotes differently — the
           failure that bills the wrong amount. A non-zero upstream rate that
           the catalog leaves unset is also reported, unless the units are
           explicitly known to be incomparable.
  MISSING  an upstream model newer than everything the catalog knows, in a
           family the catalog already tracks — i.e. the vendor moved on.
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

# Catalogs mirrored upstream. `prefixed` marks a catalog whose own ids already
# carry the vendor namespace (OpenRouter's own), so nothing is stripped.
# `track_new` is off for that one: its catalog is an intentionally partial slice
# of 300+ aggregated models, so "something newer exists" is always true and
# would be noise rather than signal.
CATALOGS: list[tuple[str, str, str, bool, bool]] = [
    # (label, module, upstream vendor prefix, prefixed, track_new)
    ("anthropic", "roomkit.providers.anthropic.models", "anthropic/", False, True),
    ("openai", "roomkit.providers.openai.models", "openai/", False, True),
    ("gemini", "roomkit.providers.gemini.models", "google/", False, True),
    ("mistral", "roomkit.providers.mistral.models", "mistralai/", False, True),
    ("xai", "roomkit.providers.xai.models", "x-ai/", False, True),
    ("openrouter", "roomkit.providers.openrouter.models", "", True, False),
]

# Upstream slugs that are routing lanes rather than distinct vendor models.
# Flagging them as "missing from the catalog" would ask for ids the vendor's
# own API rejects.
_ROUTING_SUFFIXES = ("-fast",)

# Upstream slugs the mirror invents or resolves differently, so their absence
# from a catalog is correct. Case-by-case, because the same suffix can be real
# on one tier and not another: `gpt-5.5-pro` and `gpt-5.4-pro` are genuine
# OpenAI ids, while the 5.6 tier ships no `-pro` id at all.
MIRROR_ONLY: dict[str, str] = {
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
    # Project Glasswing only — never published on a public aggregator.
    "claude-mythos-5": "Project Glasswing access only, absent from public mirrors",
    # xAI's dated variant ids; the mirror republishes only the undated forms.
    "grok-4.20-0309-reasoning": "mirror carries the undated grok-4.20 alias instead",
    "grok-4.20-0309-non-reasoning": "mirror carries the undated grok-4.20 alias instead",
    "grok-4.20-multi-agent-0309": "mirror carries the undated alias instead",
}

# Catalog ids whose *rates* deliberately differ from the mirror's, same rule as
# DELIBERATE: an entry is a claim that someone read the vendor's own price list.
PRICE_DELIBERATE: dict[str, str] = {
    # The mirror resells these two at exactly OpenAI's Batch column ($1/$6 and
    # $0.10/$0.60). roomkit calls the synchronous API, so the catalog carries
    # the standard rate from OpenAI's pricing page (2026-08-05).
    "gpt-5.6-terra": "mirror quotes the Batch rate; roomkit bills the standard one",
    "gpt-5.6-luna": "mirror quotes the Batch rate; roomkit bills the standard one",
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
}

# Upstream sometimes squeezes a charge with different units into its token
# pricing object. Those values cannot populate ModelPricing without producing
# a dimensionally wrong cost. Keep exceptions narrow: provider + exact field.
UNCOMPARABLE_PRICE_FIELDS: dict[tuple[str, str], str] = {
    ("gemini", "input_cache_write"): "Google bills cache storage by token-hour, not per write",
}

# (ModelPricing attribute, upstream pricing key, label) — upstream quotes USD
# per token, the catalog quotes USD per million.
_PRICE_FIELDS = (
    ("input_per_million", "prompt", "input"),
    ("output_per_million", "completion", "output"),
    ("cache_read_per_million", "input_cache_read", "cache read"),
    ("cache_write_per_million", "input_cache_write", "cache write"),
)

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


def price_findings(label: str, model: ModelInfo, upstream: dict[str, Any]) -> list[str]:
    """Compare the rates a catalog entry states against upstream's quote.

    A positive upstream quote that the catalog leaves unset is a finding. The
    only exceptions are explicit provider/field pairs whose units cannot be
    represented by :class:`ModelPricing`.
    """
    if model.pricing is None:
        return []
    quoted = upstream.get("pricing") or {}
    findings: list[str] = []
    for attribute, key, name in _PRICE_FIELDS:
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


def expected_price_divergences(label: str, model: ModelInfo, upstream: dict[str, Any]) -> int:
    """Count explicit field-level suppressions that apply to this quote."""
    if model.pricing is None:
        return 0
    quoted = upstream.get("pricing") or {}
    expected = 0
    for attribute, key, _name in _PRICE_FIELDS:
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


def fetch_upstream(url: str = UPSTREAM_URL, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Return upstream's text-output chat models. Raises on any fetch failure."""
    request = urllib.request.Request(  # noqa: S310 - constant https URL
        url, headers={"User-Agent": "roomkit-check-models"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{url} returned no model data")
    return [
        item
        for item in data
        if (item.get("architecture") or {}).get("output_modalities") == ["text"]
    ]


def check_catalog(
    label: str,
    module: str,
    prefix: str,
    prefixed: bool,
    track_new: bool,
    upstream: list[dict[str, Any]],
) -> Findings:
    """Compare one curated catalog against the upstream slice for its vendor."""
    curated = list(import_module(module).MODELS)
    found = Findings()

    # Upstream models for this vendor, keyed by the normalized bare id.
    scope = {}
    for item in upstream:
        raw = str(item.get("id", ""))
        if ":" in raw or not raw.startswith(prefix):
            continue
        bare = raw if prefixed else raw[len(prefix) :]
        scope[normalize(bare)] = item

    known_families = {family(normalize(m.id)) for m in curated}
    newest_known = 0

    for model in curated:
        match = scope.get(normalize(model.id))
        if match is None:
            if model.id in DELIBERATE:
                found.expected += 1
            elif not model.deprecated and not is_alias(model.id):
                found.gone.append(f"{label}: {model.id} not listed upstream — retired?")
            continue
        newest_known = max(newest_known, int(match.get("created") or 0))
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
            found.expected += expected_price_divergences(label, model, match)
            found.price.extend(price_findings(label, model, match))

    if not (track_new and newest_known):
        return found

    curated_ids = {normalize(m.id) for m in curated}
    for key, item in sorted(scope.items(), key=lambda kv: -int(kv[1].get("created") or 0)):
        if key in curated_ids or family(key) not in known_families:
            continue
        if key.endswith(_ROUTING_SUFFIXES):
            continue
        if item["id"] in MIRROR_ONLY:
            found.expected += 1
            continue
        if int(item.get("created") or 0) > newest_known:
            found.missing.append(
                f"{label}: {item['id']} is newer than anything in the catalog "
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
    for label, module, prefix, prefixed, track_new in CATALOGS:
        found = check_catalog(label, module, prefix, prefixed, track_new, upstream)
        errors.extend(found.errors)
        warnings.extend(found.gone)
        expected += found.expected

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

    print(
        f"{len(CATALOGS)} model catalogs match upstream ({len(upstream)} models compared{noted})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
