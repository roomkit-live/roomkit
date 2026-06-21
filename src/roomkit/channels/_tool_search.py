"""Channel-agnostic Tool Search core (scoring + result rendering).

Both the realtime channel (:class:`RealtimeVoiceChannel`) and the text /
HTTP agent loop (:class:`AIChannel`) expose a large tool catalogue behind
two discovery tools — ``find_tools`` and ``list_tools`` — instead of
sending every schema to the model. The *delivery* differs: realtime pushes
newly-matched tools via ``provider.reconfigure`` and keys exposure by live
session; the AIChannel re-sends its tool list every tool-loop round and
keys exposure by the per-turn loop context. But the *scoring*, the
*result payloads* and the *catalogue listing* are identical — they live
here so both paths share one implementation.

The scoring / rendering helpers operate on plain tool dicts (``{"name",
"description", ...}``) — the lingua franca both channels can produce. The
text-channel activation policy (:func:`should_activate_tool_search`) and the
discovery-tool definitions (:func:`search_tool_defs`) work with ``AITool``
objects, since that is what the AIChannel assembles per turn.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from roomkit.channels._tool_search_constants import (
    DEFAULT_FIND_TOOLS_LIMIT,
    FIND_TOOLS_SCHEMA,
    LIST_TOOLS_SCHEMA,
)
from roomkit.memory.token_estimator import estimate_tool_tokens
from roomkit.providers.ai.base import AITool

_TOKEN_SPLIT = re.compile(r"[\W_]+")
# Split camelCase / PascalCase so a tool named "SpotifySearch" tokenizes to
# ["spotify", "search"] (it would otherwise be one opaque token "spotifysearch"
# that no query word matches). Edge / device tool names are commonly PascalCase;
# MCP tools are snake_case and tokenize fine on their own.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

# A name-token match counts this many times a description-token match.
_NAME_WEIGHT = 3
# A tag-token match weighs as much as a name match. Tags are curated English
# keywords — the deliberate cross-lingual bridge — so they must weigh enough to
# surface a tool whose name/description are written in another language.
_TAGS_WEIGHT = 3
# Keep only matches scoring within this fraction of the best match — an
# incidental hit is dropped when a genuinely relevant tool exists, but a
# uniformly-weak query still returns its best candidates rather than nothing.
_RELATIVE_SCORE_CUTOFF = 0.5
# Per-match description cap in a find_tools result: enough to identify the tool,
# small enough that a handful of verbose tools can't overflow the result. The
# model gets the full description + schema when the tool is revealed into its
# tool list.
_MATCH_DESC_LIMIT = 200
# list_tools is an *inventory* (potentially the whole catalogue), so its
# per-entry description is tiny — a one-line gist. Dumping full descriptions
# here would re-send the catalogue and defeat Tool Search; the model uses
# find_tools to get details and to act.
_LIST_DESC_LIMIT = 64


def tokenize(text: str) -> list[str]:
    text = _ACRONYM_BOUNDARY.sub(" ", _CAMEL_BOUNDARY.sub(" ", text or ""))
    return [t.lower() for t in _TOKEN_SPLIT.split(text) if t]


def normalize_max_results(raw: Any, threshold: int) -> int:
    """Clamp a model-supplied ``max_results`` to ``[1, threshold]``."""
    try:
        value = int(raw) if raw is not None else DEFAULT_FIND_TOOLS_LIMIT
    except (TypeError, ValueError):
        value = DEFAULT_FIND_TOOLS_LIMIT
    return max(1, min(value, threshold))


def search_catalogue(
    catalogue: list[dict[str, Any]],
    query: str,
    max_results: int,
    *,
    exclude_names: frozenset[str] | set[str],
) -> list[dict[str, Any]]:
    """Return the top-``max_results`` tools matching ``query`` by score.

    ``exclude_names`` skips tools that are always visible anyway (search
    infra + pinned) so a match never points the model back at a tool it
    can already see.

    Scoring is IDF-weighted: each query word is weighted by how *rare* it is in
    this catalogue, so a word in nearly every tool (``on``, ``the``, ``de``,
    ``la``) contributes little while a discriminating word (``spotify``)
    dominates — no stopword list, and it adapts to the catalogue and the query's
    language. The IDF is smoothed (``log((n+1)/(df+1)) + 1``) so it never
    collapses to zero on a tiny catalogue. Name matches weigh ``_NAME_WEIGHT``×
    and tag matches ``_TAGS_WEIGHT``× (tags are curated English keywords — the
    cross-lingual bridge — so an English-normalized query matches them even when
    the tool's own name/description are in another language). Only matches within
    ``_RELATIVE_SCORE_CUTOFF`` of the best are kept, which drops the
    common-word-only hits once a genuinely relevant tool exists.
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    # Tokenize every candidate once and build document frequency for IDF.
    candidates: list[tuple[dict[str, Any], set[str], set[str], set[str]]] = []
    df: dict[str, int] = {}
    for tool in catalogue:
        if tool.get("name", "") in exclude_names:
            continue
        name_tokens = set(tokenize(tool.get("name", "")))
        desc_tokens = set(tokenize(tool.get("description", "")))
        tags_tokens = set(tokenize(" ".join(str(t) for t in (tool.get("tags") or []))))
        candidates.append((tool, name_tokens, desc_tokens, tags_tokens))
        for tok in name_tokens | desc_tokens | tags_tokens:
            df[tok] = df.get(tok, 0) + 1
    n = len(candidates)
    if n == 0:
        return []

    weights = {
        q: (math.log((n + 1) / (df[q] + 1)) + 1 if df.get(q) else 0.0) for q in query_tokens
    }

    scored: list[tuple[float, dict[str, Any]]] = []
    for tool, name_tokens, desc_tokens, tags_tokens in candidates:
        s = 0.0
        for q in query_tokens:
            w = weights[q]
            if q in name_tokens:
                s += _NAME_WEIGHT * w
            if q in tags_tokens:
                s += _TAGS_WEIGHT * w
            if q in desc_tokens:
                s += w
        if s > 0:
            scored.append((s, tool))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    cutoff = scored[0][0] * _RELATIVE_SCORE_CUTOFF
    return [tool for s, tool in scored if s >= cutoff][:max_results]


def render_find_payload(matches: list[dict[str, Any]]) -> str:
    """JSON result for a ``find_tools`` call — compact: name + truncated
    description per match.

    Deliberately omits the parameter schema and truncates the description: a
    match's only job is to tell the model which tools it can now call. The full
    schema is delivered separately — the text loop re-sends it in the next
    round's tool list, the realtime channel pushes it via ``provider.reconfigure``
    — so inlining it here would only risk blowing the tool-result size limit
    (verbose multi-action tools like ``outlook``/``gmail`` carry huge
    descriptions + schemas; a few of them would overflow the result and get
    evicted, defeating the whole point of the search).
    """
    payload: dict[str, Any] = {
        "matches": [
            {
                "name": tool.get("name"),
                "description": (tool.get("description") or "")[:_MATCH_DESC_LIMIT],
            }
            for tool in matches
        ],
        "_note": (
            "These tools are now invocable — their full schemas are in your tool "
            "list. Call the right one directly. Do not call find_tools again "
            "unless none of these fit."
        ),
    }
    if not matches:
        payload["_note"] = (
            "No tools matched. Try a different query, or call list_tools "
            "to see all available tools."
        )
    return json.dumps(payload)


def render_list_payload(
    catalogue: list[dict[str, Any]],
    category: str,
    *,
    exclude_names: frozenset[str] | set[str],
    limit: int = 60,
) -> str:
    """JSON inventory for a ``list_tools`` call (name + one-line gist).

    Deliberately tiny per entry (``_LIST_DESC_LIMIT``): this can be the whole
    catalogue, so full descriptions would re-send everything and defeat Tool
    Search. It's an orientation aid — the model uses find_tools for details and
    to act. Still capped at ``limit`` entries as a final backstop.
    """
    items: list[dict[str, str]] = []
    for tool in catalogue:
        name = tool.get("name", "")
        if name in exclude_names:
            continue
        if category and not name.startswith(category):
            continue
        items.append(
            {"name": name, "description": (tool.get("description") or "")[:_LIST_DESC_LIMIT]}
        )

    truncated = len(items) > limit
    if truncated:
        items = items[:limit]
    payload: dict[str, Any] = {"tools": items, "count": len(items)}
    if truncated:
        payload["truncated"] = True
        payload["_note"] = (
            f"Result truncated to {limit} entries. Use category= to filter, "
            "or prefer find_tools(query=...) for action."
        )
    return json.dumps(payload)


# -- Text-channel integration (AITool-based) --------------------------------


def search_tool_defs() -> list[AITool]:
    """The two discovery tools (find_tools / list_tools) as AITool definitions."""
    return [
        AITool(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"],
        )
        for schema in (FIND_TOOLS_SCHEMA, LIST_TOOLS_SCHEMA)
    ]


def should_activate_tool_search(
    *,
    mode: bool | None,
    catalogue: list[AITool],
    pinned: set[str],
    window: int | None,
    threshold_pct: float,
    threshold_count: int,
) -> bool:
    """Decide whether Tool Search hides the catalogue for a turn.

    ``mode`` ``True``/``False`` forces on/off. In ``auto`` mode (``None``) the
    decision self-tunes to the model: defer when the *deferrable* tools
    (everything except ``pinned``) would cost more than ``threshold_pct`` % of
    the model's context ``window``. When ``window`` is unknown (custom / local
    model absent from the provider catalog) it falls back to the
    ``threshold_count`` tool count — the floor that still protects small local
    models whose window cannot be read.

    ``catalogue`` is the real tool list, BEFORE the search infra tools are
    injected, so it reflects the deferrable surface only.
    """
    if mode is False:
        return False
    if mode is True:
        return True
    if window:
        budget = window * threshold_pct / 100
        deferrable = (t for t in catalogue if t.name not in pinned)
        return sum(estimate_tool_tokens(t) for t in deferrable) > budget
    return len(catalogue) > threshold_count
