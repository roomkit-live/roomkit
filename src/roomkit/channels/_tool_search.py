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


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_SPLIT.split(text or "") if t]


def score(query_tokens: list[str], tool: dict[str, Any]) -> int:
    """Cheap fuzzy match: count overlapping word tokens.

    Name matches weigh 3x, description matches weigh 1x. No external
    dependencies; runs in microseconds for a few-hundred-tool catalogue
    so it's safe in the realtime hot path.
    """
    if not query_tokens:
        return 0
    name_tokens = set(tokenize(tool.get("name", "")))
    desc_tokens = set(tokenize(tool.get("description", "")))
    s = 0
    for q in query_tokens:
        if q in name_tokens:
            s += 3
        if q in desc_tokens:
            s += 1
    return s


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
    """
    query_tokens = tokenize(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for tool in catalogue:
        name = tool.get("name", "")
        if name in exclude_names:
            continue
        s = score(query_tokens, tool)
        if s > 0:
            scored.append((s, tool))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [tool for _, tool in scored[:max_results]]


def render_find_payload(matches: list[dict[str, Any]], *, include_schema: bool = False) -> str:
    """JSON result for a ``find_tools`` call.

    By default each match carries only ``name`` + ``description`` (compact, so a
    realtime model does not derail on a long return — it gets the full schema
    via ``provider.reconfigure``). When ``include_schema`` is set, each match
    also carries its ``parameters`` JSON schema, so a model that gets the result
    inline (the text/HTTP loop) can call the matched tool correctly in one shot
    instead of guessing the arguments.
    """

    def _match(tool: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": tool.get("name"),
            "description": tool.get("description"),
        }
        if include_schema:
            entry["parameters"] = tool.get("parameters") or {}
        return entry

    payload: dict[str, Any] = {
        "matches": [_match(tool) for tool in matches],
        "_note": (
            "These tools are now invocable. Call the right one directly. "
            "Do not call find_tools again unless none of these fit."
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
    """JSON inventory for a ``list_tools`` call (name + short description).

    Bounded to ``limit`` entries so the result never blows a realtime
    tool-result threshold (~60 × ~160 chars ≈ 10 KB worst case).
    """
    items: list[dict[str, str]] = []
    for tool in catalogue:
        name = tool.get("name", "")
        if name in exclude_names:
            continue
        if category and not name.startswith(category):
            continue
        items.append({"name": name, "description": (tool.get("description") or "")[:200]})

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
