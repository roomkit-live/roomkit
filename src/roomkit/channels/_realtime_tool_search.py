"""Tool Search support for RealtimeVoiceChannel.

Google's Gemini Live docs recommend keeping the active tool set to 10–20.
Above that, function-calling reliability degrades sharply. This mixin
implements the dynamic-tool-selection pattern Google itself points to:

* The model only sees ``find_tools``, ``list_tools``, and a small pinned
  set at session start.
* When the model calls ``find_tools(query)``, the mixin scores every
  catalogue tool against the query, returns the top-N matches as the
  tool result, and triggers ``provider.reconfigure`` so the matched
  tools become directly invocable.
* The pinned set + most recently revealed matches form the live tool
  surface. Calling ``find_tools`` again with a new query swaps the
  match window so the surface stays small (≤ ``threshold``).

Activation is per-session — each session has its own match window so
parallel calls do not cross-contaminate.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from roomkit.channels._tool_search_constants import (
    DEFAULT_FIND_TOOLS_LIMIT,
    FIND_TOOLS_SCHEMA,
    LIST_TOOLS_SCHEMA,
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
    TOOL_SEARCH_INFRA_TOOL_NAMES,
)

logger = logging.getLogger("roomkit.channels.realtime_voice")


_TOKEN_SPLIT = re.compile(r"[\W_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_SPLIT.split(text or "") if t]


def _score(query_tokens: list[str], tool: dict[str, Any]) -> int:
    """Cheap fuzzy match: count overlapping word tokens.

    Name matches weigh 3x, description matches weigh 1x. No external
    dependencies; runs in microseconds for a few-hundred-tool catalogue
    so it's safe in the realtime hot path.
    """
    if not query_tokens:
        return 0
    name_tokens = set(_tokenize(tool.get("name", "")))
    desc_tokens = set(_tokenize(tool.get("description", "")))
    score = 0
    for q in query_tokens:
        if q in name_tokens:
            score += 3
        if q in desc_tokens:
            score += 1
    return score


class RealtimeToolSearchSupport:
    """Catalogue + dynamic exposure for tool-heavy realtime channels.

    Owns the *full* tool list. The channel sees only what
    ``visible_tools(session_id)`` returns — the pinned set plus the
    matches revealed by the most recent ``find_tools`` call on this
    session.
    """

    def __init__(
        self,
        catalogue: list[dict[str, Any]],
        *,
        pinned: list[str] | None = None,
        threshold: int = 20,
    ) -> None:
        self._catalogue: list[dict[str, Any]] = list(catalogue)
        self._by_name: dict[str, dict[str, Any]] = {
            t.get("name", ""): t for t in catalogue if t.get("name")
        }
        self._pinned_names: set[str] = set(pinned or [])
        self._threshold = threshold
        # session_id -> set of tool names currently exposed by find_tools
        self._exposed: dict[str, set[str]] = {}

    # -- Tool definitions (channel injects these into the live tool list) --

    def search_tool_dicts(self) -> list[dict[str, Any]]:
        return [FIND_TOOLS_SCHEMA, LIST_TOOLS_SCHEMA]

    def is_search_tool(self, name: str) -> bool:
        return name in TOOL_SEARCH_INFRA_TOOL_NAMES

    # -- Per-session lifecycle --

    def init_session(self, session_id: str) -> None:
        self._exposed[session_id] = set()

    def cleanup_session(self, session_id: str) -> None:
        self._exposed.pop(session_id, None)

    # -- Visibility (replaces the channel's full tool list) --

    def visible_tools(
        self, session_id: str, base_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return the slice of the catalogue that should be live right now.

        Always includes search infra + pinned + currently-exposed matches.
        ``base_tools`` is the original list the channel was constructed
        with; we use it only to preserve ordering for deterministic output.
        """
        exposed = self._exposed.get(session_id, set())
        keep = self._pinned_names | exposed | TOOL_SEARCH_INFRA_TOOL_NAMES
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        # Search tools first so they sit at the top of the model's
        # attention window.
        for schema in self.search_tool_dicts():
            n = schema["name"]
            if n not in seen:
                result.append(schema)
                seen.add(n)
        for tool in base_tools:
            n = tool.get("name", "")
            if n in keep and n not in seen:
                result.append(tool)
                seen.add(n)
        return result

    # -- Tool dispatch --

    async def handle_tool_call(
        self, name: str, arguments: dict[str, Any], session_id: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Handle find_tools / list_tools.

        Returns ``(json_result, updated_tool_list_or_None)``. When the
        second element is non-None, the caller MUST push it via
        ``provider.reconfigure(tools=...)`` so the realtime model sees
        the newly-exposed matches.
        """
        if name == TOOL_FIND_TOOLS:
            return self._handle_find_tools(arguments, session_id)
        if name == TOOL_LIST_TOOLS:
            return self._handle_list_tools(arguments), None
        return json.dumps({"error": f"Unknown search tool: {name}"}), None

    def _handle_find_tools(
        self, arguments: dict[str, Any], session_id: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        query = str(arguments.get("query", "")).strip()
        max_results = int(arguments.get("max_results") or DEFAULT_FIND_TOOLS_LIMIT)
        max_results = max(1, min(max_results, self._threshold))

        if not query:
            return (
                json.dumps(
                    {
                        "error": "query is required",
                        "hint": "Pass a short natural-language description.",
                    }
                ),
                None,
            )

        query_tokens = _tokenize(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for tool in self._catalogue:
            n = tool.get("name", "")
            if n in TOOL_SEARCH_INFRA_TOOL_NAMES or n in self._pinned_names:
                continue
            s = _score(query_tokens, tool)
            if s > 0:
                scored.append((s, tool))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:max_results]

        # Swap the exposure window — keep only the new matches plus
        # pinned. Prevents unbounded growth of the visible surface
        # across multiple find_tools calls.
        exposed = {tool.get("name", "") for _, tool in top}
        self._exposed[session_id] = exposed

        payload = {
            "matches": [
                {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                }
                for _, tool in top
            ],
            "_note": (
                "These tools are now invocable. Call the right one directly. "
                "Do not call find_tools again unless none of these fit."
            ),
        }
        if not top:
            payload["_note"] = (
                "No tools matched. Try a different query, or call list_tools "
                "to see all available tools."
            )
        # Tool result stays compact (name + short description only) so
        # the realtime model does not derail on a long return.
        result_str = json.dumps(payload)

        if not top:
            return result_str, None
        # Caller pushes this updated tool list via provider.reconfigure
        return result_str, self.visible_tools(session_id, base_tools=self._catalogue)

    def _handle_list_tools(self, arguments: dict[str, Any]) -> str:
        category = str(arguments.get("category", "")).strip()
        items: list[dict[str, str]] = []
        for tool in self._catalogue:
            n = tool.get("name", "")
            if n in TOOL_SEARCH_INFRA_TOOL_NAMES:
                continue
            if category and not n.startswith(category):
                continue
            items.append(
                {
                    "name": n,
                    "description": (tool.get("description") or "")[:200],
                }
            )
        # Bound the response so it never blows the realtime tool-result
        # threshold. 60 items × ~160 chars ≈ 10 KB worst case.
        truncated = len(items) > 60
        if truncated:
            items = items[:60]
        payload: dict[str, Any] = {"tools": items, "count": len(items)}
        if truncated:
            payload["truncated"] = True
            payload["_note"] = (
                "Result truncated to 60 entries. Use category= to filter, "
                "or prefer find_tools(query=...) for action."
            )
        return json.dumps(payload)
