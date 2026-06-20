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

The scoring + result rendering is shared with the text/HTTP agent loop
via :mod:`roomkit.channels._tool_search`; only the session state and the
``provider.reconfigure`` delivery are realtime-specific and live here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from roomkit.channels._tool_search import (
    normalize_max_results,
    render_find_payload,
    render_list_payload,
    search_catalogue,
)
from roomkit.channels._tool_search_constants import (
    FIND_TOOLS_SCHEMA,
    LIST_TOOLS_SCHEMA,
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
    TOOL_SEARCH_INFRA_TOOL_NAMES,
)

logger = logging.getLogger("roomkit.channels.realtime_voice")


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

        max_results = normalize_max_results(arguments.get("max_results"), self._threshold)
        exclude = self._pinned_names | TOOL_SEARCH_INFRA_TOOL_NAMES
        matches = search_catalogue(self._catalogue, query, max_results, exclude_names=exclude)

        # Swap the exposure window — keep only the new matches plus
        # pinned. Prevents unbounded growth of the visible surface
        # across multiple find_tools calls.
        self._exposed[session_id] = {tool.get("name", "") for tool in matches}

        result_str = render_find_payload(matches)
        if not matches:
            return result_str, None
        # Caller pushes this updated tool list via provider.reconfigure
        return result_str, self.visible_tools(session_id, base_tools=self._catalogue)

    def _handle_list_tools(self, arguments: dict[str, Any]) -> str:
        category = str(arguments.get("category", "")).strip()
        return render_list_payload(
            self._catalogue, category, exclude_names=TOOL_SEARCH_INFRA_TOOL_NAMES
        )
