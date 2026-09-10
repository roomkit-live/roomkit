"""Tool Search delivery for realtime sessions.

The shared scorer reads the session's authorized catalogue. Reconfigurable
providers receive a rolling window of native declarations. Providers with
fixed declarations read complete schemas through list_tools(name=...) and
use call_tool as a transport into the channel's ordinary execution path.
Searches never reconnect a fixed-declaration session or expand its rights.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from roomkit.channels._tool_search import (
    normalize_max_results,
    related_family_tools,
    render_find_payload,
    render_list_payload,
    search_catalogue,
)
from roomkit.channels._tool_search_constants import (
    CALL_TOOL_SCHEMA,
    FIND_TOOLS_SCHEMA,
    FIXED_TOOL_SEARCH_PREAMBLE,
    LIST_TOOLS_SCHEMA,
    TOOL_CALL_TOOL,
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
    TOOL_SEARCH_INFRA_TOOL_NAMES,
    TOOL_SEARCH_PREAMBLE,
)
from roomkit.tools.validation import validate_tool_arguments

logger = logging.getLogger("roomkit.channels.realtime_voice")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


class RealtimeToolSearchSupport:
    """Session catalogues and schema delivery for tool-heavy realtime channels."""

    def __init__(
        self,
        catalogue: list[dict[str, Any]],
        *,
        pinned: list[str] | None = None,
        threshold: int = 20,
        reconfigure_capable: bool = True,
    ) -> None:
        self._catalogue: list[dict[str, Any]] = list(catalogue)
        self._pinned_names: set[str] = set(pinned or [])
        self._threshold = threshold
        self.uses_call_tool = not reconfigure_capable
        self._validate_catalogue(catalogue)
        # session_id -> set of tool names currently exposed by find_tools
        self._exposed: dict[str, set[str]] = {}
        self._session_catalogues: dict[str, list[dict[str, Any]]] = {}

    def _validate_catalogue(self, catalogue: list[dict[str, Any]]) -> None:
        if self.uses_call_tool and any(t.get("name") == TOOL_CALL_TOOL for t in catalogue):
            raise ValueError("call_tool is reserved by fixed-declaration Tool Search")

    # -- Tool definitions (channel injects these into the live tool list) --

    def search_tool_dicts(self) -> list[dict[str, Any]]:
        if not self.uses_call_tool:
            return [FIND_TOOLS_SCHEMA, LIST_TOOLS_SCHEMA]
        find, inventory = deepcopy(FIND_TOOLS_SCHEMA), deepcopy(LIST_TOOLS_SCHEMA)
        find["description"] = (
            "Search the authorized catalogue by English task keywords. Returns matching "
            "tool names; use list_tools(name=...) for a complete schema, then call_tool."
        )
        inventory["description"] = (
            "Read one tool's COMPLETE description and argument schema by exact name. "
            "Without name, list a compact inventory optionally filtered by category."
        )
        inventory["parameters"]["properties"]["name"] = {
            "type": "string",
            "description": "Exact tool name whose complete schema is needed.",
        }
        return [find, inventory, deepcopy(CALL_TOOL_SCHEMA)]

    @property
    def preamble(self) -> str:
        return FIXED_TOOL_SEARCH_PREAMBLE if self.uses_call_tool else TOOL_SEARCH_PREAMBLE

    def is_search_tool(self, name: str) -> bool:
        return name in TOOL_SEARCH_INFRA_TOOL_NAMES or (
            self.uses_call_tool and name == TOOL_CALL_TOOL
        )

    def unwrap_call(
        self, arguments: dict[str, Any], session_id: str
    ) -> tuple[str, dict[str, Any], str | None]:
        """Decode transport only; the channel owns validation and execution."""
        error = validate_tool_arguments(CALL_TOOL_SCHEMA["parameters"], arguments)
        if error:
            return TOOL_CALL_TOOL, arguments, f"Invalid call_tool arguments: {error}"
        name = arguments["name"]
        try:
            decoded = json.loads(arguments["arguments_json"], parse_constant=_reject_json_constant)
        except (ValueError, RecursionError):
            return name, {}, "Invalid arguments_json: expected a JSON object encoded as a string"
        if not isinstance(decoded, dict):
            return name, {}, "Invalid arguments_json: expected a JSON object"
        # Only the session's granted catalogue is callable. An absent catalogue
        # must not activate the native channel's dynamic/hook-only fallback.
        catalogue = self._session_catalogues.get(session_id, [])
        if self.is_search_tool(name) or not any(t.get("name") == name for t in catalogue):
            return name, decoded, f"Tool '{name}' is unavailable in this session"
        return name, decoded, None

    # -- Per-session lifecycle --

    def init_session(self, session_id: str, catalogue: list[dict[str, Any]] | None = None) -> None:
        effective = self._catalogue if catalogue is None else catalogue
        self._validate_catalogue(effective)
        self._exposed[session_id] = set()
        self._session_catalogues[session_id] = list(effective)

    def cleanup_session(self, session_id: str) -> None:
        self._exposed.pop(session_id, None)
        self._session_catalogues.pop(session_id, None)

    # -- Visibility (replaces the channel's full tool list) --

    def visible_tools(
        self, session_id: str, base_tools: list[dict[str, Any]], *, reset_exposure: bool = False
    ) -> list[dict[str, Any]]:
        """Return the slice of the catalogue that should be live right now.

        Always includes search infra + pinned + currently-exposed matches.
        ``base_tools`` is the original list the channel was constructed
        with; we use it only to preserve ordering for deterministic output.
        """
        exposed = (
            set()
            if reset_exposure or self.uses_call_tool
            else self._exposed.get(session_id, set())
        )
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
            return self._handle_list_tools(arguments, session_id), None
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
        catalogue = self._session_catalogues.get(
            session_id, [] if self.uses_call_tool else self._catalogue
        )
        matches = search_catalogue(catalogue, query, max_results, exclude_names=exclude)

        # Swap the exposure window — keep only the new matches plus
        # pinned. Prevents unbounded growth of the visible surface
        # across multiple find_tools calls.
        self._exposed[session_id] = {tool.get("name", "") for tool in matches}

        result_str = render_find_payload(
            matches,
            related=related_family_tools(catalogue, matches),
            call_tool=self.uses_call_tool,
        )
        if not matches or self.uses_call_tool:
            return result_str, None
        # Caller pushes this updated tool list via provider.reconfigure
        return result_str, self.visible_tools(session_id, base_tools=catalogue)

    def _handle_list_tools(self, arguments: dict[str, Any], session_id: str) -> str:
        if self.uses_call_tool and "name" in arguments:
            name = arguments["name"]
            for tool in self._session_catalogues.get(session_id, []):
                if tool.get("name") == name and not self.is_search_tool(name):
                    return json.dumps(
                        {"tool": tool, "_note": "Execute this tool using call_tool."}
                    )
            return json.dumps({"error": f"Tool '{name}' is unavailable in this session"})
        category = str(arguments.get("category", "")).strip()
        return render_list_payload(
            self._session_catalogues.get(
                session_id, [] if self.uses_call_tool else self._catalogue
            ),
            category,
            exclude_names=TOOL_SEARCH_INFRA_TOOL_NAMES,
        )
