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
from collections.abc import Callable, Iterable
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
from roomkit.tools.result import unknown_tool_error
from roomkit.tools.validation import validate_tool_arguments

logger = logging.getLogger("roomkit.channels.realtime_voice")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


class RealtimeToolSearchSupport:
    """Session catalogues and schema delivery for tool-heavy realtime channels.

    Whether a session hides its catalogue is decided per session, on what that
    session declares: a room's active agent, a session's own tools and what
    orchestration set up for the room all change it after the channel was
    built. Without *auto*, every session hides it.
    """

    def __init__(
        self,
        catalogue: list[dict[str, Any]],
        *,
        pinned: list[str] | None = None,
        threshold: int = 20,
        reconfigure_capable: bool = True,
        reachable: Callable[[str, str], bool] | None = None,
        never_deferred: Callable[[str], Iterable[str]] | None = None,
        listed: Callable[[str], list[dict[str, Any]]] | None = None,
        auto: bool = False,
    ) -> None:
        self._catalogue: list[dict[str, Any]] = list(catalogue)
        # Hide a session's catalogue only when it is larger than the threshold.
        self._auto = auto
        # session_id -> whether the session's catalogue is hidden behind search.
        self._active: dict[str, bool] = {}
        # (tool name, session id) -> whether the session may call it. Search
        # results and listings name nothing else (RFC §21.1): a match the
        # model can never call is a false promise and discloses the gate.
        self._reachable = reachable
        # session id -> the tools never hidden in its room: the channel's own
        # and what orchestration set up (RFC §21.1). Declared already, so
        # ``find_tools`` never names them.
        self._never_deferred = never_deferred or (lambda _session_id: ())
        # session id -> every tool the session can call, declared or hidden,
        # for ``list_tools`` (RFC §21.1); its searchable catalogue without it.
        self._listed = listed
        self._pinned_names: set[str] = set(pinned or [])
        self._threshold = threshold
        self.uses_call_tool = not reconfigure_capable
        self._validate_catalogue(catalogue)
        # session_id -> set of tool names currently exposed by find_tools
        self._exposed: dict[str, set[str]] = {}
        # session_id -> the model response whose search last swapped the window
        self._exposed_response: dict[str, int] = {}
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

    @property
    def tool_names(self) -> frozenset[str]:
        """The tools Tool Search serves itself on this channel."""
        if self.uses_call_tool:
            return TOOL_SEARCH_INFRA_TOOL_NAMES | {TOOL_CALL_TOOL}
        return TOOL_SEARCH_INFRA_TOOL_NAMES

    def is_search_tool(self, name: str) -> bool:
        return name in self.tool_names

    def unwrap_call(
        self, arguments: dict[str, Any], session_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str] | None]:
        """Decode transport only; the channel owns validation and execution.
        The third value is what the model reads when the transport cannot
        carry the call, else ``None``."""
        error = validate_tool_arguments(CALL_TOOL_SCHEMA["parameters"], arguments)
        if error:
            return TOOL_CALL_TOOL, arguments, {"error": f"Invalid call_tool arguments: {error}"}
        name = arguments["name"]
        try:
            decoded = json.loads(arguments["arguments_json"], parse_constant=_reject_json_constant)
        except (ValueError, RecursionError):
            invalid = "Invalid arguments_json: expected a JSON object encoded as a string"
            return name, {}, {"error": invalid}
        if not isinstance(decoded, dict):
            return name, {}, {"error": "Invalid arguments_json: expected a JSON object"}
        # Only what the session can call is callable. An absent catalogue
        # must not activate the native channel's dynamic/hook-only fallback.
        callable_ = self._callable(session_id)
        if self.is_search_tool(name) or not any(t.get("name") == name for t in callable_):
            # A name no tool carries, worded as every door words it (RFC §21.1).
            return name, decoded, unknown_tool_error(name, searching=True)
        return name, decoded, None

    def _callable(self, session_id: str) -> list[dict[str, Any]]:
        """Every tool the session can call, declared or hidden: what
        ``list_tools`` lists, names and ``call_tool`` reaches (RFC §21.1)."""
        if session_id not in self._session_catalogues:
            return [] if self.uses_call_tool else self._catalogue
        if self._listed is not None:
            # A provider's native tool has no name to list or call.
            return [t for t in self._listed(session_id) if t.get("name")]
        return self._session_catalogues[session_id]

    def _searchable(
        self, session_id: str, catalogue: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The part of *catalogue* the session may call, for search and listing."""
        reachable = self._reachable
        if reachable is None:
            return catalogue
        return [t for t in catalogue if reachable(str(t.get("name", "")), session_id)]

    # -- Per-session lifecycle --

    def init_session(self, session_id: str, catalogue: list[dict[str, Any]] | None = None) -> None:
        effective = self._catalogue if catalogue is None else catalogue
        self._validate_catalogue(effective)
        self._exposed[session_id] = set()
        self._exposed_response.pop(session_id, None)
        self._session_catalogues[session_id] = list(effective)
        self._active[session_id] = self.activates(session_id, effective)

    def activates(self, session_id: str, catalogue: list[dict[str, Any]]) -> bool:
        """Whether *catalogue*, declared by the session, is hidden behind search."""
        return not self._auto or self._deferrable_count(session_id, catalogue) > self._threshold

    def _deferrable_count(self, session_id: str, catalogue: list[dict[str, Any]]) -> int:
        """How many of *catalogue*'s tools search could hide: what is never
        deferred does not count toward the size that decides it (RFC §21.1)."""
        never = set(self._never_deferred(session_id))
        # A provider's native tool has no name: search can hide none.
        return sum(1 for tool in catalogue if (name := tool.get("name")) and name not in never)

    def active(self, session_id: str) -> bool:
        """Whether the session's catalogue is hidden behind the search tools."""
        return self._active.get(session_id, not self._auto)

    def cleanup_session(self, session_id: str) -> None:
        self._exposed.pop(session_id, None)
        self._exposed_response.pop(session_id, None)
        self._session_catalogues.pop(session_id, None)
        self._active.pop(session_id, None)

    # -- Visibility (replaces the channel's full tool list) --

    def visible_tools(
        self,
        session_id: str,
        base_tools: list[dict[str, Any]],
        *,
        reset_exposure: bool = False,
        keep: Iterable[str] = (),
        active: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return the slice of the catalogue that should be live right now.

        Always includes search infra + pinned + *keep* (what is never deferred
        in the session's room) + currently-exposed matches. ``base_tools`` is
        the session's catalogue; we use it only to preserve ordering for
        deterministic output. A session whose catalogue is small enough sees
        it whole, without the search tools. *active* decides it for a
        catalogue the session is about to declare.
        """
        if not (self.active(session_id) if active is None else active):
            return list(base_tools)
        exposed = (
            set()
            if reset_exposure or self.uses_call_tool
            else self._exposed.get(session_id, set())
        )
        keep = self._pinned_names | set(keep) | exposed | TOOL_SEARCH_INFRA_TOOL_NAMES
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
            if not n:
                # A provider's native tool: no name to search or hide it by.
                result.append(tool)
            elif n in keep and n not in seen:
                result.append(tool)
                seen.add(n)
        return result

    def expose(self, session_id: str, names: Iterable[str], response: int | None = None) -> bool:
        """Reveal *names* as ``find_tools`` reveals its matches: the exposure
        window swapped, what is declared anyway left out. Whether anything
        was revealed, which the session's declaration then has to show.

        Searches the model ran side by side in one *response* each told it
        their matches are declared, so the second adds to the window the
        first swapped (RMK-606), as the text loop's searches of one round do.
        A search of a later response swaps it, and so does one whose response
        is not known (``None``)."""
        if self.uses_call_tool or not self.active(session_id):
            return False
        exclude = self._never_revealed(session_id)
        revealed = {name for name in names if name not in exclude}
        if not revealed:
            return False
        if response is not None and self._exposed_response.get(session_id) == response:
            self._exposed[session_id] = self._exposed.get(session_id, set()) | revealed
            return True
        self._exposed[session_id] = revealed
        if response is None:
            self._exposed_response.pop(session_id, None)
        else:
            self._exposed_response[session_id] = response
        return True

    def _never_revealed(self, session_id: str) -> set[str]:
        """What a reveal never names: what is declared anyway (pinned, never
        deferred) and the search tools themselves."""
        return (
            self._pinned_names
            | set(self._never_deferred(session_id))
            | TOOL_SEARCH_INFRA_TOOL_NAMES
        )

    # -- Tool dispatch --

    async def handle_tool_call(
        self, name: str, arguments: dict[str, Any], session_id: str
    ) -> tuple[str, list[str]]:
        """Answer find_tools / list_tools, revealing nothing.

        Returns ``(json_result, matched_names)``: the caller reveals the
        names (:meth:`expose`) once the call is served and its result went
        out, so a call ON_TOOL_CALL blocks reveals nothing (RFC §6.4).
        """
        if name == TOOL_FIND_TOOLS:
            return self._handle_find_tools(arguments, session_id)
        if name == TOOL_LIST_TOOLS:
            return self._handle_list_tools(arguments, session_id), []
        return json.dumps({"error": f"Unknown search tool: {name}"}), []

    def _handle_find_tools(
        self, arguments: dict[str, Any], session_id: str
    ) -> tuple[str, list[str]]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return (
                json.dumps(
                    {
                        "error": "query is required",
                        "hint": "Pass a short natural-language description.",
                    }
                ),
                [],
            )

        max_results = normalize_max_results(arguments.get("max_results"), self._threshold)
        exclude = self._never_revealed(session_id)
        catalogue = self._searchable(
            session_id,
            self._session_catalogues.get(
                session_id, [] if self.uses_call_tool else self._catalogue
            ),
        )
        matches = search_catalogue(catalogue, query, max_results, exclude_names=exclude)
        result_str = render_find_payload(
            matches,
            related=related_family_tools(catalogue, matches, exclude_names=exclude),
            call_tool=self.uses_call_tool,
        )
        return result_str, [name for tool in matches if (name := tool.get("name"))]

    def _handle_list_tools(self, arguments: dict[str, Any], session_id: str) -> str:
        if self.uses_call_tool and "name" in arguments:
            name = arguments["name"]
            for tool in self._searchable(session_id, self._callable(session_id)):
                if tool.get("name") == name and not self.is_search_tool(name):
                    return json.dumps(
                        {"tool": tool, "_note": "Execute this tool using call_tool."}
                    )
            return json.dumps(unknown_tool_error(name, searching=True))
        category = str(arguments.get("category", "")).strip()
        return render_list_payload(
            self._searchable(session_id, self._callable(session_id)),
            category,
            exclude_names=self.tool_names,
        )
