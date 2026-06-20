"""Shared constants for Tool Search integration on realtime channels.

Google explicitly recommends keeping the active tool set to 10–20 entries
on Gemini Live (https://ai.google.dev/gemini-api/docs/live-api/tools).
Above that range, function-calling reliability degrades — the model
narrates instead of invoking, and tends to pick the cheapest tool over
the right one. The Tool Search pattern keeps the visible surface small
by exposing two discovery tools (find_tools, list_tools) and revealing
matched tools on demand via ``provider.reconfigure``.
"""

from __future__ import annotations

from typing import Any

TOOL_FIND_TOOLS = "find_tools"
TOOL_LIST_TOOLS = "list_tools"

TOOL_SEARCH_INFRA_TOOL_NAMES: frozenset[str] = frozenset({TOOL_FIND_TOOLS, TOOL_LIST_TOOLS})

# Default ceiling — matches Google's "10–20 active tools" guidance.
# Used by the realtime channel, and by the text channel as the FALLBACK
# tool count when the model's context window is unknown (custom / local
# model ids absent from the provider catalog).
DEFAULT_TOOL_SEARCH_THRESHOLD = 20

# Text-channel auto-activation threshold as a percentage of the model's
# context window: defer the catalogue when the deferrable (non-pinned)
# tools would cost more than this share of the window. Self-tunes to model
# size — a large window is a no-op, a small one defers early. Mirrors the
# Hermes ``threshold_pct`` default.
DEFAULT_TOOL_SEARCH_THRESHOLD_PCT = 10.0

# Default number of matches returned by find_tools. Small enough to
# stay well inside the ceiling even when the model immediately invokes
# one of the matches, leaving headroom for further searches.
DEFAULT_FIND_TOOLS_LIMIT = 5

TOOL_SEARCH_PREAMBLE = (
    "You have a large tool surface. Two discovery tools are always "
    "available:\n"
    "- `find_tools(query)` — search by natural-language query (e.g. "
    '"look up contact by phone"). Returns up to a handful of '
    "matching tools with their full schemas. The matches become "
    "directly invocable for the rest of the session.\n"
    "- `list_tools(category=None)` — list every tool you have access "
    "to with name + short description, optionally filtered by "
    "category prefix. Use this only when you need an overview; "
    "prefer `find_tools` for action.\n"
    "When the caller asks for something you cannot do with the tools "
    "currently visible, call `find_tools` first, then invoke the "
    "matched tool. Do NOT narrate that you are searching — call the "
    "tool and continue."
)

FIND_TOOLS_SCHEMA: dict[str, Any] = {
    "name": TOOL_FIND_TOOLS,
    "description": (
        "Search the agent's full tool catalogue by natural-language query. "
        "Returns up to N best matches with their schemas, which become "
        "directly invocable for the rest of the session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of what you need to do "
                    '(e.g. "look up a contact by phone number", "send '
                    'an SMS").'
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum number of tools to return. Defaults to 5. "
                    "Keep small — the goal is a focused result, not a list."
                ),
            },
        },
        "required": ["query"],
    },
}

LIST_TOOLS_SCHEMA: dict[str, Any] = {
    "name": TOOL_LIST_TOOLS,
    "description": (
        "List every tool the agent has access to (name + short "
        "description), optionally filtered by a name-prefix category. "
        "Use sparingly — prefer find_tools for action."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "Optional prefix to filter tools by name (e.g. "
                    '"TnS_" or "luge_admin_"). Empty/omitted returns all.'
                ),
            },
        },
    },
}
