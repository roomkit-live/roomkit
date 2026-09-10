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
TOOL_CALL_TOOL = "call_tool"

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
    "IMPORTANT — your visible tools are only a SMALL SUBSET of what you can do. "
    "Many more tools (sending messages, playing music, web search, looking up "
    "data, calendars, files, …) exist but are hidden until you search for them. "
    "Two discovery tools are always available:\n"
    "- `find_tools(query)` — search your FULL tool catalogue by natural-language "
    'task (e.g. "play music on spotify", "look up a contact by phone"). It '
    "returns the best matches WITH their parameter schemas, ready to call.\n"
    "- `list_tools(category=None)` — list every tool you can access (overview "
    "only; prefer `find_tools` to act).\n"
    "HARD RULE: never tell the user you cannot do something, that you lack a "
    "tool, or that a task is outside your capabilities — UNTIL you have called "
    "`find_tools` for that task and it returned no usable match. If a request "
    "needs an action you don't see a tool for, your FIRST step is "
    "`find_tools(query=<the task>)`, then call the matched tool directly. Do NOT "
    "narrate that you are searching — just call the tool and continue.\n"
    "ALWAYS phrase your `find_tools` query in ENGLISH, even when the conversation "
    "is in another language — the catalogue is indexed by English keywords, so an "
    'English query matches reliably (e.g. for « liste mes fichiers » search "list '
    'files", for « envoyer un message » search "send a message"). Translate the '
    "user's intent to English keywords first, then search."
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
                    '"crm_" or "admin_"). Empty/omitted returns all.'
                ),
            },
        },
    },
}

# Fixed declarations deliver schemas as tool results; execution still uses the
# channel's ordinary tool handler and pre-execution gates.
FIXED_TOOL_SEARCH_PREAMBLE = (
    "Your visible tools are a small subset of your authorized tool catalogue. "
    "Use find_tools(query) to discover tools for a task. ALWAYS search with "
    "ENGLISH keywords, even when speaking another language. "
    "Use list_tools(name=<exact tool name>) to read one tool's COMPLETE "
    "description and argument schema, then call_tool(name=<exact tool name>, "
    "arguments_json=<JSON object encoded as a string>) to execute it. "
    "The name is the TOOL name, not an action: an action such as list belongs "
    "inside arguments_json when that tool's schema declares it. "
    'For example, the calendar tool may accept {"action":"list"} as its arguments. '
    "Tools already declared directly can be called directly. "
    "list_tools(category=...) gives a compact inventory, not argument schemas. "
    "Never claim a capability is unavailable before searching for it. "
    "Search and schema lookup do not execute a business operation; continue "
    "with call_tool and report its actual result or refusal."
)

CALL_TOOL_SCHEMA: dict[str, Any] = {
    "name": TOOL_CALL_TOOL,
    "description": (
        "Execute an authorized tool using its exact name and a JSON object "
        "encoded as a string. Read its schema with list_tools(name=...) first. "
        "Actions belong inside the arguments, not in the tool name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact tool name."},
            "arguments_json": {
                "type": "string",
                "description": "JSON object matching the tool's argument schema, e.g. {}.",
            },
        },
        "required": ["name", "arguments_json"],
        "additionalProperties": False,
    },
}
