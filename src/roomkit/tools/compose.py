"""Compose multiple ToolHandlers into a single first-match-wins handler."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.tools.base import Tool

logger = logging.getLogger("roomkit.tools.compose")

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


def _is_unknown_tool_error(result: str) -> bool:
    """Check if a tool handler result is an 'unknown tool' error."""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(parsed, dict):
        error = parsed.get("error", "")
        return isinstance(error, str) and error.lower().startswith("unknown tool")
    return False


def compose_tool_handlers(*handlers: ToolHandler) -> ToolHandler:
    """Chain multiple ToolHandlers so the first one that handles a tool wins.

    Each handler is tried in order. If a handler returns a JSON object with
    ``{"error": "Unknown tool: ..."}`` the next handler is tried. The last
    handler's result is always returned as-is (even if it's an unknown-tool
    error).

    Args:
        *handlers: Two or more ToolHandler callables.

    Returns:
        A single ToolHandler that dispatches to the first matching handler.

    Raises:
        ValueError: If fewer than two handlers are provided.
    """
    if len(handlers) < 2:
        raise ValueError("compose_tool_handlers requires at least 2 handlers")

    async def _composed(name: str, arguments: dict[str, Any]) -> str:
        for handler in handlers[:-1]:
            result = await handler(name, arguments)
            if not _is_unknown_tool_error(result):
                return result
            logger.debug("Handler %r did not handle tool %r, trying next", handler, name)
        # Last handler — return whatever it gives
        return await handlers[-1](name, arguments)

    return _composed


def extract_tools(
    tools: list[Tool | AITool | dict[str, Any]],
) -> tuple[list[AITool], ToolHandler | None]:
    """Split a mixed list of tool objects into definitions and a handler.

    Accepts any mix of:

    - :class:`Tool` objects (have ``.definition`` + ``.handler``)
    - :class:`AITool` instances (definition only, no handler)
    - Raw dicts (converted to :class:`AITool`, no handler)

    Returns:
        A tuple of ``(definitions, handler)`` where *handler* is a
        composed handler from all :class:`Tool` objects, or ``None``
        if no tool objects were provided.
    """
    from roomkit.tools.base import Tool

    definitions: list[AITool] = []
    handlers: list[ToolHandler] = []

    for tool in tools:
        if isinstance(tool, Tool):
            defn = tool.definition
            definitions.append(
                AITool(
                    name=defn["name"],
                    description=defn.get("description", ""),
                    parameters=defn.get("parameters", {}),
                ),
            )
            handlers.append(tool.handler)
        elif isinstance(tool, AITool):
            definitions.append(tool)
        elif isinstance(tool, dict):
            definitions.append(
                AITool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                ),
            )
        else:
            msg = f"Expected Tool, AITool, or dict, got {type(tool).__name__}"
            raise TypeError(msg)

    handler: ToolHandler | None = None
    if len(handlers) == 1:
        handler = handlers[0]
    elif len(handlers) >= 2:
        handler = compose_tool_handlers(*handlers)

    return definitions, handler
