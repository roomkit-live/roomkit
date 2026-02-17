"""Compose multiple ToolHandlers into a single first-match-wins handler."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

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
