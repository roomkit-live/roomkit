"""Compose multiple ToolHandlers into a single first-match-wins handler."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import UnservedToolCallError
from roomkit.providers.ai.base import AITool
from roomkit.tools.result import ToolResult, declined_answer

if TYPE_CHECKING:
    from roomkit.tools.base import Tool

logger = logging.getLogger("roomkit.tools.compose")

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


def compose_tool_handlers(*handlers: ToolHandler) -> ToolHandler:
    """Chain multiple ToolHandlers so the first one that handles a tool wins.

    Each handler is tried in order. A handler that declines the call, by
    raising :class:`~roomkit.core.exceptions.UnservedToolCallError` or by
    answering the ``{"error": "Unknown tool: ..."}`` envelope (RFC §21.4),
    hands it to the next one. The last handler's answer is the
    composition's, a decline included: the channel then reads the call as
    served by nothing.

    Args:
        *handlers: Two or more ToolHandler callables.

    Returns:
        A single ToolHandler that dispatches to the first matching handler.

    Raises:
        ValueError: If fewer than two handlers are provided.
    """
    if len(handlers) < 2:
        raise ValueError("compose_tool_handlers requires at least 2 handlers")

    async def _composed(name: str, arguments: dict[str, Any]) -> ToolResult:
        for handler in handlers[:-1]:
            try:
                return declined_answer(await handler(name, arguments), name)
            except UnservedToolCallError:
                logger.debug("Handler %r did not handle tool %r, trying next", handler, name)
        # The last handler's answer is the composition's, a decline included.
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
            definitions.append(_definition(tool.definition))
            handlers.append(tool.handler)
        elif isinstance(tool, AITool):
            definitions.append(tool)
        elif isinstance(tool, dict):
            definitions.append(_definition(tool))
        else:
            msg = f"Expected Tool, AITool, or dict, got {type(tool).__name__}"
            raise TypeError(msg)

    handler: ToolHandler | None = None
    if len(handlers) == 1:
        handler = handlers[0]
    elif len(handlers) >= 2:
        handler = compose_tool_handlers(*handlers)

    return definitions, handler


def _definition(schema: dict[str, Any]) -> AITool:
    """The definition a tool schema describes, its search tags and whether it
    changes state included."""
    return AITool(
        name=schema["name"],
        description=schema.get("description", ""),
        parameters=schema.get("parameters", {}),
        tags=list(schema.get("tags") or []),
        changes_state=schema.get("changes_state", False),
    )
