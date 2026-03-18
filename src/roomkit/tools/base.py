"""Tool protocol for unified tool registration across all channels.

A Tool bundles a definition (schema) with a handler (implementation),
so channels can accept tool objects directly instead of requiring
separate ``tools`` and ``tool_handler`` parameters.

Example::

    from roomkit import DescribeWebcamTool, AIChannel, compose_tool_handlers

    webcam = DescribeWebcamTool(vision, device=0)

    # Pass tool objects directly — definitions and handler are extracted
    ai = AIChannel("ai", provider=provider, tools=[webcam])
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Protocol for objects that combine a tool definition with a handler.

    Any object with a ``definition`` property (returning the tool schema
    dict) and an async ``handler`` method (matching the unified
    ``ToolHandler`` signature) satisfies this protocol.

    Existing tool classes like :class:`DescribeScreenTool`,
    :class:`DescribeWebcamTool`, and :class:`ListWebcamsTool` already
    implement this protocol — no changes needed.
    """

    @property
    def definition(self) -> dict[str, Any]:
        """Tool schema dict with ``name``, ``description``, ``parameters``."""
        ...

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute the tool and return a result string."""
        ...
