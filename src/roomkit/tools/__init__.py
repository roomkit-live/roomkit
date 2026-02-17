"""Tool utilities for bridging external tool systems into RoomKit."""

from __future__ import annotations

from roomkit.tools.compose import compose_tool_handlers

__all__ = [
    "MCPToolProvider",
    "compose_tool_handlers",
]


def __getattr__(name: str) -> object:
    if name == "MCPToolProvider":
        from roomkit.tools.mcp import MCPToolProvider

        return MCPToolProvider
    raise AttributeError(f"module 'roomkit.tools' has no attribute {name}")
