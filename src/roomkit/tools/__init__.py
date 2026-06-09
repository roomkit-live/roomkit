"""Tool utilities for bridging external tool systems into RoomKit."""

from __future__ import annotations

from roomkit.tools.base import Tool
from roomkit.tools.compose import compose_tool_handlers, extract_tools
from roomkit.tools.context import current_tool_allowed_names, current_tool_room_id
from roomkit.tools.external import ExternalToolHandler, PolicyExternalToolHandler, ToolDecision
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler
from roomkit.tools.policy import RoleOverride, ToolPolicy

__all__ = [
    "ExternalToolHandler",
    "HumanInputHandler",
    "HumanInputToolHandler",
    "MCPToolProvider",
    "PolicyExternalToolHandler",
    "RoleOverride",
    "Tool",
    "ToolDecision",
    "ToolPolicy",
    "compose_tool_handlers",
    "current_tool_allowed_names",
    "current_tool_room_id",
    "extract_tools",
]


def __getattr__(name: str) -> object:
    if name == "MCPToolProvider":
        from roomkit.tools.mcp import MCPToolProvider

        return MCPToolProvider
    raise AttributeError(f"module 'roomkit.tools' has no attribute {name}")
