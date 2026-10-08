"""Tool utilities for bridging external tool systems into RoomKit."""

from __future__ import annotations

from roomkit.tools.base import Tool
from roomkit.tools.compose import compose_tool_handlers, extract_tools
from roomkit.tools.context import (
    ToolCallContext,
    TurnFootprint,
    current_response_metadata,
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_call,
    current_tool_requester,
    current_tool_room,
    current_tool_room_id,
    current_turn_footprint,
    tool_turn_context,
)
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
    "ToolCallContext",
    "ToolDecision",
    "ToolPolicy",
    "TurnFootprint",
    "compose_tool_handlers",
    "current_response_metadata",
    "current_tool_actor_id",
    "current_tool_requester",
    "current_tool_call",
    "current_tool_allowed_names",
    "current_tool_room",
    "current_tool_room_id",
    "current_turn_footprint",
    "extract_tools",
    "tool_turn_context",
]


def __getattr__(name: str) -> object:
    if name == "MCPToolProvider":
        from roomkit.tools.mcp import MCPToolProvider

        return MCPToolProvider
    raise AttributeError(f"module 'roomkit.tools' has no attribute {name}")
