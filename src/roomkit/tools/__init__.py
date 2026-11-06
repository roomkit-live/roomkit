"""Tool utilities for bridging external tool systems into RoomKit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.tools.base import Tool
from roomkit.tools.compose import compose_tool_handlers, extract_tools
from roomkit.tools.context import (
    ToolCallContext,
    TurnFootprint,
    current_response_metadata,
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_call,
    current_tool_chain_depth,
    current_tool_requester,
    current_tool_room,
    current_tool_room_id,
    current_turn_footprint,
    tool_turn_context,
    turn_report_claim,
)
from roomkit.tools.external import ExternalToolHandler, PolicyExternalToolHandler, ToolDecision
from roomkit.tools.fence import FENCED_TAGS, fence, named_blocks
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler
from roomkit.tools.policy import RoleOverride, ToolPolicy

if TYPE_CHECKING:
    from roomkit.tools.mcp import MCPToolProvider
else:
    __getattr__ = lazy_exports(__name__, {"MCPToolProvider": "roomkit.tools.mcp"})

__all__ = [
    "FENCED_TAGS",
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
    "current_tool_chain_depth",
    "current_tool_allowed_names",
    "current_tool_room",
    "current_tool_room_id",
    "current_turn_footprint",
    "extract_tools",
    "fence",
    "named_blocks",
    "tool_turn_context",
    "turn_report_claim",
]
