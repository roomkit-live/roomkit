"""Screen assistant support modules — extracted from screen_assistant_ia.py."""

from __future__ import annotations

from .playwright import setup_playwright_mcp
from .prompt import LANG_NAMES, build_system_prompt
from .providers import build_vision_provider, build_voice_provider, get_voice_name
from .telemetry import CostTrackingTelemetry
from .tools import ACTION_TOOLS, LIST_SCREENS_TOOL, OPEN_APP_TOOL, SWITCH_SCREEN_TOOL
from .verify import assess_action_result, build_verify_question

__all__ = [
    "ACTION_TOOLS",
    "CostTrackingTelemetry",
    "LANG_NAMES",
    "LIST_SCREENS_TOOL",
    "OPEN_APP_TOOL",
    "SWITCH_SCREEN_TOOL",
    "assess_action_result",
    "build_system_prompt",
    "build_verify_question",
    "build_vision_provider",
    "build_voice_provider",
    "get_voice_name",
    "setup_playwright_mcp",
]
