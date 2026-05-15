"""Screen assistant support modules — extracted from screen_assistant_ai.py."""

from __future__ import annotations

from .banner import print_banner
from .handlers import ScreenToolDispatcher
from .observer import send_opening_greeting, setup_screen_vision
from .omniview import CLICK_RESULT_TOOL, OBSERVE_TOOL, OmniViewClient
from .playwright import setup_playwright_mcp
from .prompt import LANG_NAMES, build_system_prompt
from .providers import (
    build_vision_provider,
    build_voice_provider,
    get_voice_name,
    provider_config_for,
)
from .state import ScreenAssistantState
from .telemetry import CostTrackingTelemetry
from .tools import ACTION_TOOLS, LIST_SCREENS_TOOL, OPEN_APP_TOOL, SWITCH_SCREEN_TOOL
from .verify import assess_action_result, build_verify_question

__all__ = [
    "ACTION_TOOLS",
    "CLICK_RESULT_TOOL",
    "CostTrackingTelemetry",
    "LANG_NAMES",
    "LIST_SCREENS_TOOL",
    "OBSERVE_TOOL",
    "OPEN_APP_TOOL",
    "OmniViewClient",
    "SWITCH_SCREEN_TOOL",
    "ScreenAssistantState",
    "ScreenToolDispatcher",
    "assess_action_result",
    "build_system_prompt",
    "build_verify_question",
    "build_vision_provider",
    "build_voice_provider",
    "get_voice_name",
    "print_banner",
    "provider_config_for",
    "send_opening_greeting",
    "setup_playwright_mcp",
    "setup_screen_vision",
]
