"""Reusable helpers for RoomKit examples."""

from __future__ import annotations

from .args import existing_directory, non_negative_int
from .audio import (
    build_aec,
    build_debug_taps,
    build_denoiser,
    build_pipeline,
    build_turn_detector,
    build_vad,
)
from .console import console_enabled, setup_console
from .env import auto_select_provider, env_bool, os_info, require_env
from .hooks import log_tool_call
from .lifecycle import run_until_stopped
from .log import setup_logging
from .tools import WebSearchTool

__all__ = [
    "WebSearchTool",
    "auto_select_provider",
    "build_aec",
    "build_debug_taps",
    "build_denoiser",
    "build_pipeline",
    "build_turn_detector",
    "build_vad",
    "console_enabled",
    "env_bool",
    "existing_directory",
    "log_tool_call",
    "non_negative_int",
    "os_info",
    "require_env",
    "run_until_stopped",
    "setup_console",
    "setup_logging",
]
