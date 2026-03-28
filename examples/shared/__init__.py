"""Reusable helpers for RoomKit examples."""

from __future__ import annotations

from .audio import (
    build_aec,
    build_debug_taps,
    build_denoiser,
    build_pipeline,
    build_turn_detector,
    build_vad,
)
from .console import setup_console
from .env import auto_select_provider, os_info, require_env
from .hooks import log_tool_call
from .lifecycle import run_until_stopped
from .log import setup_logging

__all__ = [
    "auto_select_provider",
    "build_aec",
    "build_debug_taps",
    "build_denoiser",
    "build_pipeline",
    "build_turn_detector",
    "build_vad",
    "log_tool_call",
    "os_info",
    "require_env",
    "run_until_stopped",
    "setup_console",
    "setup_logging",
]
