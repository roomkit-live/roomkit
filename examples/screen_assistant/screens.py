"""Monitor enumeration and switching helpers."""

from __future__ import annotations

import logging

import mss

from .state import ScreenAssistantState
from .vision import analyze_with_cost

logger = logging.getLogger("screen_assistant.screens")

_DESCRIBE_PROMPT = (
    "Describe what is on this screen in one sentence. "
    "Include the main application, visible windows, and dock if present."
)


async def list_screens(state: ScreenAssistantState) -> str:
    """Enumerate monitors and describe each via the vision provider."""
    with mss.mss() as sct:
        monitors = sct.monitors
    if len(monitors) <= 1:
        return "Only 1 monitor detected (the combined virtual screen)."

    lines = [f"{len(monitors) - 1} monitor(s) detected:\n"]
    for idx in range(1, len(monitors)):
        mon = monitors[idx]
        size = f"{mon['width']}x{mon['height']}"
        active = " (ACTIVE)" if idx == state.monitor else ""
        try:
            desc = await analyze_with_cost(state, _DESCRIBE_PROMPT, monitor=idx)
        except Exception:
            desc = "(could not analyze)"
        lines.append(f"  [{idx}] {size}{active}: {desc}")
    return "\n".join(lines)


def switch_screen(state: ScreenAssistantState, new_monitor: int) -> str:
    """Switch the active monitor across all monitor-bound tools."""
    with mss.mss() as sct:
        count = len(sct.monitors)
    if new_monitor < 0 or new_monitor >= count:
        return f"Invalid monitor {new_monitor}. Available: 0-{count - 1}."

    state.switch_monitor(new_monitor)
    logger.info("Switched to monitor %d", new_monitor)
    return f"Switched to monitor {new_monitor}. All tools now target this screen."
