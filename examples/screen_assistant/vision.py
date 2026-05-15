"""Vision helpers — cost-tracked analysis and screen-change detection."""

from __future__ import annotations

import re

from roomkit.video.vision.screen_tool import capture_screen_frame

from .state import ScreenAssistantState

_APPS = (
    "chrome",
    "safari",
    "firefox",
    "iterm",
    "terminal",
    "finder",
    "code",
    "vscode",
    "slack",
    "discord",
    "teams",
    "outlook",
    "github",
    "google",
    "roomkit",
)
_URL_RE = re.compile(r"[\w-]+\.[\w.-]+")


def extract_key_terms(description: str) -> set[str]:
    """Pull recognizable app names and URLs out of a vision description."""
    lower = description.lower()
    terms = {app for app in _APPS if app in lower}
    terms.update(url.lower() for url in _URL_RE.findall(description))
    return terms


async def analyze_with_cost(
    state: ScreenAssistantState,
    query: str,
    *,
    monitor: int | None = None,
) -> str:
    """Capture the current screen, analyze it, and accumulate token usage."""
    idx = state.monitor if monitor is None else monitor
    frame = capture_screen_frame(idx)
    if frame is None:
        return "No screen frame available."
    result = await state.vision.analyze_frame(frame, prompt=query)
    totals = state.cost_telemetry.totals
    totals["vision_calls"] += 1
    usage = result.metadata.get("usage", {})
    totals["vision_prompt_tokens"] += usage.get("prompt_tokens", 0)
    totals["vision_completion_tokens"] += usage.get("completion_tokens", 0)
    return result.description or "Could not analyze the screen."


def build_change_context(previous: str, current: str) -> tuple[str | None, bool]:
    """Decide whether a vision update is worth injecting into the voice session.

    Returns ``(context, significant)`` where ``context`` is ``None`` when the
    change is too small to inject.
    """
    if not current:
        return None, False

    prev_terms = extract_key_terms(previous) if previous else set()
    curr_terms = extract_key_terms(current)
    new_terms = curr_terms - prev_terms
    significant = bool(new_terms)

    if previous and not significant and previous[:80] == current[:80]:
        return None, False

    if significant:
        return f"[Screen changed — new: {', '.join(sorted(new_terms))}] {current}", True
    return f"[Screen changed] {current}", False
