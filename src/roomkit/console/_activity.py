"""Who is working right now — the console's live agent activity registry.

The pinned shell shows a spinner naming the agents currently producing a
turn. A room can hold several intelligence channels — one agent handing off
to another, a panel answering in parallel — so activity is tracked per
source channel instead of as one global "working" flag, and the status line
names the agents rather than saying something is happening.

Pure state and formatting: no prompt_toolkit, no I/O. The shell supplies a
clock and an invalidate callback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

# Braille spinner, the de-facto CLI convention (Claude Code, cargo, npm).
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
FRAME_SECONDS = 0.12
"""Spinner period. Also the shell's redraw interval while an agent works."""

_MAX_DETAIL = 28


@dataclass
class AgentActivity:
    """One agent's in-flight turn, as the status line sees it."""

    channel_id: str
    label: str
    started_at: float
    detail: str | None = None
    """What the agent is doing right now — ``thinking``, a tool name, …"""

    context_used: int | None = None
    context_size: int | None = None

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started_at)


@dataclass
class ActivityTracker:
    """The agents currently streaming into this console.

    ``on_change`` fires only when the rendered status would actually differ,
    so per-token updates don't queue a redraw each.
    """

    on_change: Callable[[], None] | None = None
    clock: Callable[[], float] = time.monotonic
    models: dict[str, str] = field(default_factory=dict)
    """Model per agent channel, as the agents report it.

    Kept beside the live turns rather than inside them: an agent's model
    outlives any single turn, and the status bar shows it while idle.
    """

    _active: dict[str, AgentActivity] = field(default_factory=dict)

    @property
    def active(self) -> tuple[AgentActivity, ...]:
        """Live activities, oldest turn first — a stable render order."""
        return tuple(sorted(self._active.values(), key=lambda item: item.started_at))

    def __bool__(self) -> bool:
        return bool(self._active)

    def start(self, channel_id: str, label: str) -> None:
        """Mark *channel_id* as working. Re-entry keeps the original clock.

        A single agent turn can stream more than once (a tool round resumes
        the same logical turn), and restarting the timer would under-report
        how long the user has been waiting.
        """
        existing = self._active.get(channel_id)
        if existing is not None:
            existing.label = label
            return
        self._active[channel_id] = AgentActivity(
            channel_id=channel_id,
            label=label,
            started_at=self.clock(),
        )
        self._changed()

    def note(self, channel_id: str, detail: str | None) -> None:
        """Record what the agent is doing — ignored if it is not working."""
        activity = self._active.get(channel_id)
        if activity is None or activity.detail == detail:
            return
        activity.detail = detail
        self._changed()

    def observe_usage(
        self,
        channel_id: str,
        *,
        used: int | None = None,
        size: int | None = None,
    ) -> None:
        """Record reported context usage (ACP ``usage_update``)."""
        activity = self._active.get(channel_id)
        if activity is None:
            return
        if used is not None and used != activity.context_used:
            activity.context_used = used
            self._changed()
        if size is not None:
            activity.context_size = size

    def set_model(self, channel_id: str, model: str) -> None:
        """Record the model an agent reports running."""
        if not channel_id or not model or self.models.get(channel_id) == model:
            return
        self.models[channel_id] = model
        self._changed()

    def finish(self, channel_id: str) -> None:
        if self._active.pop(channel_id, None) is not None:
            self._changed()

    def clear(self) -> None:
        if self._active or self.models:
            self._active.clear()
            self.models.clear()
            self._changed()

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()


def spinner_frame(elapsed: float) -> str:
    """The spinner glyph for an elapsed time — no frame counter to keep."""
    return _SPINNER_FRAMES[int(elapsed / FRAME_SECONDS) % len(_SPINNER_FRAMES)]


def format_elapsed(seconds: float) -> str:
    """``7s`` / ``1m 04s`` — a wait counter, whole seconds only."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m {total % 60:02d}s"


def format_tokens(count: int) -> str:
    """``840`` / ``12.3k`` / ``1.2M`` — context numbers stay one glance wide."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k".replace(".0k", "k")
    return f"{count / 1_000_000:.1f}M".replace(".0M", "M")


def format_activity(tracker: ActivityTracker, *, now: float | None = None) -> str | None:
    """The status-line fragment for the working agents, or None when idle.

    One agent names itself and what it is doing::

        ⠹ claude-code working 32s · Edit · 12.3k ctx

    Several name themselves and share the oldest turn's clock, because that
    is the wait the user actually experiences::

        ⠹ 2 agents working 32s · claude-code, reviewer
    """
    activities = tracker.active
    if not activities:
        return None
    moment = tracker.clock() if now is None else now
    oldest = activities[0].elapsed(moment)
    spinner = spinner_frame(oldest)
    elapsed = format_elapsed(oldest)
    if len(activities) > 1:
        names = ", ".join(item.label for item in activities)
        return f"{spinner} {len(activities)} agents working {elapsed} · {names}"

    only = activities[0]
    parts = [f"{spinner} {only.label} working {elapsed}"]
    if only.detail:
        detail = only.detail
        if len(detail) > _MAX_DETAIL:
            detail = f"{detail[: _MAX_DETAIL - 1]}…"
        parts.append(detail)
    if only.context_used is not None:
        parts.append(f"{format_tokens(only.context_used)} ctx")
    return " · ".join(parts)


__all__ = [
    "ActivityTracker",
    "AgentActivity",
    "format_activity",
    "format_elapsed",
    "format_tokens",
    "spinner_frame",
]
