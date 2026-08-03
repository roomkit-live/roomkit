"""Tests for the console's agent activity indicator (roomkit.console._activity).

Pure state and formatting — no prompt_toolkit, no terminal. The shell's use
of it (spinner task, toolbar text) is covered in test_cli_prompt_shell.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from roomkit.channels.cli import CLIChannel
from roomkit.console._activity import (
    FRAME_SECONDS,
    ActivityTracker,
    format_activity,
    format_elapsed,
    format_tokens,
    spinner_frame,
)
from roomkit.console._chat import ConsoleStreamRenderer
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.models.streaming import ThinkingDeltaMarker


class _Clock:
    """A hand-cranked monotonic clock."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _tracker() -> tuple[ActivityTracker, _Clock, list[int]]:
    clock = _Clock()
    changes: list[int] = []
    tracker = ActivityTracker(on_change=lambda: changes.append(1), clock=clock)
    return tracker, clock, changes


class TestActivityTracker:
    def test_idle_tracker_renders_nothing(self) -> None:
        tracker, _clock, _changes = _tracker()
        assert not tracker
        assert format_activity(tracker) is None

    def test_one_agent_names_itself_with_elapsed_and_detail(self) -> None:
        tracker, clock, _changes = _tracker()
        tracker.start("acp", "Claude Code")
        clock.advance(32)
        tracker.note("acp", "Edit")

        line = format_activity(tracker)

        assert line is not None
        assert "Claude Code working 32s" in line
        assert line.endswith("· Edit")
        assert line[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def test_context_usage_joins_the_line(self) -> None:
        tracker, clock, _changes = _tracker()
        tracker.start("acp", "Claude Code")
        tracker.observe_usage("acp", used=12_345, size=200_000)
        clock.advance(3)

        assert format_activity(tracker) is not None
        assert "12.3k ctx" in format_activity(tracker)  # type: ignore[operator]

    def test_usage_for_an_idle_agent_is_ignored(self) -> None:
        tracker, _clock, changes = _tracker()
        tracker.observe_usage("acp", used=999)
        assert not changes
        assert format_activity(tracker) is None

    def test_several_agents_are_counted_and_named(self) -> None:
        tracker, clock, _changes = _tracker()
        tracker.start("planner", "Planner")
        clock.advance(10)
        tracker.start("coder", "Coder")
        clock.advance(5)

        line = format_activity(tracker)

        # The oldest turn owns the clock — that is the wait being lived.
        assert line is not None
        assert "2 agents working 15s" in line
        assert line.endswith("· Planner, Coder")

    def test_restart_keeps_the_original_clock(self) -> None:
        # A turn streams more than once (a tool round resumes it); restarting
        # the timer would under-report how long the user has waited.
        tracker, clock, _changes = _tracker()
        tracker.start("acp", "Claude Code")
        clock.advance(20)
        tracker.start("acp", "Claude Code")
        clock.advance(1)

        assert "working 21s" in format_activity(tracker)  # type: ignore[operator]

    def test_finish_clears_the_agent(self) -> None:
        tracker, _clock, _changes = _tracker()
        tracker.start("acp", "Claude Code")
        tracker.finish("acp")
        assert format_activity(tracker) is None
        tracker.finish("acp")  # idempotent

    def test_repeated_detail_does_not_queue_a_redraw(self) -> None:
        tracker, _clock, changes = _tracker()
        tracker.start("acp", "Claude Code")
        before = len(changes)
        tracker.note("acp", "responding")
        tracker.note("acp", "responding")
        tracker.note("acp", "responding")
        assert len(changes) == before + 1

    def test_long_detail_is_truncated(self) -> None:
        tracker, _clock, _changes = _tracker()
        tracker.start("acp", "Agent")
        tracker.note("acp", "a-very-long-tool-name-that-runs-off-the-bar")
        line = format_activity(tracker)
        assert line is not None
        assert line.endswith("…")
        assert len(line.split(" · ")[-1]) == 28

    def test_clear_drops_everything(self) -> None:
        tracker, _clock, _changes = _tracker()
        tracker.start("a", "A")
        tracker.start("b", "B")
        tracker.clear()
        assert not tracker


class TestFormatting:
    def test_spinner_advances_with_time_and_wraps(self) -> None:
        first = spinner_frame(0)
        assert spinner_frame(FRAME_SECONDS) != first
        assert spinner_frame(FRAME_SECONDS * 10) == first

    def test_elapsed_reads_as_a_wait(self) -> None:
        assert format_elapsed(0.4) == "0s"
        assert format_elapsed(7.9) == "7s"
        assert format_elapsed(59) == "59s"
        assert format_elapsed(64) == "1m 04s"
        assert format_elapsed(3600) == "60m 00s"

    def test_tokens_stay_one_glance_wide(self) -> None:
        assert format_tokens(840) == "840"
        assert format_tokens(1000) == "1k"
        assert format_tokens(12_345) == "12.3k"
        assert format_tokens(1_500_000) == "1.5M"


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="cli",
        room_id="room-1",
        channel_type=ChannelType.CLI,
        category=ChannelCategory.TRANSPORT,
    )


def _agent_event() -> RoomEvent:
    return RoomEvent(
        room_id="room-1",
        type=EventType.MESSAGE,
        source=EventSource(channel_id="acp-agent", channel_type=ChannelType.AI),
        content=TextContent(body=""),
    )


def _tool_event(event_type: EventType, status: str) -> RoomEvent:
    return RoomEvent(
        room_id="room-1",
        type=event_type,
        source=EventSource(channel_id="acp-agent", channel_type=ChannelType.AI),
        content=ToolCallContent(tool_name="Edit", tool_id="t1", status=status),  # type: ignore[arg-type]
    )


class TestChannelReportsActivity:
    async def test_stream_reports_who_works_and_what_it_does(self) -> None:
        cli = CLIChannel(
            "cli", console=True, use_color=False, agent_label=lambda cid: "Claude Code"
        )
        tracker, clock, _changes = _tracker()
        cli._activity = tracker
        seen: list[str | None] = []

        async def stream() -> Any:
            seen.append(_detail(tracker))
            yield ThinkingDeltaMarker(thinking="pondering")
            seen.append(_detail(tracker))
            yield _tool_event(EventType.TOOL_CALL_START, "pending")
            seen.append(_detail(tracker))
            yield _tool_event(EventType.TOOL_CALL_END, "completed")
            seen.append(_detail(tracker))
            yield "The answer."
            seen.append(_detail(tracker))

        context = RoomContext(room=Room(id="room-1"))
        await cli.deliver_stream(stream(), _agent_event(), _binding(), context)

        assert seen == [None, "thinking", "Edit", None, "responding"]
        # The turn is over: nothing left working.
        assert not tracker

    async def test_activity_is_labelled_and_keyed_by_the_source_agent(self) -> None:
        cli = CLIChannel(
            "cli", console=True, use_color=False, agent_label=lambda cid: "Claude Code"
        )
        tracker, _clock, _changes = _tracker()
        cli._activity = tracker
        captured: list[tuple[str, str]] = []

        async def stream() -> Any:
            captured.extend((item.channel_id, item.label) for item in tracker.active)
            yield "hi"

        context = RoomContext(room=Room(id="room-1"))
        await cli.deliver_stream(stream(), _agent_event(), _binding(), context)

        assert captured == [("acp-agent", "Claude Code")]

    async def test_failed_stream_still_clears_the_indicator(self) -> None:
        cli = CLIChannel("cli", console=True, use_color=False)
        tracker, _clock, _changes = _tracker()
        cli._activity = tracker

        async def stream() -> Any:
            yield "partial"
            raise RuntimeError("agent died")

        context = RoomContext(room=Room(id="room-1"))
        with pytest.raises(RuntimeError):
            await cli.deliver_stream(stream(), _agent_event(), _binding(), context)

        assert not tracker

    async def test_streaming_without_a_tracker_renders_normally(self) -> None:
        # Outside the pinned shell there is no status bar to feed; the stream
        # must render exactly as before.
        cli = CLIChannel("cli", console=True, use_color=False)
        assert cli._activity is None
        rendered: list[str] = []

        async def stream() -> Any:
            yield "hello"

        context = RoomContext(room=Room(id="room-1"))
        with patch.object(
            ConsoleStreamRenderer, "add_text", lambda _self, text: rendered.append(text)
        ):
            await cli.deliver_stream(stream(), _agent_event(), _binding(), context)

        assert rendered == ["hello"]


def _detail(tracker: ActivityTracker) -> str | None:
    active = tracker.active
    return active[0].detail if active else None
