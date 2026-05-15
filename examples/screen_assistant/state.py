"""Shared mutable state for the screen assistant example.

Wraps the objects rebuilt when the active monitor changes so that
helpers can hold a single reference instead of juggling closures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from roomkit.video.backends.screen import ScreenCaptureBackend
from roomkit.video.vision.base import VisionProvider
from roomkit.video.vision.screen_input import ScreenInputTools
from roomkit.video.vision.screen_tool import DescribeScreenTool

from .omniview import OmniViewClient
from .telemetry import CostTrackingTelemetry


@dataclass
class ScreenAssistantState:
    """Mutable state shared across handlers and observers."""

    vision: VisionProvider
    screen_backend: ScreenCaptureBackend
    cost_telemetry: CostTrackingTelemetry
    monitor: int = 1
    screen_tool: DescribeScreenTool = field(init=False)
    input_tools: ScreenInputTools = field(init=False)
    omniview: OmniViewClient | None = None
    latest_description: str = ""
    previous_description: str = ""
    auto_verify: bool = True
    frame_count: int = 0
    # True once the user has produced a final transcription. Until then,
    # vision injections stay silent=True so they can't barge in on the
    # agent's opening greeting.
    user_has_spoken: bool = False
    # True while the realtime model is mid-response. Set by
    # provider.on_response_start / on_response_end. Used to skip vision
    # injections that would barge in on the agent's own speech — even
    # silent=True is best-effort on Gemini Live once audio is flowing.
    agent_speaking: bool = False
    # True while the dispatcher is executing a tool call. Vision arriving
    # during this window can confuse Gemini into firing a new response
    # over the (forthcoming) tool result.
    tool_in_progress: bool = False
    # macOS app that should hold keyboard focus before we send keyboard
    # shortcuts (e.g. command+t). Vision sees the *rendered* foreground,
    # which can differ from the *keyboard-focused* app, so we activate
    # explicitly via osascript before any modifier-key combo.
    target_app: str = "Google Chrome"

    def __post_init__(self) -> None:
        self.screen_tool = DescribeScreenTool(self.vision, monitor=self.monitor)
        self.input_tools = ScreenInputTools(vision=self.vision, monitor=self.monitor)

    def switch_monitor(self, new_monitor: int) -> None:
        """Rebuild monitor-bound tools and update the capture backend."""
        self.monitor = new_monitor
        self.screen_tool = DescribeScreenTool(self.vision, monitor=new_monitor)
        self.input_tools = ScreenInputTools(vision=self.vision, monitor=new_monitor)
        self.screen_backend._monitor = new_monitor  # noqa: SLF001
        if self.omniview is not None:
            self.omniview.monitor = new_monitor

    def record_description(self, description: str) -> None:
        """Update the latest screen description, keeping the previous value."""
        self.previous_description = self.latest_description
        self.latest_description = description
