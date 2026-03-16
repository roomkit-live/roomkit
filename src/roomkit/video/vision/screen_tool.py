"""Reusable ``describe_screen`` tool for AI voice/chat agents.

Captures a full-resolution screenshot and analyzes it with a
:class:`VisionProvider`, returning a natural-language description.

Provides a tool definition dict and handler compatible with both
:class:`RealtimeVoiceChannel` (``tools`` / ``tool_handler``) and
:class:`AIChannel` (``AITool`` / ``tool_handler``).

Example with RealtimeVoiceChannel::

    from roomkit import DescribeScreenTool, GeminiVisionConfig, GeminiVisionProvider

    vision = GeminiVisionProvider(GeminiVisionConfig(api_key="..."))
    screen_tool = DescribeScreenTool(vision)

    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=backend,
        tools=[screen_tool.definition],
        tool_handler=screen_tool.handler,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import VisionProvider

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.video.vision.screen_tool")

TOOL_NAME = "describe_screen"

TOOL_DEFINITION: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Look at the user's screen right now and answer a specific visual question. "
        "You MUST call this tool whenever the user asks about anything visible on "
        "their screen: icon positions, button labels, menu items, text content, "
        "element ordering, colors, or layout. NEVER guess — always look first. "
        "Ask a precise, detailed question for accurate results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A precise visual question about the screen. "
                    "Be specific about what you want to know. "
                    "Examples: "
                    "'List all icons in the taskbar from left to right with their names', "
                    "'Where exactly is the Google Chrome icon? Describe its position "
                    "relative to other icons', "
                    "'What URL is shown in the browser address bar?', "
                    "'What text is visible in the main content area?', "
                    "'Is the Chrome icon to the left or right of the file manager icon?'"
                ),
            },
        },
        "required": ["query"],
    },
}


def capture_screen_frame(monitor: int = 1) -> VideoFrame | None:
    """Capture a full-resolution screenshot as a raw RGB :class:`VideoFrame`.

    Uses ``mss`` for cross-platform screen capture.  Unlike
    :class:`ScreenCaptureBackend` frames (which may be downscaled), this
    always returns native resolution for accurate text reading.

    Args:
        monitor: Monitor index (1 = primary, 2+ = secondary).

    Returns:
        A ``VideoFrame`` with ``codec="raw_rgb24"``, or ``None`` on failure.
    """
    try:
        import mss
    except ImportError:
        logger.error("mss is required — pip install roomkit[screen-capture]")
        return None

    try:
        with mss.mss() as sct:
            if monitor >= len(sct.monitors):
                logger.error(
                    "Monitor %d not found (available: %d)",
                    monitor,
                    len(sct.monitors) - 1,
                )
                return None
            mon = sct.monitors[monitor]
            shot = sct.grab(mon)
            return VideoFrame(
                data=shot.rgb,
                codec="raw_rgb24",
                width=shot.width,
                height=shot.height,
            )
    except Exception:
        logger.exception("Failed to capture screenshot")
        return None


class DescribeScreenTool:
    """Reusable tool that lets AI agents analyze the current screen.

    Bundles the tool definition (JSON schema), screen capture, and
    vision analysis into a single object.  Reuses a single
    :class:`VisionProvider` instance and passes the query as a
    per-call ``prompt`` override.

    Args:
        vision: Vision provider for frame analysis (Gemini, OpenAI, etc.).
        monitor: Monitor index to capture (1 = primary).

    Example::

        vision = GeminiVisionProvider(GeminiVisionConfig(api_key="..."))
        tool = DescribeScreenTool(vision, monitor=1)

        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=backend,
            tools=[tool.definition],
            tool_handler=tool.handler,
        )
    """

    def __init__(self, vision: VisionProvider, *, monitor: int = 1) -> None:
        self._vision = vision
        self._monitor = monitor

    @property
    def definition(self) -> dict[str, Any]:
        """Tool definition dict for ``RealtimeVoiceChannel(tools=[...])``."""
        return TOOL_DEFINITION

    async def analyze(self, query: str) -> str:
        """Capture a fresh screenshot and analyze it with *query*.

        Args:
            query: A precise visual question about the screen.

        Returns:
            Natural-language description, or an error message.
        """
        frame = capture_screen_frame(self._monitor)
        if frame is None:
            return "No screen frame available. Please wait a moment."

        result = await self._vision.analyze_frame(frame, prompt=query)
        return result.description or "Could not analyze the screen."

    async def handler(
        self,
        session: VoiceSession,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Tool handler compatible with ``RealtimeVoiceChannel(tool_handler=...)``."""
        if name != TOOL_NAME:
            return f"Unknown tool: {name}"

        query = str(arguments.get("query", "Describe what is on this screen."))
        logger.info("describe_screen(query='%s')", query[:100])
        result = await self.analyze(query)
        logger.info("describe_screen result: %s", result[:200])
        return result
