"""Reusable screen tools for AI voice/chat agents.

Provides ``describe_screen`` and ``locate_element`` tools that capture
a full-resolution screenshot and analyze it with a :class:`VisionProvider`.

Example with RealtimeVoiceChannel::

    from roomkit import DescribeScreenTool, GeminiVisionConfig, GeminiVisionProvider

    vision = GeminiVisionProvider(GeminiVisionConfig(api_key="..."))
    screen_tool = DescribeScreenTool(vision)

    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=backend,
        tools=screen_tool.definitions,
        tool_handler=screen_tool.handler,
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import VisionProvider

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.video.vision.screen_tool")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

DESCRIBE_SCREEN_TOOL: dict[str, Any] = {
    "name": "describe_screen",
    "description": (
        "Look at the user's screen right now and answer a specific visual question. "
        "You MUST call this tool whenever the user asks about anything visible on "
        "their screen. NEVER guess — always look first. "
        "Ask a precise, detailed question for accurate results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A precise visual question about the screen. "
                    "Examples: 'List all icons in the taskbar', "
                    "'What URL is in the address bar?', "
                    "'What text is visible in the main area?'"
                ),
            },
        },
        "required": ["query"],
    },
}

LOCATE_ELEMENT_TOOL: dict[str, Any] = {
    "name": "locate_element",
    "description": (
        "Find the exact pixel coordinates of a UI element on screen. "
        "You MUST call this tool before every click, move_mouse, or scroll "
        "action. Returns {x, y} center coordinates. "
        "Describe the element precisely: its label, icon, position context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "element": {
                "type": "string",
                "description": (
                    "Precise description of the element to find. "
                    "Examples: 'the Chrome icon in the taskbar', "
                    "'the address bar in the browser', "
                    "'the Search button next to the text field', "
                    "'the first link in the search results'"
                ),
            },
        },
        "required": ["element"],
    },
}

_LOCATE_PROMPT_TEMPLATE = """\
Find the UI element described below in this screenshot.
The screenshot is {width}x{height} pixels.

Element to find: {element}

Return ONLY a JSON object with the center coordinates:
{{"x": <integer>, "y": <integer>}}

Rules:
- x is horizontal (0=left edge, {width}=right edge)
- y is vertical (0=top edge, {height}=bottom edge)
- Return the CENTER of the element, not a corner
- Be as precise as possible
- If the element is not visible, return {{"x": -1, "y": -1, "error": "not found"}}
- Return ONLY the JSON, no other text\
"""


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


def _parse_coordinates(text: str) -> tuple[int, int] | None:
    """Extract (x, y) coordinates from vision model response."""
    # Try JSON parse first
    try:
        data = json.loads(text.strip())
        x, y = int(data["x"]), int(data["y"])
        if x >= 0 and y >= 0:
            return (x, y)
        return None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Try extracting JSON from surrounding text
    match = re.search(r'\{[^}]*"x"\s*:\s*(\d+)[^}]*"y"\s*:\s*(\d+)[^}]*\}', text)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    # Try simple "x, y" pattern
    match = re.search(r"(\d{2,5})\s*,\s*(\d{2,5})", text)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    return None


class DescribeScreenTool:
    """Screen vision tools for AI agents: describe and locate elements.

    Provides ``describe_screen`` and ``locate_element`` tool definitions
    and a unified handler.  Reuses a single :class:`VisionProvider`.

    Args:
        vision: Vision provider for frame analysis.
        monitor: Monitor index to capture (1 = primary).

    Example::

        vision = GeminiVisionProvider(GeminiVisionConfig(api_key="..."))
        tool = DescribeScreenTool(vision, monitor=1)

        channel = RealtimeVoiceChannel(
            "voice",
            tools=tool.definitions,
            tool_handler=tool.handler,
        )
    """

    def __init__(self, vision: VisionProvider, *, monitor: int = 1) -> None:
        self._vision = vision
        self._monitor = monitor

    @property
    def definition(self) -> dict[str, Any]:
        """Single tool definition (describe_screen only, for compat)."""
        return DESCRIBE_SCREEN_TOOL

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """All tool definitions: describe_screen + locate_element."""
        return [DESCRIBE_SCREEN_TOOL, LOCATE_ELEMENT_TOOL]

    async def analyze(self, query: str) -> str:
        """Capture a fresh screenshot and analyze it with *query*."""
        frame = capture_screen_frame(self._monitor)
        if frame is None:
            return "No screen frame available. Please wait a moment."

        result = await self._vision.analyze_frame(frame, prompt=query)
        return result.description or "Could not analyze the screen."

    async def locate(self, element: str) -> str:
        """Find element coordinates on screen via vision model.

        Returns a JSON string with ``x``, ``y`` keys, or an error.
        """
        frame = capture_screen_frame(self._monitor)
        if frame is None:
            return json.dumps({"error": "No screen frame available."})

        prompt = _LOCATE_PROMPT_TEMPLATE.format(
            width=frame.width,
            height=frame.height,
            element=element,
        )
        result = await self._vision.analyze_frame(frame, prompt=prompt)
        raw = result.description or ""

        coords = _parse_coordinates(raw)
        if coords is None:
            logger.warning("locate_element: could not parse coords from: %s", raw[:200])
            return json.dumps({"error": f"Could not locate: {element}"})

        x, y = coords
        logger.info("locate_element(%s) → (%d, %d)", element, x, y)
        return json.dumps({"x": x, "y": y})

    async def handler(
        self,
        session: VoiceSession,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Unified handler for describe_screen and locate_element."""
        if name == "describe_screen":
            query = str(arguments.get("query", "Describe what is on this screen."))
            logger.info("describe_screen(query='%s')", query[:100])
            result = await self.analyze(query)
            logger.info("describe_screen result: %s", result[:200])
            return result

        if name == "locate_element":
            element = str(arguments.get("element", ""))
            logger.info("locate_element(element='%s')", element[:100])
            return await self.locate(element)

        return f"Unknown tool: {name}"
