"""Keyboard and mouse control tools for AI screen assistants.

Lets an AI agent type text, press keys, and click on UI elements
described in natural language. Mouse clicks use a vision model to
locate the element precisely (DPI-aware, PNG, system instruction).

Requires ``pyautogui`` and ``mss``::

    pip install roomkit[screen-input,screen-capture]

Example::

    from roomkit.video.vision import ScreenInputTools

    input_tools = ScreenInputTools(
        gemini_api_key="...",
        gemini_model="gemini-3.1-flash-image-preview",
    )

    voice = RealtimeVoiceChannel(
        "voice",
        tools=[*input_tools.definitions],
        tool_handler=input_tools.handler,
    )
"""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.video.vision.screen_input")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TYPE_TEXT_TOOL: dict[str, Any] = {
    "name": "type_text",
    "description": (
        "Type text using the keyboard into the currently focused field. "
        "Make sure the right field has focus first (use press_key with "
        "'ctrl+l' for browser address bar, 'tab' to move between fields)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to type.",
            },
        },
        "required": ["text"],
    },
}

PRESS_KEY_TOOL: dict[str, Any] = {
    "name": "press_key",
    "description": (
        "Press a keyboard key or key combination. "
        "Special keys: enter, tab, escape, backspace, delete, space, "
        "up, down, left, right, home, end, f1-f12, ctrl, alt, shift. "
        "Combos use + separator: 'ctrl+a', 'ctrl+c', 'alt+tab', 'ctrl+l'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Key name or combo. Examples: 'enter', 'tab', "
                    "'ctrl+a', 'ctrl+c', 'ctrl+v', 'alt+tab', 'ctrl+l'."
                ),
            },
        },
        "required": ["key"],
    },
}

SCROLL_TOOL: dict[str, Any] = {
    "name": "scroll",
    "description": (
        "Scroll the mouse wheel at the current position. "
        "Positive clicks scroll up, negative scroll down."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "clicks": {
                "type": "integer",
                "description": "Scroll amount (positive=up, negative=down).",
            },
        },
        "required": ["clicks"],
    },
}

CLICK_ELEMENT_TOOL: dict[str, Any] = {
    "name": "click_element",
    "description": (
        "Click on a UI element described in natural language. "
        "The tool captures the screen, uses vision AI to find the "
        "element's exact position, and clicks on it. "
        "Describe the element precisely: its label, icon, location."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "element": {
                "type": "string",
                "description": (
                    "Precise description of the UI element to click. "
                    "Examples: 'the Chrome icon in the taskbar', "
                    "'the address bar in the browser', "
                    "'the Search button', 'the first search result link'"
                ),
            },
            "button": {
                "type": "string",
                "enum": ["left", "right"],
                "description": "Mouse button (default: left).",
            },
            "double": {
                "type": "boolean",
                "description": "Double-click if true (default: false).",
            },
        },
        "required": ["element"],
    },
}

# ---------------------------------------------------------------------------
# VisionClick — locate elements via Gemini Vision (PNG + system instruction)
# ---------------------------------------------------------------------------

_LOCATE_SYSTEM = """\
You are specialized in locating GUI elements in screenshots.
You analyze desktop screenshots and identify VISUAL elements: \
icons, buttons, windows, widgets.

Important rules:
- Look for GRAPHICAL elements (icons, buttons, images), NOT text displayed in applications.
- Coordinates must ALWAYS be in absolute pixels of the provided image.
- Return ONLY a valid JSON object, no markdown, no explanation.\
"""

_LOCATE_USER = """\
Image size: {w}x{h} pixels.

Element to find: "{element}"

Instructions:
- Find the VISUAL element (icon, button, widget) matching the description.
- Ignore text displayed inside application windows (terminal, editor, browser, file manager).
- Focus on toolbars, taskbar, dock, sidebar, desktop, system menus.
- IMPORTANT: All coordinates must be integers in absolute pixels (NOT normalized 0-1).
- IMPORTANT: cx and cy must be the CENTER of the element's bounding box.
- cx must be between 0 and {w}. cy must be between 0 and {h}.

Return ONLY this JSON (no markdown, no text around):
{{"found": true, "cx": <center x in pixels>, "cy": <center y in pixels>, \
"box": {{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}}, "label": "<visible name>"}}

If not found:
{{"found": false, "cx": 0, "cy": 0, \
"box": {{"x1": 0, "y1": 0, "x2": 0, "y2": 0}}, "label": ""}}\
"""


def _get_scale_factor() -> tuple[float, float]:
    """Detect DPI scale factor per OS for coordinate conversion."""
    system = platform.system()
    if system == "Darwin":
        try:
            import mss
            import pyautogui

            with mss.mss() as sct:
                mon = sct.monitors[1]
                phys_w, phys_h = mon["width"], mon["height"]
            logical_w, logical_h = pyautogui.size()
            return logical_w / phys_w, logical_h / phys_h
        except Exception:
            return 1.0, 1.0
    if system == "Windows":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        except Exception:  # nosec B110 — best-effort DPI awareness
            pass
        return 1.0, 1.0
    # Linux
    gdk_scale = os.environ.get("GDK_SCALE", "1")
    try:
        s = float(gdk_scale)
        return 1.0 / s, 1.0 / s
    except ValueError:
        return 1.0, 1.0


def _capture_png(monitor: int = 1) -> tuple[bytes, int, int] | None:
    """Capture screen as PNG bytes + dimensions."""
    try:
        import mss
        from PIL import Image
    except ImportError:
        logger.error("mss + Pillow required")
        return None

    try:
        with mss.mss() as sct:
            if monitor >= len(sct.monitors):
                return None
            shot = sct.grab(sct.monitors[monitor])
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), shot.width, shot.height
    except Exception:
        logger.exception("Screenshot failed")
        return None


def _denormalize(result: dict[str, Any], w: int, h: int) -> dict[str, Any]:
    """Convert 0-1 normalized coords to pixels if model ignored instructions."""
    cx, cy = result.get("cx", 0), result.get("cy", 0)
    if 0 < cx <= 1 and 0 < cy <= 1:
        result["cx"] = int(cx * w)
        result["cy"] = int(cy * h)
        box = result.get("box", {})
        if box:
            result["box"] = {
                "x1": int(box.get("x1", 0) * w),
                "y1": int(box.get("y1", 0) * h),
                "x2": int(box.get("x2", 0) * w),
                "y2": int(box.get("y2", 0) * h),
            }
    return result


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from vision model response."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _find_element_sync(
    api_key: str,
    model: str,
    element: str,
    monitor: int,
) -> tuple[int, int] | None:
    """Locate a UI element using Gemini Vision (synchronous).

    Uses the exact same approach as VisionClick (test-click.py):
    sync client, PNG screenshot, system instruction, denormalize.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai required — pip install roomkit[gemini]")
        return None

    capture = _capture_png(monitor)
    if capture is None:
        return None
    png_bytes, img_w, img_h = capture

    prompt = _LOCATE_USER.format(w=img_w, h=img_h, element=element)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=_LOCATE_SYSTEM,
        ),
    )

    raw = response.text or ""
    logger.debug("Gemini raw response for '%s': %s", element, raw[:500])

    parsed = _parse_json_response(raw)
    if parsed is None or not parsed.get("found"):
        logger.warning("Element not found: %s (raw: %s)", element, raw[:300])
        return None

    parsed = _denormalize(parsed, img_w, img_h)

    # Compute center from bounding box (more stable than model's cx/cy)
    box = parsed.get("box", {})
    if box and box.get("x2", 0) > box.get("x1", 0):
        cx = (int(box["x1"]) + int(box["x2"])) // 2
        cy = (int(box["y1"]) + int(box["y2"])) // 2
    else:
        cx, cy = int(parsed["cx"]), int(parsed["cy"])

    logger.info(
        "Gemini locate '%s': center=(%d,%d) box=%s (img %dx%d)",
        element,
        cx,
        cy,
        box,
        img_w,
        img_h,
    )

    scale_x, scale_y = _get_scale_factor()
    screen_x = int(cx * scale_x)
    screen_y = int(cy * scale_y)

    logger.info(
        "click_element(%s) → img(%d,%d) → screen(%d,%d) label=%s",
        element,
        cx,
        cy,
        screen_x,
        screen_y,
        parsed.get("label", ""),
    )
    return screen_x, screen_y


async def _find_element_gemini(
    api_key: str,
    model: str,
    element: str,
    monitor: int,
) -> tuple[int, int] | None:
    """Async wrapper — runs the sync Gemini call in a thread pool."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _find_element_sync, api_key, model, element, monitor)


# ---------------------------------------------------------------------------
# pyautogui wrapper
# ---------------------------------------------------------------------------


def _get_pyautogui() -> Any:
    """Lazy-import pyautogui."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = False
        return pyautogui
    except ImportError:
        raise ImportError("pyautogui is required — pip install roomkit[screen-input]") from None


# ---------------------------------------------------------------------------
# ScreenInputTools
# ---------------------------------------------------------------------------


class ScreenInputTools:
    """Keyboard and mouse tools for AI agents.

    Provides ``type_text``, ``press_key``, ``scroll``, and
    ``click_element`` tools. ``click_element`` uses Gemini Vision
    directly (PNG + system instruction) for accurate element location.

    Args:
        gemini_api_key: API key for Gemini Vision (for click_element).
            If None, click_element is not available.
        gemini_model: Gemini model for element location.
        monitor: Monitor index for screenshots (1 = primary).

    Example::

        input_tools = ScreenInputTools(gemini_api_key="...")

        channel = RealtimeVoiceChannel(
            "voice",
            tools=[*input_tools.definitions],
            tool_handler=input_tools.handler,
        )
    """

    def __init__(
        self,
        *,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-3.1-flash-image-preview",
        monitor: int = 1,
    ) -> None:
        self._gemini_api_key = gemini_api_key
        self._gemini_model = gemini_model
        self._monitor = monitor

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Tool definitions. Includes click_element if Gemini key is set."""
        tools = [TYPE_TEXT_TOOL, PRESS_KEY_TOOL, SCROLL_TOOL]
        if self._gemini_api_key:
            tools.append(CLICK_ELEMENT_TOOL)
        return tools

    async def handler(
        self,
        session: VoiceSession,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Unified tool handler."""
        if name == "type_text":
            return self._type_text(arguments)
        if name == "press_key":
            return self._press_key(arguments)
        if name == "scroll":
            return self._scroll(arguments)
        if name == "click_element":
            return await self._click_element(arguments)
        return f"Unknown tool: {name}"

    @staticmethod
    def _type_text(args: dict[str, Any]) -> str:
        pag = _get_pyautogui()
        text = str(args["text"])
        logger.info("type_text(%r)", text)
        pag.typewrite(text, interval=0.02)
        return f"Typed: {text!r}"

    @staticmethod
    def _press_key(args: dict[str, Any]) -> str:
        pag = _get_pyautogui()
        key = str(args["key"])
        logger.info("press_key(%r)", key)
        if "+" in key:
            keys = [k.strip() for k in key.split("+")]
            pag.hotkey(*keys)
            return f"Pressed key combo: {key}"
        pag.press(key)
        return f"Pressed key: {key}"

    @staticmethod
    def _scroll(args: dict[str, Any]) -> str:
        pag = _get_pyautogui()
        clicks = int(args["clicks"])
        logger.info("scroll(%d)", clicks)
        pag.scroll(clicks)
        return f"Scrolled {clicks} clicks."

    async def _click_element(self, args: dict[str, Any]) -> str:
        if not self._gemini_api_key:
            return "click_element unavailable: no Gemini API key."

        element = str(args["element"])
        button = args.get("button", "left")
        double = args.get("double", False)

        logger.info(
            "click_element(%r, button=%s, double=%s)",
            element,
            button,
            double,
        )
        coords = await _find_element_gemini(
            self._gemini_api_key,
            self._gemini_model,
            element,
            self._monitor,
        )
        if coords is None:
            return f"Could not find element: {element}"

        pag = _get_pyautogui()
        x, y = coords
        clicks = 2 if double else 1
        pag.click(x, y, clicks=clicks, button=button)

        action = "Double-clicked" if double else "Clicked"
        return f"{action} on '{element}' at ({x}, {y}). Call describe_screen to verify the result."
