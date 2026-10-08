"""Keyboard and mouse control tools for AI screen assistants.

Lets an AI agent type text, press keys, and click on UI elements
described in natural language. Mouse clicks use a VisionProvider to
locate the element precisely (DPI-aware).

Requires ``pyautogui`` and ``mss``::

    pip install roomkit[screen-input,screen-capture]

Example::

    from roomkit.video.vision import ScreenInputTools, GeminiVisionProvider

    vision = GeminiVisionProvider(GeminiVisionConfig(api_key="..."))
    input_tools = ScreenInputTools(vision=vision)

    voice = RealtimeVoiceChannel(
        "voice",
        tools=[*input_tools.definitions],
        tool_handler=input_tools.handler,
    )
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess  # nosec B404
import sys
from typing import TYPE_CHECKING, Any

from roomkit._text import quoted
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.telemetry.redaction import redact
from roomkit.video.vision.screen_tool import capture_screen_frame

if TYPE_CHECKING:
    from roomkit.video.vision.base import VisionProvider

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


def _press_key_description() -> str:
    """Build OS-aware press_key tool description."""
    base = (
        "Press a keyboard key or key combination. "
        "Special keys: enter, tab, escape, backspace, delete, space, "
        "up, down, left, right, home, end, f1-f12, ctrl, alt, shift, command. "
        "Combos use + separator."
    )
    if platform.system() == "Darwin":
        return (
            f"{base} "
            "IMPORTANT: On macOS, use 'command' (NOT 'ctrl') for standard shortcuts. "
            "Examples: 'command+t' (new tab), 'command+l' (address bar), "
            "'command+c' (copy), 'command+v' (paste), 'command+space' (Spotlight)."
        )
    return (
        f"{base} "
        "Examples: 'ctrl+t' (new tab), 'ctrl+l' (address bar), "
        "'ctrl+c' (copy), 'ctrl+v' (paste), 'alt+tab' (switch window)."
    )


def _press_key_param_description() -> str:
    """Build OS-aware key parameter description."""
    if platform.system() == "Darwin":
        return (
            "Key name or combo. macOS examples: 'enter', 'tab', "
            "'command+t', 'command+l', 'command+c', 'command+space'."
        )
    return (
        "Key name or combo. Examples: 'enter', 'tab', "
        "'ctrl+a', 'ctrl+c', 'ctrl+v', 'alt+tab', 'ctrl+l'."
    )


def _build_press_key_tool() -> dict[str, Any]:
    """Build the press_key tool definition lazily (OS-aware descriptions)."""
    return {
        "name": "press_key",
        "description": _press_key_description(),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": _press_key_param_description(),
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
# Element location via VisionProvider
# ---------------------------------------------------------------------------

_LOCATE_PROMPT = """\
Image size: {w}x{h} pixels.

Element to find: {element}

Instructions:
- Find the VISUAL element (icon, button, widget) matching the description.
- Ignore text displayed inside application windows.
- Focus on toolbars, taskbar, dock, sidebar, desktop, system menus.
- IMPORTANT: All coordinates must be integers in absolute pixels \
(NOT normalized 0-1).
- IMPORTANT: cx and cy must be the CENTER of the element's bounding box.
- cx must be between 0 and {w}. cy must be between 0 and {h}.

Return ONLY this JSON (no markdown, no text around):
{{"found": true, "cx": <center x in pixels>, "cy": <center y in pixels>, \
"box": {{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}}, \
"label": "<visible name>"}}

If not found:
{{"found": false, "cx": 0, "cy": 0, \
"box": {{"x1": 0, "y1": 0, "x2": 0, "y2": 0}}, "label": ""}}\
"""

_ELEMENT_CHARS = 200
"""The characters of the element a locate prompt quotes."""


def _locate_prompt(element: str, width: int, height: int) -> str:
    """The prompt that asks a vision model where *element* is on a screen of
    *width* by *height*: the element, a tool call's argument, quoted
    (RFC §6.4)."""
    return _LOCATE_PROMPT.format(w=width, h=height, element=quoted(element, _ELEMENT_CHARS))


_INT = {"type": "integer"}

#: The shape of a locate answer, for a provider that constrains its output to
#: it (RFC §6.7): the answer is then parsed as it is, with no repair.
_LOCATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "cx": {"type": "integer", "description": "Center x in absolute pixels."},
        "cy": {"type": "integer", "description": "Center y in absolute pixels."},
        "box": {
            "type": "object",
            "properties": {"x1": _INT, "y1": _INT, "x2": _INT, "y2": _INT},
            "required": ["x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
        "label": {"type": "string", "description": "The element's visible name."},
    },
    "required": ["found", "cx", "cy", "box", "label"],
    "additionalProperties": False,
}


_dpi_initialized = False


def _get_scale_factor() -> tuple[float, float]:
    """Detect DPI scale factor per OS for coordinate conversion."""
    global _dpi_initialized  # noqa: PLW0603

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
        if not _dpi_initialized:
            try:
                import ctypes

                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # ty: ignore[unresolved-attribute]
            except Exception:  # nosec B110 — best-effort DPI awareness
                pass
            _dpi_initialized = True
        return 1.0, 1.0
    # Linux
    gdk_scale = os.environ.get("GDK_SCALE", "1")
    try:
        s = float(gdk_scale)
        return 1.0 / s, 1.0 / s
    except ValueError:
        return 1.0, 1.0


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
    """Extract JSON from vision model response.

    Gemini sometimes returns malformed box JSON (e.g. missing keys,
    stray quotes). When full JSON parsing fails, fall back to regex
    extraction of the top-level fields (found, cx, cy).
    """
    # Try clean JSON first
    try:
        result: dict[str, Any] = json.loads(text.strip())
        return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return result
        except json.JSONDecodeError:
            pass

    # Fallback: extract cx/cy/label via regex when JSON is garbled
    found_match = re.search(r'"found"\s*:\s*(true|false)', text, re.IGNORECASE)
    cx_match = re.search(r'"cx"\s*:\s*(\d+)', text)
    cy_match = re.search(r'"cy"\s*:\s*(\d+)', text)
    if found_match and cx_match and cy_match:
        found = found_match.group(1).lower() == "true"
        label_match = re.search(r'"label"\s*:\s*"([^"]*)"', text)
        label = label_match.group(1) if label_match else ""
        logger.debug(
            "Fallback JSON parse: found=%s cx=%s cy=%s label=%s",
            found,
            cx_match.group(1),
            cy_match.group(1),
            label,
        )
        return {
            "found": found,
            "cx": int(cx_match.group(1)),
            "cy": int(cy_match.group(1)),
            "box": {},
            "label": label,
        }
    return None


async def _locate(vision: VisionProvider, frame: Any, prompt: str) -> dict[str, Any] | None:
    """The locate answer as a dict, or ``None`` when there is no usable one.

    A provider that constrains its output returns the document the schema
    describes, parsed as it is; any other gets the prompt alone, and its answer
    goes through the repairs of :func:`_parse_json_response`.
    """
    if vision.supports_response_schema:
        try:
            result = await vision.analyze_frame(
                frame, prompt=prompt, response_schema=_LOCATE_SCHEMA
            )
        except ResponseSchemaError as exc:
            logger.warning("Vision locate gave no answer in the locate shape: %s", exc)
            return None
        return json.loads(result.description)
    result = await vision.analyze_frame(frame, prompt=prompt)
    raw = result.description or ""
    logger.debug("Vision locate raw: %s", raw[:500])
    parsed = _parse_json_response(raw)
    if parsed is None:
        logger.warning("Vision locate answer has no locate document: %s", raw[:300])
    return parsed


async def _find_element(
    vision: VisionProvider,
    element: str,
    monitor: int,
) -> tuple[int, int] | None:
    """Locate a UI element on screen using any VisionProvider.

    Captures a screenshot, sends it to the vision provider with a
    locate prompt, parses coordinates, and applies DPI scaling.
    """
    frame = capture_screen_frame(monitor)
    if frame is None:
        return None

    prompt = _locate_prompt(element, frame.width, frame.height)
    parsed = await _locate(vision, frame, prompt)
    if parsed is None or not parsed.get("found"):
        logger.warning("Element not found: %s (answer: %s)", element, str(parsed)[:300])
        return None

    parsed = _denormalize(parsed, frame.width, frame.height)

    # Compute center from bounding box (more stable than model's cx/cy)
    box = parsed.get("box", {})
    has_full_box = (
        box
        and all(k in box for k in ("x1", "y1", "x2", "y2"))
        and box.get("x2", 0) > box.get("x1", 0)
    )
    if has_full_box:
        cx = (int(box["x1"]) + int(box["x2"])) // 2
        cy = (int(box["y1"]) + int(box["y2"])) // 2
    else:
        cx, cy = int(parsed["cx"]), int(parsed["cy"])

    logger.info(
        "locate '%s': center=(%d,%d) box=%s (img %dx%d)",
        element,
        cx,
        cy,
        box,
        frame.width,
        frame.height,
    )

    # Reject coordinates outside the captured image bounds
    if cx < 0 or cx >= frame.width or cy < 0 or cy >= frame.height:
        logger.warning(
            "locate '%s': coordinates (%d,%d) outside image bounds (%dx%d)",
            element,
            cx,
            cy,
            frame.width,
            frame.height,
        )
        return None

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


# ---------------------------------------------------------------------------
# pyautogui wrapper
# ---------------------------------------------------------------------------

_FAILSAFE_ERROR = (
    "Stopped by the user: the mouse was moved to a screen corner (pyautogui "
    "failsafe), so nothing was typed or clicked. Do not retry until the user "
    "asks you to continue."
)


def _get_pyautogui() -> Any:
    """Lazy-import pyautogui.

    Its failsafe is left as the host set it, on by default: a person who
    moves the mouse to a screen corner makes every following call raise
    ``FailSafeException``. That is the only emergency stop a person has
    over a mouse and keyboard driven by a model, so the library never
    turns it off; a host that wants it off sets ``pyautogui.FAILSAFE``
    itself.
    """
    try:
        import pyautogui

        return pyautogui
    except ImportError:
        raise ImportError("pyautogui is required — pip install roomkit[screen-input]") from None


def _is_failsafe(exc: BaseException) -> bool:
    """Whether *exc* is pyautogui's failsafe: the mouse sits in a screen corner.

    pyautogui is necessarily imported once it raised, so the class is read
    from ``sys.modules`` rather than importing the optional dependency here.
    """
    pyautogui = sys.modules.get("pyautogui")
    return pyautogui is not None and isinstance(exc, pyautogui.FailSafeException)


def _clipboard_paste(text: str) -> None:
    """Type text via clipboard paste — works with all keyboard layouts.

    Uses platform-native clipboard commands (pbcopy/xclip/clip) and
    keyboard paste shortcuts. Falls back to pyautogui.typewrite on failure.
    """
    pag = _get_pyautogui()
    system = platform.system()

    try:
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)  # nosec B603 B607
            pag.hotkey("command", "v")
        elif system == "Windows":
            subprocess.run(["clip"], input=text.encode(), check=True)  # nosec B603 B607
            pag.hotkey("ctrl", "v")
        else:
            subprocess.run(  # nosec B603 B607
                ["xclip", "-selection", "clipboard"],
                input=text.encode(),
                check=True,
            )
            pag.hotkey("ctrl", "v")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Clipboard paste failed, falling back to typewrite")
        pag.typewrite(text, interval=0.02)


# ---------------------------------------------------------------------------
# ScreenInputTools
# ---------------------------------------------------------------------------


class ScreenInputTools:
    """Keyboard and mouse tools for AI agents.

    Provides ``type_text``, ``press_key``, ``scroll``, and
    ``click_element`` tools. ``click_element`` uses any
    :class:`VisionProvider` to locate elements by description.

    Args:
        vision: Vision provider for element location (optional).
            Required for ``click_element``. Works with Gemini, OpenAI,
            Ollama, or any VisionProvider implementation.
        monitor: Monitor index for screenshots (1 = primary).

    Example::

        input_tools = ScreenInputTools(vision=vision_provider)

        channel = RealtimeVoiceChannel(
            "voice",
            tools=[*input_tools.definitions],
            tool_handler=input_tools.handler,
        )
    """

    def __init__(
        self,
        *,
        vision: VisionProvider | None = None,
        monitor: int = 1,
    ) -> None:
        self._vision = vision
        self._monitor = monitor

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Tool definitions. Includes click_element if vision is set."""
        tools = [TYPE_TEXT_TOOL, _build_press_key_tool(), SCROLL_TOOL]
        if self._vision is not None:
            tools.append(CLICK_ELEMENT_TOOL)
        return tools

    async def handler(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Unified tool handler.

        When the person stops the input with pyautogui's failsafe (mouse in
        a screen corner), the call answers an error the model reads instead
        of raising: nothing was typed or clicked, and every further call is
        refused the same way while the mouse stays there.
        """
        try:
            return await self._dispatch(name, arguments)
        except Exception as exc:
            if not _is_failsafe(exc):
                raise
            logger.warning("%s refused: pyautogui failsafe (mouse in a screen corner)", name)
            return json.dumps({"error": _FAILSAFE_ERROR})

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Run the tool *name* asks for."""
        if name == "type_text":
            return self._type_text(arguments)
        if name == "press_key":
            return self._press_key(arguments)
        if name == "scroll":
            return self._scroll(arguments)
        if name == "click_element":
            return await self._click_element(arguments)
        raise UnservedToolCallError(f"tool {name!r} is not served here")

    @staticmethod
    def _type_text(args: dict[str, Any]) -> str:
        text = str(args["text"])
        logger.debug("type_text(%s)", redact(text))
        _clipboard_paste(text)
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
        if self._vision is None:
            return "click_element unavailable: no vision provider."

        element = str(args["element"])
        button = args.get("button", "left")
        double = args.get("double", False)

        logger.info(
            "click_element(%r, button=%s, double=%s)",
            element,
            button,
            double,
        )
        coords = await _find_element(self._vision, element, self._monitor)
        if coords is None:
            return f"Could not find element: {element}"

        pag = _get_pyautogui()
        x, y = coords
        clicks = 2 if double else 1
        pag.click(x, y, clicks=clicks, button=button)

        action = "Double-clicked" if double else "Clicked"
        return f"{action} on '{element}' at ({x}, {y}). Call describe_screen to verify the result."
