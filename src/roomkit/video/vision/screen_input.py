"""Mouse and keyboard control tools for AI screen assistants.

Lets an AI agent move the mouse, click, type text, and press keys
on the user's screen via ``pyautogui``.

Provides tool definitions and a handler compatible with
:class:`RealtimeVoiceChannel` and :class:`AIChannel`.

Requires ``pyautogui``::

    pip install roomkit[screen-input]

Example::

    from roomkit.video.vision import ScreenInputTools

    input_tools = ScreenInputTools()

    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=backend,
        tools=[screen_tool.definition, *input_tools.definitions],
        tool_handler=...,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.video.vision.screen_input")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

MOVE_MOUSE_TOOL: dict[str, Any] = {
    "name": "move_mouse",
    "description": (
        "Move the mouse cursor to exact screen coordinates. "
        "Use describe_screen first to find the position of the target element."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate in pixels."},
            "y": {"type": "integer", "description": "Y coordinate in pixels."},
        },
        "required": ["x", "y"],
    },
}

CLICK_TOOL: dict[str, Any] = {
    "name": "click",
    "description": (
        "Click at exact screen coordinates. "
        "Use describe_screen first to find the position of the target element. "
        "Defaults to left click."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate in pixels."},
            "y": {"type": "integer", "description": "Y coordinate in pixels."},
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button (default: left).",
            },
            "double": {
                "type": "boolean",
                "description": "Double-click if true (default: false).",
            },
        },
        "required": ["x", "y"],
    },
}

TYPE_TEXT_TOOL: dict[str, Any] = {
    "name": "type_text",
    "description": (
        "Type text using the keyboard. The text is typed character by character "
        "into whatever field or application currently has focus. "
        "Click on the target input field first if needed."
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
        "For special keys use: enter, tab, escape, backspace, delete, space, "
        "up, down, left, right, home, end, pageup, pagedown, "
        "f1-f12, ctrl, alt, shift, command. "
        "For combos use + separator: 'ctrl+a', 'ctrl+c', 'alt+tab'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Key name or combo. Examples: 'enter', 'tab', "
                    "'ctrl+a', 'ctrl+c', 'ctrl+v', 'alt+tab'."
                ),
            },
        },
        "required": ["key"],
    },
}

SCROLL_TOOL: dict[str, Any] = {
    "name": "scroll",
    "description": (
        "Scroll the mouse wheel at a given position. "
        "Positive clicks scroll up, negative scroll down."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "clicks": {
                "type": "integer",
                "description": "Scroll amount (positive=up, negative=down).",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate (optional, uses current pos).",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate (optional, uses current pos).",
            },
        },
        "required": ["clicks"],
    },
}

ALL_DEFINITIONS: list[dict[str, Any]] = [
    MOVE_MOUSE_TOOL,
    CLICK_TOOL,
    TYPE_TEXT_TOOL,
    PRESS_KEY_TOOL,
    SCROLL_TOOL,
]


# ---------------------------------------------------------------------------
# pyautogui wrapper
# ---------------------------------------------------------------------------


def _get_pyautogui() -> Any:
    """Lazy-import pyautogui with a clear error message."""
    try:
        import pyautogui

        # Disable the fail-safe (moving mouse to corner raises exception)
        # since the AI controls the mouse position.
        pyautogui.FAILSAFE = False
        return pyautogui
    except ImportError:
        raise ImportError(
            "pyautogui is required for screen input tools. "
            "Install with: pip install roomkit[screen-input]"
        ) from None


# ---------------------------------------------------------------------------
# ScreenInputTools
# ---------------------------------------------------------------------------


class ScreenInputTools:
    """Mouse and keyboard control tools for AI agents.

    Provides tool definitions and a unified handler for mouse
    movement, clicking, typing, key presses, and scrolling.

    Example::

        input_tools = ScreenInputTools()

        channel = RealtimeVoiceChannel(
            "voice",
            tools=[*input_tools.definitions],
            tool_handler=input_tools.handler,
        )
    """

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """List of tool definition dicts."""
        return ALL_DEFINITIONS

    async def handler(
        self,
        session: VoiceSession,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Unified tool handler for all input tools."""
        dispatch = {
            "move_mouse": self._move_mouse,
            "click": self._click,
            "type_text": self._type_text,
            "press_key": self._press_key,
            "scroll": self._scroll,
        }
        fn = dispatch.get(name)
        if fn is None:
            return f"Unknown tool: {name}"

        logger.info("%s(%s)", name, arguments)
        return fn(arguments)

    @staticmethod
    def _move_mouse(args: dict[str, Any]) -> str:
        pag = _get_pyautogui()
        x, y = int(args["x"]), int(args["y"])
        pag.moveTo(x, y)
        return f"Mouse moved to ({x}, {y})."

    @staticmethod
    def _click(args: dict[str, Any]) -> str:
        pag = _get_pyautogui()
        x, y = int(args["x"]), int(args["y"])
        button = args.get("button", "left")
        double = args.get("double", False)
        clicks = 2 if double else 1
        pag.click(x, y, clicks=clicks, button=button)
        action = "Double-clicked" if double else "Clicked"
        return f"{action} at ({x}, {y}) with {button} button."

    @staticmethod
    def _type_text(args: dict[str, Any]) -> str:
        pag = _get_pyautogui()
        text = str(args["text"])
        pag.typewrite(text, interval=0.02)
        return f"Typed: {text!r}"

    @staticmethod
    def _press_key(args: dict[str, Any]) -> str:
        pag = _get_pyautogui()
        key = str(args["key"])
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
        x = args.get("x")
        y = args.get("y")
        if x is not None and y is not None:
            pag.scroll(clicks, x=int(x), y=int(y))
            return f"Scrolled {clicks} at ({x}, {y})."
        pag.scroll(clicks)
        return f"Scrolled {clicks} at current position."
