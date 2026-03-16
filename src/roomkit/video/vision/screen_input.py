"""Keyboard control tools for AI screen assistants.

Lets an AI agent type text and press keys on the user's computer
via ``pyautogui``.

Requires ``pyautogui``::

    pip install roomkit[screen-input]

Example::

    from roomkit.video.vision import ScreenInputTools

    input_tools = ScreenInputTools()

    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=backend,
        tools=[*input_tools.definitions],
        tool_handler=input_tools.handler,
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

TYPE_TEXT_TOOL: dict[str, Any] = {
    "name": "type_text",
    "description": (
        "Type text using the keyboard. The text is typed character by character "
        "into whatever field or application currently has focus. "
        "Make sure the right field has focus first (use press_key with "
        "'ctrl+l' for browser address bar, 'tab' to move between fields, etc.)."
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
        "For combos use + separator: 'ctrl+a', 'ctrl+c', 'alt+tab', 'ctrl+l'."
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

ALL_DEFINITIONS: list[dict[str, Any]] = [
    TYPE_TEXT_TOOL,
    PRESS_KEY_TOOL,
]


# ---------------------------------------------------------------------------
# pyautogui wrapper
# ---------------------------------------------------------------------------


def _get_pyautogui() -> Any:
    """Lazy-import pyautogui with a clear error message."""
    try:
        import pyautogui

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
    """Keyboard control tools for AI agents.

    Provides tool definitions and a unified handler for typing text
    and pressing keys.

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
        """Unified tool handler for keyboard tools."""
        dispatch = {
            "type_text": self._type_text,
            "press_key": self._press_key,
        }
        fn = dispatch.get(name)
        if fn is None:
            return f"Unknown tool: {name}"

        logger.info("%s(%s)", name, arguments)
        return fn(arguments)

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
