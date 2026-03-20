"""Tool definitions for the screen assistant example."""

from __future__ import annotations

ACTION_TOOLS: set[str] = {"click_element", "type_text", "press_key", "scroll"}

LIST_SCREENS_TOOL: dict[str, object] = {
    "name": "list_screens",
    "description": (
        "List all available monitors/screens with their resolution and a "
        "short description of what is visible on each. Use this at the start "
        "to find the right monitor, or when the user says you're looking at "
        "the wrong screen."
    ),
    "parameters": {"type": "object", "properties": {}},
}

OPEN_APP_TOOL: dict[str, object] = {
    "name": "open_app",
    "description": (
        "Open a macOS application by name. More reliable than clicking "
        "dock icons or using Spotlight. Examples: 'Google Chrome', 'Safari', "
        "'Firefox', 'Finder', 'Terminal'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Application name exactly as it appears in /Applications.",
            },
        },
        "required": ["app_name"],
    },
}

SWITCH_SCREEN_TOOL: dict[str, object] = {
    "name": "switch_screen",
    "description": (
        "Switch to a different monitor by index. Use after list_screens "
        "to select the correct one. Index 0 = all monitors combined, "
        "1 = first monitor, 2 = second, etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "monitor": {
                "type": "integer",
                "description": "Monitor index from list_screens.",
            },
        },
        "required": ["monitor"],
    },
}
