"""Shared environment helpers for RoomKit examples."""

from __future__ import annotations

import os
import platform


def os_info() -> dict[str, str]:
    """Return OS-specific info for system prompts.

    Keys: ``os_name``, ``mod``, ``open_app_desc``, ``open_app_note``.
    """
    system = platform.system()
    if system == "Darwin":
        return {
            "os_name": "macOS",
            "mod": "command",
            "open_app_desc": 'open a macOS app by name (e.g. "Google Chrome")',
            "open_app_note": "do NOT use Spotlight — command+space does not work",
        }
    return {
        "os_name": "Linux",
        "mod": "ctrl",
        "open_app_desc": 'open an app by name (e.g. "google-chrome", "firefox")',
        "open_app_note": "use the executable name as it appears in the system",
    }


def auto_select_provider(env_var: str, label: str) -> str:
    """Pick ``"openai"`` or ``"gemini"`` based on available API keys.

    Checks ``env_var`` for an explicit override first, then falls back
    to whichever key is present.  Prompts interactively if both are set.
    """
    forced = os.environ.get(env_var, "").lower()
    if forced in ("openai", "gemini"):
        return forced

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_gemini = bool(os.environ.get("GOOGLE_API_KEY"))

    if has_openai and has_gemini:
        print("Both OPENAI_API_KEY and GOOGLE_API_KEY are set.")
        print(f"Which provider for {label}?")
        print("  1) OpenAI")
        print("  2) Gemini")
        choice = input("Choice [1]: ").strip()
        return "gemini" if choice == "2" else "openai"
    if has_openai:
        return "openai"
    return "gemini"
