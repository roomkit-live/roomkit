"""Startup banner for the screen assistant example."""

from __future__ import annotations

from .prompt import LANG_NAMES


def print_banner(
    *,
    voice_choice: str,
    tool_choice: str,
    voice_name: str,
    lang: str,
    aec: object | None,
    denoiser: object | None,
    mute_mic: bool,
    auto_verify: bool,
    browser_mode: str,
    playwright_tool_count: int,
    omniview_url: str | None,
    vision_interval_ms: int,
) -> None:
    """Print the startup banner — pure formatting, no side effects on the kit."""
    voice_label = "OpenAI" if voice_choice == "openai" else "Gemini"
    tool_label = "OpenAI" if tool_choice == "openai" else "Gemini"
    pw_label = (
        f"playwright ({playwright_tool_count} tools)" if browser_mode == "playwright" else "vision"
    )
    omniview_label = f"OmniView @ {omniview_url}" if omniview_url else "off"

    print()
    print(f"Screen Assistant ({voice_label} Voice + {tool_label} Vision)")
    print("=" * 60)
    print(f"Voice: {voice_name} | Language: {LANG_NAMES.get(lang, lang)}")
    print(f"AEC: {'on' if aec else 'off'} | Denoiser: {'on' if denoiser else 'off'}")
    print(
        f"Interruption: {'off' if mute_mic else 'on'} | Auto-verify: {'on' if auto_verify else 'off'}"
    )
    print(f"Browser: {pw_label}")
    print(f"OmniView: {omniview_label}")
    print(f"Vision: every {vision_interval_ms}ms (diff-gated, silent injection)")
    print()
    tools_line2 = "       click_element, type_text, press_key, scroll"
    if omniview_url:
        tools_line2 += ", observe, click_result"
    print("Tools: list_screens, switch_screen, open_app, describe_screen,")
    print(tools_line2)
    print("Press Ctrl+C to stop.")
    print()
