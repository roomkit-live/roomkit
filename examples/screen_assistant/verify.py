"""Action verification heuristics for the screen assistant example.

Pure string-matching logic that assesses whether a tool action
succeeded based on the vision model's screen description.
"""

from __future__ import annotations


def build_verify_question(tool_name: str, args: dict[str, object]) -> str:
    """Build a targeted verification question for the vision model."""
    if tool_name == "click_result":
        element_id = args.get("element_id", "")
        return (
            f"Answer precisely: Did clicking element {element_id} produce a visible change? "
            "What app is in the foreground? Did a new window/page/menu appear? "
            "State the foreground app name and URL if visible."
        )
    if tool_name == "click_element":
        element = args.get("element", "")
        return (
            f"Answer precisely: Did clicking '{element}' produce a visible change? "
            "What app is in the foreground? Did a new window/page/menu appear? "
            "State the foreground app name and URL if visible."
        )
    if tool_name == "press_key":
        key = args.get("key", "")
        if "command+t" in str(key) or "ctrl+t" in str(key):
            return (
                "Did a NEW BROWSER TAB open? Is the address bar empty/focused? "
                "Or is the same page still showing? State the foreground app and "
                "whether this is a new tab or the same page as before."
            )
        if "command+l" in str(key) or "ctrl+l" in str(key):
            return (
                "Is the browser address bar now selected/highlighted? "
                "State what is in the address bar and the foreground app."
            )
        if "enter" in str(key).lower():
            return (
                "Did pressing Enter trigger a navigation or action? "
                "What page or content is now showing? State the foreground app and URL."
            )
        return (
            f"What changed after pressing '{key}'? "
            "State the foreground app and any visible changes."
        )
    if tool_name == "type_text":
        text = args.get("text", "")
        return (
            f"Is the text '{text}' visible on screen (in an input field or address bar)? "
            "Where was it typed? State the foreground app."
        )
    if tool_name == "scroll":
        return "Did the page content change after scrolling? What is now visible?"
    return "Describe the current screen state in one sentence."


def _has_negation(text: str, keyword: str) -> bool:
    """Check if *keyword* appears in *text* in a negative context.

    Looks 40 chars before AND after the keyword for negation patterns.
    """
    lower = text.lower()
    if keyword not in lower:
        return False
    idx = lower.index(keyword)
    before = lower[max(0, idx - 40) : idx]
    after = lower[idx : idx + len(keyword) + 40]
    context = before + " " + after
    negations = (
        "no ",
        "not ",
        "isn't",
        "is not",
        "there is no",
        "not visible",
        "not selected",
        "not highlighted",
        "not focused",
        "not open",
        "did not",
        "didn't",
        "cannot",
        "can't",
        "no browser",
        "**not",
        "**no",  # markdown bold negations from Gemini
    )
    return any(neg in context for neg in negations)


def assess_action_result(
    tool_name: str,
    args: dict[str, object],
    screen_desc: str,
) -> dict[str, str]:
    """Assess whether an action succeeded based on the screen description."""
    desc_lower = screen_desc.lower()

    # Check if we're still looking at the terminal (common failure mode)
    if "iterm" in desc_lower or "terminal" in desc_lower:
        if tool_name in ("press_key", "type_text") and tool_name != "click_element":
            return {
                "status": "FAILED",
                "verdict": "Action was sent to the TERMINAL, not the browser. The browser is not focused.",
                "suggestion": "Use open_app to bring the browser to the foreground first.",
            }

    if tool_name == "press_key":
        key = str(args.get("key", "")).lower()
        if "command+t" in key or "ctrl+t" in key:
            if (
                "new tab" in desc_lower and not _has_negation(screen_desc, "new tab")
            ) or "empty" in desc_lower:
                return {
                    "status": "OK",
                    "verdict": "New tab opened successfully.",
                    "suggestion": "",
                }
            return {
                "status": "FAILED",
                "verdict": "New tab did NOT open. Same page still showing.",
                "suggestion": "The browser may not be focused. Try open_app first.",
            }
        if "command+l" in key or "ctrl+l" in key:
            if (
                _has_negation(screen_desc, "address bar")
                or _has_negation(screen_desc, "selected")
                or "not selected" in desc_lower
                or "not highlighted" in desc_lower
                or "not focused" in desc_lower
            ):
                return {
                    "status": "FAILED",
                    "verdict": "Address bar is NOT selected.",
                    "suggestion": "The browser may not be in the foreground. Use open_app first.",
                }
            if "selected" in desc_lower or "highlighted" in desc_lower or "focused" in desc_lower:
                return {"status": "OK", "verdict": "Address bar is selected.", "suggestion": ""}
            return {
                "status": "UNCERTAIN",
                "verdict": "Could not confirm address bar is selected.",
                "suggestion": "Use describe_screen to verify before typing.",
            }
        if "enter" in key:
            if "did not" in desc_lower or "didn't" in desc_lower or "no navigation" in desc_lower:
                return {
                    "status": "FAILED",
                    "verdict": "Enter did not trigger navigation or any visible change.",
                    "suggestion": "The input may not have been focused, or the typed text was lost. Start over.",
                }

    if tool_name == "click_result":
        element_id = str(args.get("element_id", ""))
        if "could not find" in desc_lower or "not found" in desc_lower:
            return {
                "status": "FAILED",
                "verdict": f"Element {element_id} click did not produce a visible change.",
                "suggestion": "Run observe again to get fresh element IDs, then retry.",
            }
        return {"status": "OK", "verdict": f"Element {element_id} clicked.", "suggestion": ""}

    if tool_name == "click_element":
        element = str(args.get("element", "")).lower()
        if "could not find" in desc_lower or "not found" in desc_lower:
            return {
                "status": "FAILED",
                "verdict": f"Element '{element}' was not found on screen.",
                "suggestion": "Try describing the element differently, or use describe_screen to see what's visible.",
            }

    if tool_name == "type_text":
        text = str(args.get("text", "")).lower()
        words = [w for w in text.split() if len(w) > 3]
        found = any(w in desc_lower for w in words)
        if found and not _has_negation(screen_desc, words[0] if words else ""):
            return {
                "status": "OK",
                "verdict": f"Text '{args.get('text', '')}' appears on screen.",
                "suggestion": "",
            }
        return {
            "status": "UNCERTAIN",
            "verdict": "Could not confirm the typed text is visible.",
            "suggestion": "The input field may not have been focused. Use describe_screen to check.",
        }

    # Default
    return {"status": "OK", "verdict": "Action completed.", "suggestion": ""}
