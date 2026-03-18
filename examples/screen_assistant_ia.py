"""RoomKit — AI screen assistant with speech-to-speech voice.

Talk to an AI while it sees your screen. The framework handles
observation automatically:

- **Periodic vision** runs in the background and silently injects
  ``[Screen changed]`` context into the voice session when the
  screen changes significantly (diff-threshold gated).
- **Auto-verify** — every action tool (click, type, press, scroll)
  automatically captures a fresh screen description and appends it
  to the tool result. The LLM always knows what happened without
  needing to call ``describe_screen`` manually.
- **On-demand vision** — ``describe_screen`` is still available for
  targeted queries (with a question) or instant cached lookups
  (without a question).

Supports **OpenAI Realtime** or **Gemini Live** for voice, and
**OpenAI** or **Gemini** for vision.

Requirements:
    pip install roomkit[screen-capture,local-audio,gemini,sherpa-onnx]
    pip install roomkit[realtime-openai]   # for OpenAI voice
    pip install roomkit[realtime-gemini]   # for Gemini voice
    pip install roomkit[screen-input]      # for keyboard control
    pip install aec-audio-processing       # WebRTC echo cancellation

Run with (Gemini only):
    GOOGLE_API_KEY=... uv run python examples/screen_assistant_ia.py

Run with (OpenAI voice + Gemini vision):
    GOOGLE_API_KEY=... OPENAI_API_KEY=... VISION_TOOL=gemini \
        uv run python examples/screen_assistant_ia.py

Environment variables:
    GOOGLE_API_KEY       (required) Google API key
    OPENAI_API_KEY       (optional) OpenAI API key
    VOICE_PROVIDER       Force voice: openai | gemini (auto)
    VISION_TOOL          Force tool:  openai | gemini (auto)
    GEMINI_MODEL         Gemini speech model
    GEMINI_VOICE         Gemini voice preset (default: Aoede)
    GEMINI_VISION_MODEL  Vision model (default: gemini-3.1-flash-image-preview)
    OPENAI_MODEL         OpenAI speech model (default: gpt-realtime-1.5)
    OPENAI_VOICE         OpenAI voice preset (default: alloy)
    OPENAI_VISION_MODEL  OpenAI tool model (default: gpt-4o)
    SCALE                Capture scale 0.0-1.0 (default: 0.75)
    AEC                  Echo cancellation: webrtc | 0 (default: webrtc)
    DENOISE              Noise suppression: 1 | 0 (default: 1)
    MUTE_MIC             Mute mic during AI playback: 1 | 0 (default: 0)
    LANG_VOICE           Language (default: en)
    MONITOR              Monitor index: 1=primary (default: 1)
    VISION_INTERVAL      Vision interval in ms (default: 5000)
    DIFF_THRESHOLD       Screen diff threshold 0.0-1.0 (default: 0.15)
    AUTO_VERIFY          Auto-verify after actions: 1 | 0 (default: 1)
    BROWSER_MODE         Browser control: vision | playwright (default: vision)

Press Ctrl+C to stop.

When BROWSER_MODE=playwright, the example launches @playwright/mcp as a
stdio MCP server and exposes Playwright browser tools alongside the
screen tools. The agent can then use Playwright for precise DOM
interactions (clicking links by text, filling forms) while still using
vision tools for screen-level awareness.

NOTE: Playwright mode requires OpenAI voice (VOICE_PROVIDER=openai).
Gemini Live does not support the number of tool declarations that
Playwright MCP exposes and will disconnect with 1008 errors.

Requires: npx @playwright/mcp (installed globally or via npx).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from roomkit import (
    DescribeScreenTool,
    GeminiVisionConfig,
    GeminiVisionProvider,
    HookExecution,
    HookResult,
    HookTrigger,
    OpenAIVisionConfig,
    OpenAIVisionProvider,
    RealtimeVoiceChannel,
    RoomKit,
    ScreenInputTools,
    VideoChannel,
)
from roomkit.video.backends.screen import ScreenCaptureBackend
from roomkit.voice.backends.local import LocalAudioBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("screen_assistant_ia")

LANG_NAMES = {
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "nl": "Dutch",
    "en": "English",
}


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _auto_select(env_var: str, label: str) -> str:
    """Pick openai or gemini based on available keys and env override."""
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _playwright_prompt(browser_mode: str) -> str:
    """Return Playwright-specific system prompt section."""
    if browser_mode != "playwright":
        return ""
    return """

## Playwright browser tools (PREFERRED for web interaction)

You have Playwright browser tools available. **Use them for all web \
interactions** — they are far more reliable than click_element for \
clicking links, filling forms, and navigating pages.

Playwright manages its OWN browser — you do NOT need to open Chrome \
or any browser manually. Just call browser_navigate and Playwright \
opens the page directly. Do NOT use open_app, click_element, \
press_key, or type_text for any browser task.

Key Playwright tools:
- **browser_navigate(url)** — go to a URL (opens browser automatically)
- **browser_click(element, ref)** — click by element description or ref
- **browser_type(element, ref, text)** — type into a field
- **browser_snapshot** — get the page accessibility tree
- **browser_tab_list** — list open tabs
- **browser_tab_new** — open a new tab

Preferred workflow: \
browser_navigate("https://google.com") → \
browser_type(element="search box", text="roomkit conversation AI") → \
browser_click(element="Google Search button") → \
browser_snapshot to read results → \
browser_click(element="RoomKit link").

IMPORTANT: Do NOT use open_app, press_key('command+t'), type_text, or \
click_element for anything browser-related. Playwright handles \
everything. Use vision tools ONLY for desktop/native app questions.\
"""


def _build_system_prompt(lang: str, *, browser_mode: str = "vision") -> str:
    """Build system prompt."""
    lang_name = LANG_NAMES.get(lang, lang)
    lang_instruction = f"\nIMPORTANT: You MUST speak ONLY in {lang_name}." if lang != "en" else ""

    return f"""\
You are a professional IT support assistant helping users via voice. \
You can check the user's screen and type on their keyboard.{lang_instruction}

## Your mission

Help the user find RoomKit by searching "roomkit conversation AI" in \
their browser. Open a browser, search using the search engine, and \
guide them to click the right result. Start by checking their screen.

## Rules

- **Be concise.** One short sentence per step. No repetition.
- **Do NOT speak unless needed.** Only talk when giving the next step, \
answering a question, or confirming progress.
- **The user can interrupt you at any time.** Stop and listen.
- **Never repeat what you just said.**
- **React to screen changes.** When you receive a [Screen changed] \
notification showing the goal was reached (e.g. roomkit.live is open), \
immediately confirm it to the user. Do NOT stay silent when the task \
is complete — tell them right away.
- **When the user says they did something** (clicked, opened, navigated), \
ALWAYS call describe_screen immediately to get a fresh view. Never \
answer from memory or cached vision — the screen may have changed \
since the last update.

## Screen awareness

You receive silent **[Screen changed]** context updates when the screen \
changes significantly. These are injected as background context — you \
do NOT need to respond to them. They keep you informed so that when \
the user speaks or you need to act, you already know what's on screen.

Every action tool (click_element, type_text, press_key, scroll) \
automatically captures the screen after execution and includes the \
result in its response. You do NOT need to call describe_screen to \
verify — just read the tool result.

You can still call **describe_screen** for targeted queries \
(with a specific question) or to get the latest cached view (no query).

## Screen management

If what you see doesn't match what the user describes, the active \
monitor may be wrong. Call **list_screens** to see all monitors with \
descriptions, then **switch_screen(monitor)** to target the correct one.

## Taking actions

Ask permission once before acting. Available tools:
- **list_screens** — list all monitors with a description of each
- **switch_screen(monitor)** — switch to a different monitor index
- **open_app(app_name)** — open a macOS app by name (e.g. "Google Chrome")
- **click_element(element)** — click on a UI element by description. \
The tool finds it on screen automatically. Example: \
click_element("the Chrome icon in the taskbar")
- **press_key(key)** — press keys: 'enter', 'command+l' (macOS), 'command+t'
- **type_text(text)** — type into the focused field
- **scroll(clicks)** — scroll up (positive) or down (negative)

## CRITICAL: Never break existing navigation

When the user already has a browser open with content (a page, a meeting, \
a form), you MUST open a **new tab** first (command+t) before doing \
anything. NEVER type into the address bar of an existing tab — this \
would destroy the user's current page. \
Always: command+t → then command+l → then type.

If the browser is not open yet, use **open_app("Google Chrome")** to \
launch it (do NOT use Spotlight — command+space does not work). \
Then proceed with a new tab.

Typical workflow: \
describe_screen → check if browser is open → \
open_app("Google Chrome") if needed → \
press_key('command+t') to open a new tab → \
type_text('roomkit conversation AI') → \
press_key('enter') → read search results → click the right link.\
{_playwright_prompt(browser_mode)}\
"""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_aec(sample_rate: int, block_ms: int) -> object | None:
    """Build AEC provider based on AEC env var."""
    try:
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        logger.info("AEC enabled (WebRTC AEC3)")
        return WebRTCAECProvider(sample_rate=sample_rate)
    except ImportError:
        print("\n  >>> Install AEC: pip install aec-audio-processing <<<\n")
        return None
    return None


def _build_denoiser() -> object | None:
    """Build sherpa-onnx GTCRN denoiser based on DENOISE env var."""
    if os.environ.get("DENOISE", "1") == "0":
        return None
    model = os.environ.get("DENOISE_MODEL", "gtcrn_simple.onnx")
    try:
        from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
            SherpaOnnxDenoiserConfig,
            SherpaOnnxDenoiserProvider,
        )

        logger.info("Denoiser enabled (sherpa-onnx GTCRN, model=%s)", model)
        return SherpaOnnxDenoiserProvider(SherpaOnnxDenoiserConfig(model=model))
    except ImportError:
        logger.warning("sherpa-onnx not available (DENOISE=0 to skip)")
        return None


def _build_vision_provider(
    tool_choice: str,
    google_api_key: str,
) -> GeminiVisionProvider | OpenAIVisionProvider:
    """Build a single vision provider used for both periodic and on-demand analysis."""
    if tool_choice == "openai":
        from roomkit.video.vision.base import VisionProvider

        vision: VisionProvider = OpenAIVisionProvider(
            OpenAIVisionConfig(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url="https://api.openai.com/v1",
                model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o"),
                max_tokens=4096,
                detail="high",
            )
        )
    else:
        vision = GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=google_api_key,
                model=os.environ.get(
                    "GEMINI_VISION_MODEL",
                    "gemini-3.1-flash-image-preview",
                ),
                max_tokens=4096,
                prompt=(
                    "Describe what is shown on this screen in 2-3 sentences. "
                    "Focus on the FOREGROUND application. "
                    "Include application name, visible text, URLs, "
                    "and what the user appears to be doing. Be concise."
                ),
            )
        )
    return vision


def _build_voice_provider(voice_choice: str) -> object:
    """Build the realtime voice provider."""
    if voice_choice == "openai":
        from roomkit.providers.openai.realtime import OpenAIRealtimeProvider

        return OpenAIRealtimeProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-realtime-1.5"),
        )
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    return GeminiLiveProvider(
        api_key=os.environ["GOOGLE_API_KEY"],
        model=os.environ.get(
            "GEMINI_MODEL",
            "gemini-2.5-flash-native-audio-preview-12-2025",
        ),
    )


def _get_voice_name(voice_choice: str) -> str:
    """Get the voice preset for the chosen provider."""
    if voice_choice == "openai":
        return os.environ.get("OPENAI_VOICE", "alloy")
    return os.environ.get("GEMINI_VOICE", "Aoede")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_ACTION_TOOLS = {"click_element", "type_text", "press_key", "scroll"}


def _build_verify_question(tool_name: str, args: dict[str, object]) -> str:
    """Build a targeted verification question for the vision model."""
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
    """Check if a keyword appears in text in a negative context.

    Looks both before AND after the keyword for negation patterns.
    """
    lower = text.lower()
    if keyword not in lower:
        return False
    idx = lower.index(keyword)
    # Check 40 chars before and 40 chars after the keyword
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


def _assess_action_result(
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
            # Check for explicit negation anywhere in the description
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

    if tool_name == "click_element":
        element = str(args.get("element", "")).lower()
        if "could not find" in desc_lower or "not found" in desc_lower:
            return {
                "status": "FAILED",
                "verdict": f"Element '{element}' was not found on screen.",
                "suggestion": "Try describing the element differently, or use describe_screen to see what's visible.",
            }
        # If clicking a link, check if navigation happened
        if "link" in element or "result" in element:
            if "search results" in desc_lower or "google" in desc_lower:
                return {
                    "status": "UNCERTAIN",
                    "verdict": "Still on search results page — the link click may not have navigated.",
                    "suggestion": "The click may have missed. Try clicking again or use describe_screen to check the URL.",
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


async def main() -> None:
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        print("GOOGLE_API_KEY is required.")
        print("  GOOGLE_API_KEY=... uv run python examples/screen_assistant_ia.py")
        return

    # --- Provider selection --------------------------------------------------
    voice_choice = _auto_select("VOICE_PROVIDER", "voice")
    tool_choice = _auto_select("VISION_TOOL", "vision tool")

    if voice_choice == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for OpenAI voice.")
        return
    if tool_choice == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for OpenAI vision tool.")
        return

    lang = os.environ.get("LANG_VOICE", os.environ.get("LANG", "en")).lower()[:2]
    auto_verify = os.environ.get("AUTO_VERIFY", "1") != "0"

    # --- Telemetry: token cost tracking via RoomKit's telemetry system ------
    from roomkit.telemetry.base import Attr, SpanKind, TelemetryProvider
    from roomkit.telemetry.console import ConsoleTelemetryProvider

    class CostTrackingTelemetry(ConsoleTelemetryProvider):
        """Extends console telemetry to accumulate token costs."""

        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.totals: dict[str, int] = {
                "vision_calls": 0,
                "vision_prompt_tokens": 0,
                "vision_completion_tokens": 0,
                "realtime_input_tokens": 0,
                "realtime_output_tokens": 0,
            }

        def end_span(self, span_id: str, **kwargs: object) -> None:
            span = self._spans.get(span_id)
            if span is not None:
                attrs = {**span.attributes, **(kwargs.get("attributes") or {})}  # type: ignore[arg-type]
                inp = attrs.get(Attr.LLM_INPUT_TOKENS, 0)
                out = attrs.get(Attr.LLM_OUTPUT_TOKENS, 0)
                if inp or out:
                    # Realtime turn spans carry token usage
                    if span.kind == SpanKind.REALTIME_TURN:
                        self.totals["realtime_input_tokens"] += int(inp)
                        self.totals["realtime_output_tokens"] += int(out)
            super().end_span(span_id, **kwargs)  # type: ignore[arg-type]

        def record_metric(self, name: str, value: float, **kwargs: object) -> None:
            if name == "roomkit.realtime.input_tokens":
                self.totals["realtime_input_tokens"] += int(value)
            elif name == "roomkit.realtime.output_tokens":
                self.totals["realtime_output_tokens"] += int(value)
            super().record_metric(name, value, **kwargs)  # type: ignore[arg-type]

        def print_summary(self) -> None:
            v_total = self.totals["vision_prompt_tokens"] + self.totals["vision_completion_tokens"]
            r_total = self.totals["realtime_input_tokens"] + self.totals["realtime_output_tokens"]
            print()
            print("Session Cost Summary")
            print("-" * 40)
            print(f"  Vision API calls:       {self.totals['vision_calls']}")
            print(
                f"  Vision tokens:          {v_total:,} ({self.totals['vision_prompt_tokens']:,} in / {self.totals['vision_completion_tokens']:,} out)"
            )
            print(
                f"  Realtime voice tokens:  {r_total:,} ({self.totals['realtime_input_tokens']:,} in / {self.totals['realtime_output_tokens']:,} out)"
            )
            print(f"  Total tokens:           {v_total + r_total:,}")

    cost_telemetry = CostTrackingTelemetry()
    kit = RoomKit(telemetry=cost_telemetry)

    # --- Shared vision provider ----------------------------------------------
    monitor = int(os.environ.get("MONITOR", "1"))
    vision = _build_vision_provider(tool_choice, google_api_key)
    screen_tool = DescribeScreenTool(vision, monitor=monitor)

    # --- Vision analysis wrapper (tracks cost) --------------------------------
    async def _analyze_with_cost(query: str, monitor_idx: int | None = None) -> str:
        """Capture + analyze via vision, tracking token usage."""
        from roomkit.video.vision.screen_tool import capture_screen_frame

        idx = monitor_idx if monitor_idx is not None else monitor
        frame = capture_screen_frame(idx)
        if frame is None:
            return "No screen frame available."
        result = await vision.analyze_frame(frame, prompt=query)
        cost_telemetry.totals["vision_calls"] += 1
        usage = result.metadata.get("usage", {})
        cost_telemetry.totals["vision_prompt_tokens"] += usage.get("prompt_tokens", 0)
        cost_telemetry.totals["vision_completion_tokens"] += usage.get("completion_tokens", 0)
        return result.description or "Could not analyze the screen."

    # --- Periodic screen vision (background) ---------------------------------
    vision_interval = int(os.environ.get("VISION_INTERVAL", "5000"))
    scale = float(os.environ.get("SCALE", "0.75"))
    diff_threshold = float(os.environ.get("DIFF_THRESHOLD", "0.15"))
    screen_backend = ScreenCaptureBackend(
        monitor=monitor,
        fps=2,
        scale=scale,
        diff_threshold=diff_threshold,
    )

    video_channel = VideoChannel(
        "video-screen",
        backend=screen_backend,
        vision=vision,
        vision_interval_ms=vision_interval,
    )
    kit.register_channel(video_channel)

    # --- Vision cache --------------------------------------------------------
    latest_vision: dict[str, str] = {"description": "", "previous": ""}
    frame_count = 0

    import re as _re

    def _extract_key_terms(description: str) -> set[str]:
        """Extract app names and URLs from a vision description."""
        desc_lower = description.lower()
        terms: set[str] = set()
        for app in (
            "chrome",
            "safari",
            "firefox",
            "iterm",
            "terminal",
            "finder",
            "code",
            "vscode",
            "slack",
            "discord",
            "teams",
            "outlook",
            "github",
            "google",
            "roomkit",
        ):
            if app in desc_lower:
                terms.add(app)
        for url in _re.findall(r"[\w-]+\.[\w.-]+", description):
            terms.add(url.lower())
        return terms

    # --- Hook: periodic vision → smart injection (AFTER_BROADCAST) ---------
    # Registered as an async hook on video_vision_result. Compares key terms
    # to detect significant changes (new app, new URL) and injects with
    # silent=False so the agent proactively reacts.

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="vision_observer")
    async def vision_observer_hook(event: object, ctx: object) -> None:
        """Observe broadcast events and inject vision updates for transcriptions.

        When the user speaks (transcription event from voice channel), the
        periodic vision may have new info. This hook ensures the agent's
        context stays current after each broadcast cycle.
        """
        # Also handle video_vision_result framework events
        pass  # AFTER_BROADCAST fires for room events, not framework events

    @kit.on("video_vision_result")
    async def on_vision(event: object) -> None:
        nonlocal frame_count
        data = event.data  # type: ignore[attr-defined]
        if event.room_id != "screen-assistant":  # type: ignore[attr-defined]
            return

        description = data.get("description", "")
        if not description:
            return

        frame_count += 1
        previous = latest_vision["description"]
        latest_vision["previous"] = previous
        latest_vision["description"] = description

        elapsed = data.get("elapsed_ms", 0)
        short = description[:150] + "..." if len(description) > 150 else description
        logger.info("[Vision %d] (%dms) %s", frame_count, elapsed, short)

        prev_terms = _extract_key_terms(previous) if previous else set()
        curr_terms = _extract_key_terms(description)
        new_terms = curr_terms - prev_terms
        significant = bool(new_terms)

        if previous and not significant and previous[:80] == description[:80]:
            logger.debug("[Vision %d] No meaningful change, skipping injection", frame_count)
            return

        sessions = voice_channel.get_room_sessions("screen-assistant")
        if not sessions:
            return

        if significant:
            context = f"[Screen changed — new: {', '.join(sorted(new_terms))}] {description}"
            logger.info("[Vision %d] Significant change detected: %s", frame_count, new_terms)
        else:
            context = f"[Screen changed] {description}"

        for session in sessions:
            try:
                await voice_channel.inject_text(
                    session,
                    context,
                    role="user",
                    silent=not significant,
                )
            except Exception:
                logger.exception("[Vision %d] Failed to inject", frame_count)

    # --- Playwright MCP (optional) -------------------------------------------
    browser_mode = os.environ.get("BROWSER_MODE", "vision").lower()
    playwright_mcp: object | None = None
    playwright_tools: list[dict[str, object]] = []
    playwright_tool_names: set[str] = set()
    _pw_cleanup: list[object] = []  # context managers to close on shutdown

    if browser_mode == "playwright" and voice_choice != "openai":
        print("WARNING: Playwright mode requires OpenAI voice.")
        print("Gemini Live cannot handle Playwright's tool declarations.")
        print("Falling back to vision mode. Use VOICE_PROVIDER=openai for Playwright.\n")
        browser_mode = "vision"

    if browser_mode == "playwright":
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            logger.info("Starting Playwright MCP server via npx...")

            _pw_params = StdioServerParameters(
                command="npx",
                args=["@playwright/mcp"],
            )
            _pw_transport_ctx = stdio_client(_pw_params)
            _pw_streams = await _pw_transport_ctx.__aenter__()
            _pw_read, _pw_write = _pw_streams[0], _pw_streams[1]
            _pw_session = ClientSession(_pw_read, _pw_write)
            await _pw_session.__aenter__()
            await _pw_session.initialize()

            def _clean_schema(schema: dict[str, object]) -> dict[str, object]:
                """Deep-clean a JSON Schema for Gemini compatibility.

                Gemini FunctionDeclaration rejects non-standard keys like
                $schema, additionalProperties, additional_properties, etc.
                """
                _STRIP_KEYS = {
                    "$schema",
                    "additionalProperties",
                    "additional_properties",
                    "default",
                    "title",
                }
                out: dict[str, object] = {}
                for k, v in schema.items():
                    if k in _STRIP_KEYS:
                        continue
                    if isinstance(v, dict):
                        out[k] = _clean_schema(v)
                    elif isinstance(v, list):
                        out[k] = [
                            _clean_schema(item) if isinstance(item, dict) else item for item in v
                        ]
                    else:
                        out[k] = v
                return out

            pw_result = await _pw_session.list_tools()
            for tool in pw_result.tools:
                params = _clean_schema(dict(tool.inputSchema)) if tool.inputSchema else {}
                playwright_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": params,
                    }
                )
                playwright_tool_names.add(tool.name)

            playwright_mcp = _pw_session
            _pw_cleanup.extend([_pw_session, _pw_transport_ctx])
            logger.info(
                "Playwright MCP ready — %d tools: %s",
                len(playwright_tools),
                sorted(playwright_tool_names),
            )
        except Exception:
            logger.exception("Failed to start Playwright MCP — falling back to vision mode")
            browser_mode = "vision"

    # --- System prompt (built after browser_mode is resolved) ---------------
    system_prompt = _build_system_prompt(lang, browser_mode=browser_mode)

    # --- Hook: tool execution via ON_TOOL_CALL ------------------------------
    input_tools = ScreenInputTools(
        vision=vision,
        monitor=monitor,
    )

    all_tools = [
        screen_tool.definition,
        LIST_SCREENS_TOOL,
        SWITCH_SCREEN_TOOL,
        OPEN_APP_TOOL,
        *input_tools.definitions,
        *playwright_tools,
    ]

    async def _list_screens() -> str:
        """Capture each monitor and describe it via vision."""
        import mss

        with mss.mss() as sct:
            monitors = sct.monitors
        if len(monitors) <= 1:
            return "Only 1 monitor detected (the combined virtual screen)."

        lines = [f"{len(monitors) - 1} monitor(s) detected:\n"]
        for idx in range(1, len(monitors)):
            mon = monitors[idx]
            size = f"{mon['width']}x{mon['height']}"
            active = " (ACTIVE)" if idx == monitor else ""
            try:
                desc = await _analyze_with_cost(
                    "Describe what is on this screen in one sentence. "
                    "Include the main application, visible windows, and dock if present.",
                    monitor_idx=idx,
                )
            except Exception:
                desc = "(could not analyze)"
            lines.append(f"  [{idx}] {size}{active}: {desc}")

        return "\n".join(lines)

    def _switch_screen(new_monitor: int) -> str:
        """Switch the active monitor for capture and tools."""
        nonlocal monitor, screen_tool, input_tools
        import mss

        with mss.mss() as sct:
            count = len(sct.monitors)
        if new_monitor < 0 or new_monitor >= count:
            return f"Invalid monitor {new_monitor}. Available: 0-{count - 1}."

        monitor = new_monitor
        screen_tool = DescribeScreenTool(vision, monitor=monitor)
        input_tools = ScreenInputTools(vision=vision, monitor=monitor)
        screen_backend._monitor = monitor
        logger.info("Switched to monitor %d", monitor)
        return f"Switched to monitor {monitor}. All tools now target this screen."

    # --- Tool audit log -------------------------------------------------------
    # Records every tool call with input, output, timing, and screen state.
    # Written as JSONL to /tmp/screen_ai/<timestamp>_session.jsonl
    import json as _audit_json
    import time as _audit_time
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    _audit_dir = _Path("/tmp/screen_ai")
    _audit_dir.mkdir(parents=True, exist_ok=True)
    _audit_file = _audit_dir / f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_session.jsonl"
    _audit_entries: list[dict[str, object]] = []

    def _audit_log(entry: dict[str, object]) -> None:
        """Append an audit entry and write to disk."""
        _audit_entries.append(entry)
        with open(_audit_file, "a") as f:
            f.write(_audit_json.dumps(entry, default=str) + "\n")

    def _audit_print_summary() -> None:
        """Print a summary of tool calls at the end of the session."""
        if not _audit_entries:
            print("\nNo tool calls recorded.")
            return
        print(f"\nTool Audit Log ({_audit_file})")
        print("=" * 70)
        for i, e in enumerate(_audit_entries, 1):
            status = "OK" if not e.get("error") else "ERROR"
            duration = e.get("duration_ms", 0)
            result_preview = str(e.get("result", ""))[:120]
            if len(str(e.get("result", ""))) > 120:
                result_preview += "..."
            print(
                f"  {i:2d}. [{status}] {e['tool']}({_audit_json.dumps(e.get('input', {}), default=str)[:80]})"
            )
            print(f"      → {result_preview}")
            print(f"      ({duration:.0f}ms)")
        # Stats
        tools_used = {}
        total_ms = 0.0
        for e in _audit_entries:
            t = str(e["tool"])
            tools_used[t] = tools_used.get(t, 0) + 1
            total_ms += float(e.get("duration_ms", 0))
        print(f"\n  Total: {len(_audit_entries)} calls, {total_ms:.0f}ms")
        print(f"  Tools: {', '.join(f'{k}({v})' for k, v in sorted(tools_used.items()))}")

    @kit.hook(
        HookTrigger.ON_TOOL_CALL,
        execution=HookExecution.SYNC,
        name="screen_tool_handler",
    )
    async def screen_tool_hook(event: object, ctx: object) -> HookResult:
        """Handle all tool calls via the hook pipeline with audit logging."""
        from roomkit.models.event import RoomEvent

        tool_event = event  # type: ignore[assignment]
        name = tool_event.name  # type: ignore[attr-defined]
        arguments = tool_event.arguments  # type: ignore[attr-defined]
        session = tool_event.session  # type: ignore[attr-defined]

        _t0 = _audit_time.monotonic()
        result: str
        error: str | None = None

        if name == "list_screens":
            result = await _list_screens()

        elif name == "switch_screen":
            result = _switch_screen(int(arguments.get("monitor", 1)))

        elif name == "open_app":
            import subprocess

            app_name = str(arguments.get("app_name", ""))
            logger.info("open_app(%r)", app_name)
            try:
                subprocess.run(
                    ["open", "-a", app_name],
                    check=True,
                    capture_output=True,
                    timeout=5,
                )
                await asyncio.sleep(2.0)  # wait for app to come to foreground
                # Activate the app to ensure it has focus
                subprocess.run(
                    ["osascript", "-e", f'tell application "{app_name}" to activate'],
                    capture_output=True,
                    timeout=3,
                )
                await asyncio.sleep(0.5)
                verify = await _analyze_with_cost(
                    f"Which application is now in the foreground? Is {app_name} focused?"
                )
                latest_vision["description"] = verify
                app_lower = app_name.lower()
                is_focused = app_lower in verify.lower() or "chrome" in verify.lower()
                status = "OK" if is_focused else "FAILED"
                result = (
                    f"ACTION: open_app({app_name})\n"
                    f"STATUS: {status}\n"
                    f"SCREEN: {verify[:200]}\n"
                    f"VERDICT: {app_name} {'is' if is_focused else 'is NOT'} in the foreground."
                )
                if not is_focused:
                    result += f"\nSUGGESTION: {app_name} may not have opened. Try again or check the app name."
            except subprocess.CalledProcessError as exc:
                result = f"Failed to open {app_name}: {exc.stderr.decode().strip()}"
            except FileNotFoundError:
                result = f"Application '{app_name}' not found."

        elif name == "describe_screen":
            query = str(arguments.get("query", ""))
            # Always take a fresh screenshot — cached vision gets stale fast
            if not query:
                query = "Describe the current screen: foreground app, visible content, URLs."
            result = await _analyze_with_cost(query)
            latest_vision["description"] = result

        elif name in _ACTION_TOOLS:
            # Execute the action tool
            action_result = await input_tools.handler(name, arguments)

            # Structured auto-verify: ask vision a targeted question
            # about whether the action succeeded, then format as verdict.
            if auto_verify:
                try:
                    await asyncio.sleep(0.5)
                    # Build a verify question specific to the action
                    verify_q = _build_verify_question(name, arguments)
                    screen_desc = await _analyze_with_cost(verify_q)
                    latest_vision["description"] = screen_desc
                    verdict = _assess_action_result(name, arguments, screen_desc)
                    result = (
                        f"ACTION: {name}({_audit_json.dumps(dict(arguments), default=str)})\n"
                        f"STATUS: {verdict['status']}\n"
                        f"SCREEN: {screen_desc[:200]}\n"
                        f"VERDICT: {verdict['verdict']}"
                    )
                    if verdict["status"] == "FAILED":
                        result += f"\nSUGGESTION: {verdict['suggestion']}"
                        result += "\nDo NOT proceed with the next step — fix this first."
                    logger.info(
                        "Auto-verify %s: %s → %s", name, verdict["status"], verdict["verdict"]
                    )
                except Exception:
                    logger.exception("Auto-verify failed after %s", name)
                    result = action_result
            else:
                result = action_result
        elif playwright_mcp is not None and name in playwright_tool_names:
            # Route to Playwright MCP server. Truncate results to avoid
            # blowing through the OpenAI realtime TPM limit — full DOM
            # snapshots can be 25k+ chars (≈14k tokens).
            import json as _json

            _PW_MAX_RESULT = 4000
            logger.info("Playwright MCP tool: %s(%s)", name, arguments)
            pw_result = await playwright_mcp.call_tool(name, arguments)  # type: ignore[union-attr]
            texts = []
            for content in pw_result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
            raw = "\n".join(texts) if texts else _json.dumps({"status": "ok"})
            if len(raw) > _PW_MAX_RESULT:
                result = (
                    raw[:_PW_MAX_RESULT] + f"\n... [truncated {len(raw)} → {_PW_MAX_RESULT} chars]"
                )
            else:
                result = raw

        else:
            return HookResult.allow()  # Unknown tool — let other hooks handle it

        # Audit log
        _audit_log(
            {
                "tool": name,
                "input": dict(arguments),
                "result": result[:500],
                "result_full_length": len(result),
                "error": error,
                "duration_ms": round((_audit_time.monotonic() - _t0) * 1000),
                "timestamp": _dt.now().isoformat(),
                "screen_after": latest_vision.get("description", "")[:200],
            }
        )

        # Return the result via metadata (simplified — no RoomEvent needed)
        return HookResult(action="allow", metadata={"result": result})

    # --- Speech-to-speech voice ----------------------------------------------
    sample_rate = 24000
    block_ms = 20

    provider = _build_voice_provider(voice_choice)

    aec = _build_aec(sample_rate, block_ms)
    denoiser = _build_denoiser()

    pipeline = None
    if aec or denoiser:
        from roomkit.voice.pipeline.config import AudioPipelineConfig

        pipeline = AudioPipelineConfig(aec=aec, denoiser=denoiser)

    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env == "1" if mute_env is not None else False
    audio_backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
        pipeline=pipeline,
    )

    voice_name = _get_voice_name(voice_choice)
    voice_channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=audio_backend,
        system_prompt=system_prompt,
        voice=voice_name,
        input_sample_rate=sample_rate,
        tools=all_tools,
        # No tool_handler — tools are handled via ON_TOOL_CALL hook
        mute_on_tool_call=True,
    )
    kit.register_channel(voice_channel)

    # --- Room + sessions -----------------------------------------------------
    await kit.create_room(room_id="screen-assistant")
    await kit.attach_channel("screen-assistant", "video-screen")
    await kit.attach_channel("screen-assistant", "voice")

    video_session = await kit.connect_video("screen-assistant", "local-user", "video-screen")
    await screen_backend.start_capture(video_session)

    provider_config = {}
    if voice_choice == "gemini":
        provider_config = {
            "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
            "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
            "silence_duration_ms": 1500,
        }

    await voice_channel.start_session(
        "screen-assistant",
        "local-user",
        connection=None,
        metadata={"provider_config": provider_config} if provider_config else None,
    )

    # --- Banner --------------------------------------------------------------
    voice_label = "OpenAI" if voice_choice == "openai" else "Gemini"
    tool_label = "OpenAI" if tool_choice == "openai" else "Gemini"
    verify_label = "on" if auto_verify else "off"
    print()
    print(f"Screen Assistant ({voice_label} Voice + {tool_label} Vision)")
    print("=" * 60)
    print(f"Voice: {voice_name} | Language: {LANG_NAMES.get(lang, lang)}")
    print(f"AEC: {'on' if aec else 'off'} | Denoiser: {'on' if denoiser else 'off'}")
    pw_label = (
        f"playwright ({len(playwright_tools)} tools)" if browser_mode == "playwright" else "vision"
    )
    print(f"Interruption: {'off' if mute_mic else 'on'} | Auto-verify: {verify_label}")
    print(f"Browser: {pw_label}")
    print(f"Vision: every {vision_interval}ms (diff-gated, silent injection)")
    print()
    print("Tools: list_screens, switch_screen, open_app, describe_screen,")
    print("       click_element, type_text, press_key, scroll")
    print("Press Ctrl+C to stop.")
    print()

    # --- Keep running until Ctrl+C -------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup -------------------------------------------------------------
    logger.info("Stopping...")
    await screen_backend.stop_capture(video_session)
    # Close Playwright MCP session and transport before kit shutdown
    for ctx in _pw_cleanup:
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            pass
    await kit.close()
    cost_telemetry.print_summary()
    _audit_print_summary()
    logger.info("Done. Vision analyzed %d frames.", frame_count)


if __name__ == "__main__":
    asyncio.run(main())
