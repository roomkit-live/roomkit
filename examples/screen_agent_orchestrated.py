"""RoomKit — Orchestrated screen assistant with OmniView vision.

Three-agent architecture with OmniView for precise UI element detection:

    User ←→ Voice Agent (RealtimeVoiceChannel)
                │  reads session log → narrates progress
                │  delegate_task → exec agent
                │
    Session Log ← all agents write structured events →
                │
                ├── Exec Agent (Sonnet 4.6 / GPT-4o)
                │     plans steps, calls tools, logs each step
                │     uses OmniView for precise element detection
                │
                └── OmniView API (http://GPU:8100)
                      YOLO + EasyOCR + Florence-2
                      returns numbered elements with bboxes
                      exec agent clicks by element ID

Requirements:
    pip install roomkit[screen-capture,local-audio,screen-input]
    pip install roomkit[realtime-openai]   # for OpenAI voice
    pip install roomkit[realtime-gemini]   # for Gemini voice
    pip install roomkit[anthropic]         # for Anthropic execution agent
    pip install roomkit[openai]            # for OpenAI execution agent

    OmniView service running on a GPU server (see github.com/sboily/omniview)

Run with:
    OMNIVIEW_URL=http://192.168.50.169:8100 \\
    ANTHROPIC_API_KEY=... \\
        python examples/screen_agent_orchestrated.py

Environment variables:
    OMNIVIEW_URL         (required) OmniView API URL
    GOOGLE_API_KEY       (optional) Google API key (voice)
    OPENAI_API_KEY       (optional) OpenAI API key (voice or exec)
    ANTHROPIC_API_KEY    (optional) Anthropic API key (exec agent)
    VOICE_PROVIDER       Force voice: openai | gemini (auto)
    EXEC_PROVIDER        Execution agent: anthropic | openai | gemini (auto)
    EXEC_MODEL           Execution model (default: per provider)
    MUTE_MIC             Mute mic during playback: 1 | 0 (default: 1)
    MONITOR              Monitor index (default: 1)
    LANG_VOICE           Language (default: fr)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("screen_orchestrated")
logging.getLogger("roomkit.core.event_router").setLevel(logging.ERROR)

from roomkit import (
    Agent,
    ChannelCategory,
    ChannelRecordingConfig,
    ConversationState,
    DelegateHandler,
    GeminiVisionConfig,
    GeminiVisionProvider,
    HookExecution,
    HookTrigger,
    JSONLToolAuditor,
    MediaRecordingConfig,
    RealtimeVoiceChannel,
    RoomKit,
    ScreenInputTools,
    SlidingWindowMemory,
    StatusBus,
    VideoChannel,
    WaitForIdleDelivery,
    audit_tool_handler,
    build_delegate_tool,
    set_conversation_state,
)
from roomkit.models.hook import HookResult
from roomkit.video.vision.screen_tool import capture_screen_frame
from roomkit.voice.backends.local import LocalAudioBackend

# ---------------------------------------------------------------------------
# OmniView client — calls the GPU vision service
# ---------------------------------------------------------------------------


class OmniViewClient:
    """Client for the OmniView API (screenshot → UI elements with bboxes)."""

    def __init__(self, base_url: str, monitor: int = 1) -> None:
        self.base_url = base_url.rstrip("/")
        self.monitor = monitor
        self.last_elements: list[dict[str, object]] = []

    def _capture_b64(self) -> str | None:
        """Capture screen as base64 PNG."""
        import base64
        import io

        frame = capture_screen_frame(self.monitor)
        if frame is None:
            return None
        from PIL import Image

        img = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    async def parse(self) -> dict[str, object]:
        """Capture screen → OmniView /parse → all elements."""
        import urllib.request

        b64 = self._capture_b64()
        if b64 is None:
            return {"status": "error", "error": "No screen frame"}

        req_body = json.dumps({"image": b64}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/parse",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=30),  # noqa: ASYNC210
        )
        result = json.loads(resp.read())
        self.last_elements = result.get("elements", [])
        return result

    async def locate(self, query: str) -> dict[str, object]:
        """Capture screen → OmniView /locate → best matching element."""
        import urllib.request

        b64 = self._capture_b64()
        if b64 is None:
            return {"found": False, "error": "No screen frame"}

        req_body = json.dumps({"image": b64, "query": query}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/locate",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=30),  # noqa: ASYNC210
        )
        result = json.loads(resp.read())
        return result

    def get_element_by_id(self, element_id: int) -> dict[str, object] | None:
        """Get an element from the last parse result by ID."""
        for el in self.last_elements:
            if el.get("id") == element_id:
                return el
        return None


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------


def _auto_select_voice() -> str:
    forced = os.environ.get("VOICE_PROVIDER", "").lower()
    if forced in ("openai", "gemini"):
        return forced
    return "openai" if os.environ.get("OPENAI_API_KEY") else "gemini"


def _build_exec_provider() -> object:
    choice = os.environ.get("EXEC_PROVIDER", "").lower()
    if not choice:
        if os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        else:
            choice = "gemini"

    if choice == "anthropic":
        from roomkit.providers.anthropic.ai import AnthropicAIProvider, AnthropicConfig

        return AnthropicAIProvider(
            AnthropicConfig(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                model=os.environ.get("EXEC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=2048,
            )
        )
    if choice == "openai":
        from roomkit.providers.openai.ai import OpenAIAIProvider, OpenAIConfig

        return OpenAIAIProvider(
            OpenAIConfig(
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.environ.get("EXEC_MODEL", "gpt-4o"),
                max_tokens=2048,
            )
        )
    from roomkit import GeminiAIProvider, GeminiConfig

    return GeminiAIProvider(
        GeminiConfig(
            api_key=os.environ["GOOGLE_API_KEY"],
            model=os.environ.get("EXEC_MODEL", "gemini-2.0-flash"),
            max_tokens=2048,
        )
    )


def _build_voice_provider(voice_choice: str) -> object:
    if voice_choice == "openai":
        from roomkit.providers.openai.realtime import OpenAIRealtimeProvider

        return OpenAIRealtimeProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-realtime-1.5"),
        )
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    return GeminiLiveProvider(
        api_key=os.environ["GOOGLE_API_KEY"],
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
    )


# ---------------------------------------------------------------------------
# Exec agent system prompt
# ---------------------------------------------------------------------------


def _exec_system_prompt() -> str:
    return """\
You are a computer automation agent. You have HIGH-LEVEL tools.

## Two vision modes

- **observe()** — detect UI elements with bounding boxes. Use BEFORE clicking.
  Returns element IDs and positions. For ACTING on screen.
- **read_screen(query?)** — understand what is on screen. Use to check state,
  read content, verify results, summarize pages.
  Returns natural language description. For UNDERSTANDING the screen.

## Action tools

- **search_google(query)** — Opens Chrome, new tab, searches Google.
  Returns result links with IDs. ONE call does everything.
- **click_result(element_id)** — Clicks a result by ID from search_google().
- **navigate(url)** — Opens a URL in a new tab.
- **log_progress(message)** — Reports progress to the user.

## CONTEXT section

Your task may include a CONTEXT section with previous activity. READ IT.
If the context shows a page is already open, do NOT search or navigate again.
Start with read_screen() to see the current state, then act on what's there.

## CRITICAL: Do NOT open new tabs or navigate if already on the right page

If read_screen() or observe() shows you're already on the page you need:
- Do NOT call search_google or navigate — you're already there!
- Just use observe() to find the element, then click_result() to click it.
- Opening a new tab when you're already on the right page is WRONG.

## Workflow

1. Read the CONTEXT — check if the task is partially done
2. read_screen() — understand current screen state
3. If already on the right page → observe() + click_result()
4. If NOT on the right page → search_google or navigate
5. read_screen() — verify the action worked
6. log_progress("Done: [summary of what happened]")

## Rules

- ALWAYS read the CONTEXT section before starting.
- NEVER search for something that is already open.
- read_screen() first to understand where you are.
- Maximum 10 tool calls. Be efficient.
- If task is "summarize", just call read_screen() and report.\
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    omniview_url = os.environ.get("OMNIVIEW_URL", "http://192.168.50.169:8100")

    voice_choice = _auto_select_voice()
    if voice_choice == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for OpenAI voice.")
        return
    if voice_choice == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is required for Gemini voice.")
        return

    lang = os.environ.get("LANG_VOICE", "fr").lower()[:2]
    monitor = int(os.environ.get("MONITOR", "1"))

    # --- Telemetry -----------------------------------------------------------
    from roomkit.telemetry.console import ConsoleTelemetryProvider

    class CostTracker(ConsoleTelemetryProvider):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.totals: dict[str, int] = {
                "vision_calls": 0,
                "vision_tokens": 0,
                "realtime_input": 0,
                "realtime_output": 0,
                "exec_input": 0,
                "exec_output": 0,
            }

        def record_metric(self, name: str, value: float, **kwargs: object) -> None:
            if name == "roomkit.realtime.input_tokens":
                self.totals["realtime_input"] += int(value)
            elif name == "roomkit.realtime.output_tokens":
                self.totals["realtime_output"] += int(value)
            elif name == "roomkit.llm.input_tokens":
                self.totals["exec_input"] += int(value)
            elif name == "roomkit.llm.output_tokens":
                self.totals["exec_output"] += int(value)
            super().record_metric(name, value, **kwargs)  # type: ignore[arg-type]

        def print_summary(self) -> None:
            v = self.totals["vision_tokens"]
            r = self.totals["realtime_input"] + self.totals["realtime_output"]
            e = self.totals["exec_input"] + self.totals["exec_output"]
            print("\nSession Cost Summary")
            print("-" * 50)
            print(f"  Vision:    {v:>8,} tokens ({self.totals['vision_calls']} calls)")
            print(
                f"  Voice:     {r:>8,} tokens ({self.totals['realtime_input']:,} in / {self.totals['realtime_output']:,} out)"
            )
            print(
                f"  Exec agent:{e:>8,} tokens ({self.totals['exec_input']:,} in / {self.totals['exec_output']:,} out)"
            )
            print(f"  Total:     {v + r + e:>8,} tokens")

    cost = CostTracker()
    kit = RoomKit(telemetry=cost)

    # --- StatusBus (shared between all agents) --------------------------------
    status_bus = StatusBus(
        persist_path=Path("/tmp/screen_ai")
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_status.jsonl",
    )

    # --- ToolAuditor (automatic tool call logging) ----------------------------
    audit_path = Path("/tmp/screen_ai") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_audit.jsonl"
    auditor = JSONLToolAuditor(audit_path)

    # --- OmniView client (GPU vision service) --------------------------------
    omniview = OmniViewClient(omniview_url, monitor=monitor)
    logger.info("OmniView service: %s", omniview_url)

    # --- Gemini vision (for read_screen — semantic understanding) -------------
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    gemini_vision = None
    if google_api_key:
        gemini_vision = GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=google_api_key,
                model="gemini-3.1-flash-image-preview",
                max_tokens=4096,
            )
        )
        logger.info("Gemini vision: enabled (read_screen)")
    else:
        logger.warning("GOOGLE_API_KEY not set — read_screen() will be unavailable")

    # Also keep ScreenInputTools for keyboard/mouse actions
    input_tools = ScreenInputTools(monitor=monitor)

    # --- Exec agent tool handler ---------------------------------------------
    async def _exec_tool_handler(name: str, arguments: dict[str, object]) -> str:
        import subprocess

        import pyautogui

        pyautogui.FAILSAFE = False

        key = "command" if platform.system() == "Darwin" else "ctrl"

        if name == "search_google":
            """High-level: open Chrome → new tab → type query → enter → return results."""
            query = str(arguments.get("query", ""))
            status_bus.post("exec", f'search_google("{query[:40]}")', "info", detail="Starting...")

            # Step 1: Open Chrome
            try:
                subprocess.run(
                    ["open", "-a", "Google Chrome"], check=True, capture_output=True, timeout=5
                )
                await asyncio.sleep(1.5)
                subprocess.run(
                    ["osascript", "-e", 'tell application "Google Chrome" to activate'],
                    capture_output=True,
                    timeout=3,
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                status_bus.post(
                    "exec", "search_google", "failed", detail=f"Can't open Chrome: {e}"
                )
                return json.dumps({"status": "failed", "error": f"Can't open Chrome: {e}"})

            # Step 2: New tab
            pyautogui.hotkey(key, "t")
            await asyncio.sleep(1.0)

            # Step 3: Type search query via clipboard (handles all keyboard layouts)
            import subprocess as _sp

            _sp.run(["pbcopy"], input=query.encode(), check=True)
            pyautogui.hotkey(key, "v")
            await asyncio.sleep(0.5)

            # Step 4: Press enter
            pyautogui.press("enter")
            await asyncio.sleep(2.0)  # wait for results to load

            # Step 5: Observe and return search results
            cost.totals["vision_calls"] += 1
            result = await omniview.parse()
            elements = result.get("elements", [])

            # Filter to likely search result links (text elements with meaningful content)
            results = []
            for el in elements:
                content = str(el.get("content", ""))
                if len(content) > 15 and el.get("element_type") == "text":
                    eid = el.get("id")
                    center = el.get("center", [0, 0])
                    results.append(
                        {
                            "id": eid,
                            "text": content[:80],
                            "center": center,
                        }
                    )

            status_bus.post(
                "exec",
                f'search_google("{query[:40]}")',
                "ok",
                detail=f"Found {len(results)} text elements on results page",
            )
            return json.dumps(
                {
                    "status": "ok",
                    "query": query,
                    "results": results[:15],  # top 15 most relevant
                    "total_elements": len(elements),
                    "note": "Use click_result(element_id=N) to click a result.",
                }
            )

        if name == "click_result":
            """Click a search result by element ID."""
            element_id = int(arguments.get("element_id", -1))
            el = omniview.get_element_by_id(element_id)
            if el is None:
                status_bus.post(
                    "exec", f"click_result({element_id})", "failed", detail="ID not found"
                )
                return json.dumps(
                    {
                        "status": "failed",
                        "error": f"Element {element_id} not found. Run search_google or observe first.",
                    }
                )

            cx, cy = int(el["center"][0]), int(el["center"][1])
            content = str(el.get("content", ""))

            pyautogui.click(cx, cy)
            await asyncio.sleep(2.0)  # wait for page to load

            # Verify what loaded
            cost.totals["vision_calls"] += 1
            result = await omniview.parse()
            elements = result.get("elements", [])
            # Find URL-like elements
            urls = [
                e.get("content", "")
                for e in elements
                if "http" in str(e.get("content", ""))
                or ".com" in str(e.get("content", ""))
                or ".live" in str(e.get("content", ""))
            ]
            page_texts = [
                str(e.get("content", ""))[:60]
                for e in elements
                if len(str(e.get("content", ""))) > 20
            ][:5]

            status_bus.post(
                "exec",
                f"click_result({element_id})",
                "ok",
                detail=f'Clicked "{content[:40]}" at ({cx},{cy})',
            )
            return json.dumps(
                {
                    "status": "ok",
                    "clicked": content[:60],
                    "page_hints": page_texts,
                    "urls_found": urls[:3],
                }
            )

        if name == "observe":
            """Filtered observation with URL/title extraction and OCR cleanup."""
            cost.totals["vision_calls"] += 1
            result = await omniview.parse()
            elements = result.get("elements", [])

            # Fix 5: OCR cleanup
            import re

            def _clean_ocr(text: str) -> str:
                text = re.sub(r"Jl(?=www|[a-z])", "//", text)
                text = re.sub(r"https?\s*:\s*//", "https://", text)
                text = text.replace(" .", ".").replace(". ", ".")
                return text

            # Fix 6: Extract URL and title from elements
            url = ""
            title = ""
            filtered = []
            for el in elements:
                raw = str(el.get("content", ""))
                content = _clean_ocr(raw)
                if len(content) < 10:
                    continue
                # Detect URL
                if not url and ("http" in content or ".com" in content or ".live" in content):
                    url = content
                # Detect title (first long text that's not a URL)
                if not title and len(content) > 20 and "http" not in content:
                    title = content[:80]
                filtered.append(
                    {
                        "id": el.get("id"),
                        "type": el.get("element_type"),
                        "text": content[:60],
                        "center": el.get("center"),
                    }
                )

            status_bus.post(
                "exec",
                "observe",
                "ok",
                detail=f"url={url[:40]} title={title[:40]} ({len(filtered)} elements)",
            )
            return json.dumps(
                {
                    "status": "ok",
                    "current_url": url,
                    "page_title": title,
                    "elements": filtered[:20],
                    "total_raw": len(elements),
                }
            )

        if name == "navigate":
            """Open a URL directly in a new tab."""
            url = str(arguments.get("url", ""))
            try:
                subprocess.run(
                    ["open", "-a", "Google Chrome"], check=True, capture_output=True, timeout=5
                )
                await asyncio.sleep(1.0)
                subprocess.run(
                    ["osascript", "-e", 'tell application "Google Chrome" to activate'],
                    capture_output=True,
                    timeout=3,
                )
                await asyncio.sleep(0.5)
                pyautogui.hotkey(key, "t")
                await asyncio.sleep(0.5)
                import subprocess as _sp

                _sp.run(["pbcopy"], input=url.encode(), check=True)
                pyautogui.hotkey(key, "v")
                await asyncio.sleep(0.3)
                pyautogui.press("enter")
                await asyncio.sleep(2.0)

                cost.totals["vision_calls"] += 1
                result = await omniview.parse()
                status_bus.post("exec", f"navigate({url[:40]})", "ok", detail=f"Opened {url}")
                return json.dumps({"status": "ok", "url": url})
            except Exception as e:
                status_bus.post("exec", f"navigate({url[:40]})", "failed", detail=str(e))
                return json.dumps({"status": "failed", "error": str(e)})

        if name == "read_screen":
            """Semantic understanding of the screen via Gemini vision."""
            if gemini_vision is None:
                return json.dumps(
                    {"status": "failed", "error": "No GOOGLE_API_KEY — read_screen unavailable"}
                )
            query = str(arguments.get("query", "Describe what is currently on screen."))
            frame = capture_screen_frame(monitor)
            if frame is None:
                return json.dumps({"status": "failed", "error": "No screen frame"})
            result = await gemini_vision.analyze_frame(frame, prompt=query)
            desc = result.description or "Could not analyze."
            cost.totals["vision_calls"] += 1
            status_bus.post("exec", "read_screen", "ok", detail=desc[:100])
            return json.dumps({"status": "ok", "description": desc[:500]})

        if name == "log_progress":
            msg = str(arguments.get("message", ""))
            status_bus.post("exec", "progress", "info", detail=msg)
            return json.dumps({"status": "ok"})

        return json.dumps({"error": f"Unknown tool: {name}"})

    # --- Exec agent tools (high-level only) -----------------------------------
    from roomkit.providers.ai.base import AITool

    SEARCH_TOOL = AITool(
        name="search_google",
        description="Open Chrome, new tab, search Google, return results with IDs. ONE call does everything.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    )
    CLICK_RESULT_TOOL = AITool(
        name="click_result",
        description="Click a search result by element ID from search_google(). Returns new page state.",
        parameters={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "integer",
                    "description": "Element ID from search results.",
                },
            },
            "required": ["element_id"],
        },
    )
    OBSERVE_TOOL = AITool(
        name="observe",
        description="Detect UI elements with bounding boxes for clicking. Returns element IDs + positions. Use BEFORE clicking.",
        parameters={"type": "object", "properties": {}},
    )
    READ_SCREEN_TOOL = AITool(
        name="read_screen",
        description=(
            "Understand what is on screen. Returns a natural language description "
            "of the current screen — which app, what content, URLs, page titles. "
            "Use to verify results, summarize pages, or understand context. "
            "Works for any app (browser, terminal, Finder, etc.)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific question about the screen (optional).",
                },
            },
        },
    )
    NAVIGATE_TOOL = AITool(
        name="navigate",
        description="Open a URL directly in a new Chrome tab.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to open."},
            },
            "required": ["url"],
        },
    )
    LOG_TOOL = AITool(
        name="log_progress",
        description="Report progress to the user.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Progress message."},
            },
            "required": ["message"],
        },
    )

    exec_provider = _build_exec_provider()
    # Wrap tool handler with automatic audit recording
    audited_handler = audit_tool_handler(_exec_tool_handler, auditor, agent_id="exec")
    browser_agent = Agent(
        "agent-browser",
        provider=exec_provider,
        role="Browser automation specialist",
        description="Executes browser tasks using high-level tools",
        system_prompt=_exec_system_prompt(),
        tool_handler=audited_handler,
        tools=[
            SEARCH_TOOL,
            CLICK_RESULT_TOOL,
            OBSERVE_TOOL,
            READ_SCREEN_TOOL,
            NAVIGATE_TOOL,
            LOG_TOOL,
        ],
        memory=SlidingWindowMemory(max_events=20),
        max_tool_rounds=10,
    )
    kit.register_channel(browser_agent)

    # --- Voice channel -------------------------------------------------------
    voice_provider = _build_voice_provider(voice_choice)

    sample_rate = 24000
    aec = None
    try:
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        aec = WebRTCAECProvider(sample_rate=sample_rate)
    except ImportError:
        pass

    audio_backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=20,
        mute_mic_during_playback=os.environ.get("MUTE_MIC", "1") == "1",
        aec=aec,
    )

    lang_name = {"fr": "French", "en": "English"}.get(lang, lang)

    # Voice agent has delegate_task + read_progress
    READ_PROGRESS_TOOL: dict[str, object] = {
        "name": "read_progress",
        "description": (
            "Read the latest progress updates from the execution agent. "
            "Call this to check what the browser agent has done so far."
        ),
        "parameters": {"type": "object", "properties": {}},
    }

    delegate_tool = build_delegate_tool(
        [
            ("agent-browser", "Executes browser and desktop automation tasks"),
        ]
    )

    voice_system_prompt = f"""\
You are a friendly IT support voice assistant. You speak {lang_name}.

## Your mission

Help the user find RoomKit by searching "roomkit conversation AI" in
their browser. Delegate all computer actions to agent-browser.

## Tools

- **delegate_task** — send a task to agent-browser. Be VERY specific:
  "Open Google Chrome, open a new tab, search Google for 'roomkit
  conversation AI', and click on the first result link to roomkit.live."
- **read_progress** — check what the browser agent has done so far.
  Use this to give the user status updates.

## Rules

- ALWAYS use delegate_task for any computer action.
- After delegating, tell the user "I'm working on it."
- Use read_progress to check progress and update the user.
- When the task completes, confirm what happened.
- Be concise. One short sentence per response.\
"""

    voice_channel = RealtimeVoiceChannel(
        "voice",
        provider=voice_provider,
        transport=audio_backend,
        system_prompt=voice_system_prompt,
        voice="alloy" if voice_choice == "openai" else "Aoede",
        input_sample_rate=sample_rate,
        tools=[delegate_tool.model_dump(), READ_PROGRESS_TOOL],
        mute_on_tool_call=True,
        recording=ChannelRecordingConfig(audio=True),
    )
    kit.register_channel(voice_channel)

    # --- StatusBus subscriber: auto-inject progress into voice session ------
    async def _on_status(entry: Any) -> None:
        """When exec agent posts progress/completed, inject into voice session."""
        if entry.agent_id != "exec":
            return
        if entry.status not in ("info", "completed", "failed"):
            return
        # Inject as silent context so voice agent knows what happened
        sessions = voice_channel.get_room_sessions("screen-room")
        for session in sessions:
            try:
                text = f"[Agent update] {entry.action}: {entry.detail}"
                await voice_channel.inject_text(session, text, role="user", silent=True)
            except Exception:
                pass

    await status_bus.subscribe(_on_status)

    # --- Delegation + hooks --------------------------------------------------
    delegate_handler = DelegateHandler(kit, delivery_strategy=WaitForIdleDelivery())

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="tool_router")
    async def handle_voice_tools(event: object, ctx: object) -> HookResult:
        tool_event = event  # type: ignore[assignment]
        name = tool_event.name  # type: ignore[attr-defined]
        arguments = tool_event.arguments  # type: ignore[attr-defined]

        if name == "delegate_task":
            # Fix 1: Delegation lock — prevent re-delegation if task just completed
            completed = await status_bus.recent(1, status="completed")
            if completed:
                last_ts = datetime.fromisoformat(completed[0].ts)
                age = (datetime.now(UTC) - last_ts).total_seconds()
                if age < 5:
                    status_bus.post(
                        "voice",
                        "delegate_task",
                        "info",
                        detail=f"Skipped — task completed {age:.0f}s ago: {completed[0].detail}",
                    )
                    result_str = json.dumps(
                        {
                            "status": "already_done",
                            "message": f"Previous task completed {age:.0f}s ago: {completed[0].detail}. "
                            "Ask the user if they need something different.",
                        }
                    )
                    return HookResult(action="allow", metadata={"result": result_str})

            try:
                # Fix 2: Inject previous context into new delegations
                args = dict(arguments)
                recent_context = await status_bus.recent_text(8)
                original_task = args.get("task", "")
                args["task"] = (
                    f"CONTEXT (what happened before):\n{recent_context}\n\n"
                    f"NEW TASK: {original_task}"
                )

                result = await delegate_handler.handle(
                    room_id="screen-room",
                    calling_agent_id="voice",
                    arguments=args,
                )
                status_bus.post("voice", "delegate_task", "ok", detail=str(original_task)[:100])
                result_str = json.dumps(result)
            except Exception as exc:
                logger.exception("Delegation failed")
                status_bus.post("voice", "delegate_task", "failed", detail=str(exc))
                result_str = json.dumps({"error": str(exc)})

        elif name == "read_progress":
            recent = await status_bus.recent_text(8)
            status_bus.post("voice", "read_progress", "ok")
            result_str = json.dumps({"progress": recent})

        else:
            return HookResult.allow()

        return HookResult(action="allow", metadata={"result": result_str})

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event, ctx):
        logger.info("Task delegated → agent=%s", event.metadata.get("agent_id"))

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event, ctx):
        duration = event.metadata.get("duration_ms", 0)
        status_bus.post(
            "system", "task_completed", "completed", detail=f"Duration: {duration:.0f}ms"
        )

    # --- Screen recording with watermark ------------------------------------
    from roomkit.recorder import RoomRecorderBinding
    from roomkit.video.backends.screen import ScreenCaptureBackend
    from roomkit.video.pipeline.filter.watermark import WatermarkFilter

    try:
        from roomkit.recorder.pyav import PyAVMediaRecorder

        recorder = PyAVMediaRecorder()
        recorder_label = "PyAV → MP4"
    except ImportError:
        from roomkit.recorder import MockMediaRecorder

        recorder = MockMediaRecorder()
        recorder_label = "Mock (install roomkit[video] for MP4)"

    recording_dir = Path("/tmp/screen_ai") / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)
    recording_config = MediaRecordingConfig(
        storage=str(recording_dir),
        video_fps=5,
        audio_sample_rate=24000,
    )

    screen_backend = ScreenCaptureBackend(
        monitor=monitor,
        fps=5,
        scale=0.75,
    )
    watermark = WatermarkFilter(
        text="RoomKit | {timestamp}",
        position="bottom-right",
        color=(255, 255, 255),
        bg_color=(0, 0, 0),
        font_scale=0.5,
    )
    from roomkit.video.pipeline.config import VideoPipelineConfig

    screen_channel = VideoChannel(
        "screen-rec",
        backend=screen_backend,
        recording=ChannelRecordingConfig(video=True),
        pipeline=VideoPipelineConfig(filters=[watermark]),
    )
    kit.register_channel(screen_channel)

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(
        room_id="screen-room",
        recorders=[
            RoomRecorderBinding(
                recorder=recorder,
                config=recording_config,
                name="screen-recording",
            ),
        ],
    )
    await kit.attach_channel("screen-room", "voice")
    await kit.attach_channel("screen-room", "agent-browser", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("screen-room", "screen-rec")

    room = await kit.get_room("screen-room")
    room = set_conversation_state(room, ConversationState(phase="conversation"))
    await kit.store.update_room(room)

    # Start voice BEFORE video capture — PyAV/FFmpeg requires all streams
    # added to the container before the first packet is muxed. This is a
    # known limitation; dynamic participant recording needs per-track files.
    provider_config = {}
    if voice_choice == "gemini":
        provider_config = {
            "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
            "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
            "silence_duration_ms": 1500,
        }

    await voice_channel.start_session(
        "screen-room",
        "local-user",
        connection=None,
        metadata={"provider_config": provider_config} if provider_config else None,
    )

    # Start screen capture (after voice so all tracks are registered)
    video_session = await kit.connect_video("screen-room", "local-user", "screen-rec")
    await screen_backend.start_capture(video_session)

    # --- Banner --------------------------------------------------------------
    print()
    print("Screen Assistant — Orchestrated + OmniView")
    print("=" * 60)
    print(f"Voice: {voice_choice} | Language: {lang}")
    print(f"Exec agent: {type(exec_provider).__name__}")
    print(f"Vision: OmniView @ {omniview_url}")
    print(f"Recording: {recorder_label} → {recording_dir}/")
    print(f"AEC: {'on' if aec else 'off'} | Mute mic: {os.environ.get('MUTE_MIC', '1')}")
    print()
    print("Architecture: voice → delegate → exec agent → OmniView (GPU)")
    print("Press Ctrl+C to stop.")
    print()

    # --- Run -----------------------------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    logger.info("Stopping...")
    await screen_backend.stop_capture(video_session)
    await kit.close()
    print(f"\nRecording: {recording_dir}/ ({recorder_label})")
    cost.print_summary()
    await status_bus.print_summary()
    auditor.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
