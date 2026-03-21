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
    OMNIVIEW_URL         (optional) OmniView GPU service URL for precise element detection

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
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from screen_assistant import (
    ACTION_TOOLS,
    CLICK_RESULT_TOOL,
    LANG_NAMES,
    LIST_SCREENS_TOOL,
    OBSERVE_TOOL,
    OPEN_APP_TOOL,
    SWITCH_SCREEN_TOOL,
    CostTrackingTelemetry,
    OmniViewClient,
    assess_action_result,
    build_system_prompt,
    build_verify_question,
    build_vision_provider,
    build_voice_provider,
    get_voice_name,
    setup_playwright_mcp,
)
from shared import (
    auto_select_provider,
    build_aec,
    build_denoiser,
    build_pipeline,
    run_until_stopped,
    setup_logging,
)

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RealtimeVoiceChannel,
    RoomKit,
    VideoChannel,
)
from roomkit.orchestration.session_audit import JSONLSessionAuditor
from roomkit.orchestration.tool_audit import ToolAuditEntry
from roomkit.video.backends.screen import ScreenCaptureBackend
from roomkit.video.vision.screen_input import ScreenInputTools
from roomkit.video.vision.screen_tool import DescribeScreenTool
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("screen_assistant_ia")


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------


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
    for url in re.findall(r"[\w-]+\.[\w.-]+", description):
        terms.add(url.lower())
    return terms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        print("GOOGLE_API_KEY is required.")
        print("  GOOGLE_API_KEY=... uv run python examples/screen_assistant_ia.py")
        return

    # --- Provider selection --------------------------------------------------
    voice_choice = auto_select_provider("VOICE_PROVIDER", "voice")
    tool_choice = auto_select_provider("VISION_TOOL", "vision tool")

    if voice_choice == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for OpenAI voice.")
        return
    if tool_choice == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for OpenAI vision tool.")
        return

    lang = os.environ.get("LANG_VOICE", os.environ.get("LANG", "en")).lower()[:2]
    auto_verify = os.environ.get("AUTO_VERIFY", "1") != "0"

    # --- OmniView (optional GPU vision) --------------------------------------
    omniview_url = os.environ.get("OMNIVIEW_URL", "")
    omniview: OmniViewClient | None = None
    if omniview_url:
        monitor_init = int(os.environ.get("MONITOR", "1"))
        omniview = OmniViewClient(omniview_url, monitor=monitor_init)
        logger.info("OmniView enabled: %s", omniview_url)

    # --- RoomKit + telemetry -------------------------------------------------
    cost_telemetry = CostTrackingTelemetry()
    kit = RoomKit(telemetry=cost_telemetry)

    # --- Shared vision provider ----------------------------------------------
    monitor = int(os.environ.get("MONITOR", "1"))
    vision = build_vision_provider(tool_choice, google_api_key)
    screen_tool = DescribeScreenTool(vision, monitor=monitor)

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

    # --- Periodic screen vision (VideoChannel) -------------------------------
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

    # --- Vision change detection → inject into voice session -----------------
    latest_vision: dict[str, str] = {"description": "", "previous": ""}
    frame_count = 0

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="vision_observer")
    async def vision_observer_hook(event: object, ctx: object) -> None:
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
    (
        browser_mode,
        playwright_mcp,
        playwright_tools,
        playwright_tool_names,
        _pw_cleanup,
    ) = await setup_playwright_mcp(voice_choice, browser_mode)

    # --- System prompt (built after browser_mode is resolved) ----------------
    system_prompt = build_system_prompt(
        lang, browser_mode=browser_mode, omniview=omniview is not None
    )

    # --- Tool definitions + handler ------------------------------------------
    input_tools = ScreenInputTools(vision=vision, monitor=monitor)

    all_tools = [
        screen_tool.definition,
        LIST_SCREENS_TOOL,
        SWITCH_SCREEN_TOOL,
        OPEN_APP_TOOL,
        *input_tools.definitions,
        *playwright_tools,
        *([] if omniview is None else [OBSERVE_TOOL, CLICK_RESULT_TOOL]),
    ]

    async def _list_screens() -> str:
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
        if omniview is not None:
            omniview.monitor = monitor
        logger.info("Switched to monitor %d", monitor)
        return f"Switched to monitor {monitor}. All tools now target this screen."

    _audit_file = f"/tmp/screen_ai/{datetime.now().strftime('%Y%m%d_%H%M%S')}_session.jsonl"
    auditor = JSONLSessionAuditor(_audit_file)
    auditor.attach(kit)  # auto-capture speech, vision, barge-in

    @kit.hook(
        HookTrigger.ON_TOOL_CALL,
        execution=HookExecution.SYNC,
        name="screen_tool_handler",
    )
    async def screen_tool_hook(event: object, ctx: object) -> HookResult:
        """Handle all tool calls via the hook pipeline with audit logging."""
        tool_event = event  # type: ignore[assignment]
        name = tool_event.name  # type: ignore[attr-defined]
        arguments = tool_event.arguments  # type: ignore[attr-defined]

        _t0 = time.monotonic()
        result: str

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
                await asyncio.sleep(2.0)
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
                is_focused = app_name.lower() in verify.lower() or "chrome" in verify.lower()
                status = "OK" if is_focused else "FAILED"
                result = (
                    f"ACTION: open_app({app_name})\nSTATUS: {status}\n"
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
            if not query:
                query = "Describe the current screen: foreground app, visible content, URLs."
            result = await _analyze_with_cost(query)
            latest_vision["description"] = result

        elif name == "observe" and omniview is not None:
            logger.info("observe: calling OmniView /parse")
            cost_telemetry.totals["vision_calls"] += 1
            result_data = await omniview.parse()
            elements = result_data.get("elements", [])
            logger.info("observe: OmniView returned %d elements", len(elements))
            img_h = result_data.get("height", 2160)

            # Build element list with cleaned content
            all_els = []
            for el in elements:
                content = omniview.clean_ocr(str(el.get("content", "")))
                if len(content) < 3:
                    continue
                center = el.get("center", [0, 0])
                all_els.append(
                    {
                        "id": el.get("id"),
                        "type": el.get("element_type"),
                        "text": content[:80],
                        "center": center,
                        "interactable": el.get("interactable", False),
                    }
                )

            # Prioritize: text elements and interactable icons in the
            # main content area (middle 60% of screen) come first.
            # This ensures search results, links, and buttons are shown
            # before toolbar icons at the top/bottom edges.
            top_band = img_h * 0.15
            bottom_band = img_h * 0.85

            def _priority(e: dict) -> tuple:
                cy = e["center"][1]
                in_content = top_band < cy < bottom_band
                is_text = e["type"] == "text"
                # Sort: content-area text first, then content-area icons,
                # then edge text, then edge icons
                return (not in_content, not is_text, cy)

            all_els.sort(key=_priority)

            result = json.dumps(
                {
                    "status": "ok",
                    "elements": all_els[:40],
                    "total": len(all_els),
                    "note": "Use click_result(element_id=N) to click an element. "
                    "Elements are sorted by relevance — content area first.",
                }
            )

        elif name == "click_result" and omniview is not None:
            element_id = int(arguments.get("element_id", -1))
            el = omniview.get_element_by_id(element_id)
            logger.info(
                "click_result: element_id=%d found=%s content=%r",
                element_id,
                el is not None,
                str(el.get("content", ""))[:60] if el else "N/A",
            )
            if el is None:
                result = json.dumps(
                    {
                        "status": "failed",
                        "error": f"Element {element_id} not found. Run observe first.",
                    }
                )
            else:
                cx, cy = int(el["center"][0]), int(el["center"][1])  # type: ignore[index]
                content = str(el.get("content", ""))
                button = str(arguments.get("button", "left"))
                double = bool(arguments.get("double", False))
                omniview.click_at(cx, cy, button=button, clicks=2 if double else 1)
                await asyncio.sleep(0.5)

                if auto_verify:
                    verify_q = build_verify_question("click_result", arguments)
                    screen_desc = await _analyze_with_cost(verify_q)
                    latest_vision["description"] = screen_desc
                    verdict = assess_action_result("click_result", arguments, screen_desc)
                    result = (
                        f'ACTION: click_result({element_id}) -> "{content[:40]}"\n'
                        f"STATUS: {verdict['status']}\n"
                        f"SCREEN: {screen_desc[:200]}\nVERDICT: {verdict['verdict']}"
                    )
                    if verdict["status"] == "FAILED":
                        result += f"\nSUGGESTION: {verdict['suggestion']}"
                else:
                    result = json.dumps(
                        {"status": "ok", "clicked": content[:60], "center": [cx, cy]}
                    )

        elif name in ACTION_TOOLS:
            # For click_element, try OmniView /locate first when available
            if name == "click_element" and omniview is not None:
                element_desc = str(arguments.get("element", ""))
                logger.info(
                    "click_element: trying OmniView /locate for %r",
                    element_desc,
                )
                try:
                    locate_result = await omniview.locate(element_desc)
                    el = locate_result.get("element") or {}
                    center = el.get("center")
                    logger.info(
                        "click_element: OmniView /locate result: found=%s center=%s content=%r score=%s",
                        locate_result.get("found"),
                        center,
                        str(el.get("content", ""))[:60],
                        locate_result.get("match_score", "?"),
                    )
                    if locate_result.get("found") and center and center[0] and center[1]:
                        cx, cy = int(center[0]), int(center[1])
                        omniview.click_at(cx, cy)
                        logger.info(
                            "click_element via OmniView: %r → click at (%d,%d)",
                            element_desc,
                            cx,
                            cy,
                        )
                        action_result = f"Clicked '{element_desc}' via OmniView at ({cx},{cy})."
                    else:
                        reason = (
                            "not found" if not locate_result.get("found") else "no center coords"
                        )
                        logger.info(
                            "click_element: OmniView miss for %r (%s), falling back to vision",
                            element_desc,
                            reason,
                        )
                        action_result = await input_tools.handler(name, arguments)
                except Exception:
                    logger.exception(
                        "click_element: OmniView /locate failed, falling back to vision"
                    )
                    action_result = await input_tools.handler(name, arguments)
            elif name == "click_element":
                logger.info("click_element: OmniView not available, using vision")
                action_result = await input_tools.handler(name, arguments)
            else:
                action_result = await input_tools.handler(name, arguments)
            if auto_verify:
                try:
                    await asyncio.sleep(0.5)
                    verify_q = build_verify_question(name, arguments)
                    screen_desc = await _analyze_with_cost(verify_q)
                    latest_vision["description"] = screen_desc
                    verdict = assess_action_result(name, arguments, screen_desc)
                    result = (
                        f"ACTION: {name}({json.dumps(dict(arguments), default=str)})\n"
                        f"STATUS: {verdict['status']}\n"
                        f"SCREEN: {screen_desc[:200]}\nVERDICT: {verdict['verdict']}"
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
            _PW_MAX_RESULT = 4000
            logger.info("Playwright MCP tool: %s(%s)", name, arguments)
            pw_result = await playwright_mcp.call_tool(name, arguments)  # type: ignore[union-attr]
            texts = [c.text for c in pw_result.content if hasattr(c, "text")]
            raw = "\n".join(texts) if texts else json.dumps({"status": "ok"})
            result = (
                raw[:_PW_MAX_RESULT] + f"\n... [truncated {len(raw)} → {_PW_MAX_RESULT} chars]"
                if len(raw) > _PW_MAX_RESULT
                else raw
            )
        else:
            return HookResult.allow()

        auditor.record_tool(
            ToolAuditEntry(
                ts=datetime.now().isoformat(),
                agent_id="screen-assistant",
                tool_name=name,
                arguments=dict(arguments),
                result=result[:500],
                status="ok",
                duration_ms=round((time.monotonic() - _t0) * 1000),
                metadata={"screen_after": latest_vision.get("description", "")[:200]},
            )
        )
        return HookResult(action="allow", metadata={"result": result})

    # --- Speech-to-speech voice (RealtimeVoiceChannel) -----------------------
    sample_rate = 24000
    block_ms = 20

    provider = build_voice_provider(voice_choice)
    aec = build_aec(sample_rate, block_ms, default="webrtc")
    denoiser = build_denoiser(default="sherpa")
    pipeline = build_pipeline(aec=aec, denoiser=denoiser)

    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env == "1" if mute_env is not None else False
    audio_backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        pipeline=pipeline,
    )

    voice_name = get_voice_name(voice_choice)
    voice_channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=audio_backend,
        system_prompt=system_prompt,
        voice=voice_name,
        input_sample_rate=sample_rate,
        tools=all_tools,
        mute_on_tool_call=True,
    )
    kit.register_channel(voice_channel)

    # --- Room + sessions -----------------------------------------------------
    await kit.create_room(room_id="screen-assistant")
    await kit.attach_channel("screen-assistant", "video-screen")
    await kit.attach_channel("screen-assistant", "voice")

    video_session = await kit.join(
        "screen-assistant",
        "video-screen",
        participant_id="local-user",
    )
    await screen_backend.start_capture(video_session)

    provider_config: dict[str, object] = {}
    if voice_choice == "gemini":
        provider_config = {
            "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
            "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
            "silence_duration_ms": 1500,
        }
    elif voice_choice == "openai":
        provider_config = {"eagerness": "low"}

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
    pw_label = (
        f"playwright ({len(playwright_tools)} tools)" if browser_mode == "playwright" else "vision"
    )
    omniview_label = f"OmniView @ {omniview_url}" if omniview else "off"
    print()
    print(f"Screen Assistant ({voice_label} Voice + {tool_label} Vision)")
    print("=" * 60)
    print(f"Voice: {voice_name} | Language: {LANG_NAMES.get(lang, lang)}")
    print(f"AEC: {'on' if aec else 'off'} | Denoiser: {'on' if denoiser else 'off'}")
    print(f"Interruption: {'off' if mute_mic else 'on'} | Auto-verify: {verify_label}")
    print(f"Browser: {pw_label}")
    print(f"OmniView: {omniview_label}")
    print(f"Vision: every {vision_interval}ms (diff-gated, silent injection)")
    print()
    tools_line = "Tools: list_screens, switch_screen, open_app, describe_screen,"
    tools_line2 = "       click_element, type_text, press_key, scroll"
    if omniview:
        tools_line2 += ", observe, click_result"
    print(tools_line)
    print(tools_line2)
    print("Press Ctrl+C to stop.")
    print()

    # --- Keep running until Ctrl+C -------------------------------------------
    async def _cleanup() -> None:
        await screen_backend.stop_capture(video_session)
        for ctx in _pw_cleanup:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        cost_telemetry.print_summary()
        auditor.print_summary()
        logger.info("Vision analyzed %d frames.", frame_count)

    await run_until_stopped(kit, cleanup=_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
