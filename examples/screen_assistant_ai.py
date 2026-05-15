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
    GOOGLE_API_KEY=... uv run python examples/screen_assistant_ai.py

Run with (OpenAI voice + Gemini vision):
    GOOGLE_API_KEY=... OPENAI_API_KEY=... VISION_TOOL=gemini \
        uv run python examples/screen_assistant_ai.py

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
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from screen_assistant import (
    CLICK_RESULT_TOOL,
    LIST_SCREENS_TOOL,
    OBSERVE_TOOL,
    OPEN_APP_TOOL,
    SWITCH_SCREEN_TOOL,
    CostTrackingTelemetry,
    OmniViewClient,
    ScreenAssistantState,
    ScreenToolDispatcher,
    build_system_prompt,
    build_vision_provider,
    build_voice_provider,
    get_voice_name,
    print_banner,
    provider_config_for,
    send_opening_greeting,
    setup_playwright_mcp,
    setup_screen_vision,
)
from shared import (
    auto_select_provider,
    build_aec,
    build_denoiser,
    build_pipeline,
    run_until_stopped,
    setup_console,
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
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("screen_assistant_ai")

# Drop the AEC stats/reference logs (emitted every ~1s) but keep
# one-shot init/activated/reset lines so the user can still see the
# AEC actually started.
import logging  # noqa: E402


class _DropAECStats(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not (msg.startswith("AEC stats") or msg.startswith("AEC reference"))


logging.getLogger("roomkit.voice.pipeline.aec.webrtc").addFilter(_DropAECStats())

ROOM_ID = "screen-assistant"
PARTICIPANT_ID = "local-user"
SAMPLE_RATE = 24000
BLOCK_MS = 20


async def main() -> None:
    # --- Env + provider selection -------------------------------------------
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        print("GOOGLE_API_KEY is required.")
        print("  GOOGLE_API_KEY=... uv run python examples/screen_assistant_ai.py")
        return

    voice_choice = auto_select_provider("VOICE_PROVIDER", "voice")
    tool_choice = auto_select_provider("VISION_TOOL", "vision tool")
    if voice_choice == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for OpenAI voice.")
        return
    if tool_choice == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for OpenAI vision tool.")
        return

    lang = os.environ.get("LANG_VOICE", os.environ.get("LANG", "en")).lower()[:2]
    monitor = int(os.environ.get("MONITOR", "1"))
    auto_verify = os.environ.get("AUTO_VERIFY", "1") != "0"
    omniview_url = os.environ.get("OMNIVIEW_URL", "")
    vision_interval = int(os.environ.get("VISION_INTERVAL", "5000"))

    # --- RoomKit kit + telemetry --------------------------------------------
    cost_telemetry = CostTrackingTelemetry()
    kit = RoomKit(telemetry=cost_telemetry)
    console_cleanup = setup_console(kit)  # CONSOLE=1 to enable dashboard

    audit_path = f"/tmp/screen_ai/{datetime.now().strftime('%Y%m%d_%H%M%S')}_session.jsonl"
    auditor = JSONLSessionAuditor(audit_path)
    auditor.attach(kit)

    # --- Shared screen state ------------------------------------------------
    vision = build_vision_provider(tool_choice, google_api_key)
    screen_backend = ScreenCaptureBackend(
        monitor=monitor,
        fps=2,
        scale=float(os.environ.get("SCALE", "0.75")),
        diff_threshold=float(os.environ.get("DIFF_THRESHOLD", "0.15")),
    )
    omniview = OmniViewClient(omniview_url, monitor=monitor) if omniview_url else None
    if omniview:
        logger.info("OmniView enabled: %s", omniview_url)

    state = ScreenAssistantState(
        vision=vision,
        screen_backend=screen_backend,
        cost_telemetry=cost_telemetry,
        monitor=monitor,
        omniview=omniview,
        auto_verify=auto_verify,
    )

    # --- Optional Playwright MCP server -------------------------------------
    (
        browser_mode,
        playwright_mcp,
        playwright_tools,
        playwright_tool_names,
        pw_cleanup,
    ) = await setup_playwright_mcp(voice_choice, os.environ.get("BROWSER_MODE", "vision").lower())

    # --- Video channel for periodic screen vision ---------------------------
    video_channel = VideoChannel(
        "video-screen",
        backend=screen_backend,
        vision=vision,
        vision_interval_ms=vision_interval,
    )
    kit.register_channel(video_channel)

    # --- Realtime voice channel ---------------------------------------------
    aec = build_aec(SAMPLE_RATE, BLOCK_MS, default="webrtc")
    denoiser = build_denoiser(default="sherpa")
    pipeline = build_pipeline(aec=aec, denoiser=denoiser)
    mute_mic = os.environ.get("MUTE_MIC", "0") == "1"
    audio_backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=mute_mic,
    )
    voice_name = get_voice_name(voice_choice)
    tools = [
        state.screen_tool.definition,
        LIST_SCREENS_TOOL,
        SWITCH_SCREEN_TOOL,
        OPEN_APP_TOOL,
        *state.input_tools.definitions,
        *playwright_tools,
        *([OBSERVE_TOOL, CLICK_RESULT_TOOL] if omniview else []),
    ]
    voice_channel = RealtimeVoiceChannel(
        "voice",
        provider=build_voice_provider(voice_choice),
        transport=audio_backend,
        system_prompt=build_system_prompt(
            lang, browser_mode=browser_mode, omniview=omniview is not None
        ),
        voice=voice_name,
        input_sample_rate=SAMPLE_RATE,
        pipeline=pipeline,
        tools=tools,
        mute_on_tool_call=True,
    )
    kit.register_channel(voice_channel)

    # --- Tool dispatch hook -------------------------------------------------
    dispatcher = ScreenToolDispatcher(
        state,
        playwright_mcp=playwright_mcp,
        playwright_tool_names=playwright_tool_names,
    )

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="screen_tool_handler")
    async def on_tool_call(event: object, ctx: object) -> HookResult:
        t0 = time.monotonic()
        result = await dispatcher.handle(event.name, event.arguments)  # type: ignore[attr-defined]
        if result is None:
            return HookResult.allow()
        auditor.record_tool(
            ToolAuditEntry(
                ts=datetime.now().isoformat(),
                agent_id=ROOM_ID,
                tool_name=event.name,  # type: ignore[attr-defined]
                arguments=dict(event.arguments),  # type: ignore[attr-defined]
                result=result[:500],
                status="ok",
                duration_ms=round((time.monotonic() - t0) * 1000),
                metadata={"screen_after": state.latest_description[:200]},
            )
        )
        return HookResult(action="allow", metadata={"result": result})

    # --- Vision change → voice injection ------------------------------------
    # Same API shape as roomkit.video.setup_realtime_vision, but with a
    # term-set diff so the agent only reacts to meaningful changes.
    setup_screen_vision(kit, ROOM_ID, voice_channel, state)

    # --- Room + sessions ----------------------------------------------------
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "video-screen")
    await kit.attach_channel(ROOM_ID, "voice")

    video_session = await kit.join(ROOM_ID, "video-screen", participant_id=PARTICIPANT_ID)
    await screen_backend.start_capture(video_session)

    pcfg = provider_config_for(voice_choice)
    await voice_channel.start_session(
        ROOM_ID,
        PARTICIPANT_ID,
        connection=None,
        metadata={"provider_config": pcfg} if pcfg else None,
    )

    # Make the agent speak first. Realtime providers wait for input by
    # default; this nudges the model to produce its opening turn from the
    # system prompt's ## Greeting section. (Equivalent to Agent.auto_greet
    # for the orchestrated-Agent pattern; we use a bare RealtimeVoiceChannel
    # here for simplicity, same as examples/realtime_voice_local_*.py.)
    await send_opening_greeting(voice_channel, ROOM_ID)

    # --- Banner + run -------------------------------------------------------
    print_banner(
        voice_choice=voice_choice,
        tool_choice=tool_choice,
        voice_name=voice_name,
        lang=lang,
        aec=aec,
        denoiser=denoiser,
        mute_mic=mute_mic,
        auto_verify=auto_verify,
        browser_mode=browser_mode,
        playwright_tool_count=len(playwright_tools),
        omniview_url=omniview_url or None,
        vision_interval_ms=vision_interval,
    )

    async def _cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await screen_backend.stop_capture(video_session)
        for ctx in pw_cleanup:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        cost_telemetry.print_summary()
        auditor.print_summary()
        logger.info("Vision analyzed %d frames.", state.frame_count)

    await run_until_stopped(kit, cleanup=_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
