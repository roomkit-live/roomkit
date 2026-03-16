"""RoomKit — AI screen assistant with speech-to-speech voice.

Talk to an AI while it sees your screen in real time. The AI observes
what's on your display and guides you through tasks step by step using
natural voice conversation.

Supports **OpenAI Realtime** or **Gemini Live** for voice, and
**OpenAI** or **Gemini** for the ``describe_screen`` vision tool.
Periodic screen vision always uses Gemini (requires GOOGLE_API_KEY).

Demo task: open Google Chrome, search "roomkit python conversation ai",
and navigate to the RoomKit website.

Requirements:
    pip install roomkit[screen-capture,local-audio,gemini,sherpa-onnx]
    pip install roomkit[realtime-openai]   # for OpenAI voice
    pip install roomkit[realtime-gemini]   # for Gemini voice
    pip install aec-audio-processing       # WebRTC echo cancellation

Run with (Gemini voice + Gemini tool — only GOOGLE_API_KEY needed):
    GOOGLE_API_KEY=... uv run python examples/screen_assistant_ia.py

Run with (OpenAI voice + Gemini tool):
    GOOGLE_API_KEY=... OPENAI_API_KEY=... VISION_TOOL=gemini \
        uv run python examples/screen_assistant_ia.py

Environment variables:
    GOOGLE_API_KEY       (required) Google API key (periodic vision)
    OPENAI_API_KEY       (optional) OpenAI API key (voice / tool)
    VOICE_PROVIDER       Force voice: openai | gemini (auto if unset)
    VISION_TOOL          Force tool:  openai | gemini (auto if unset)
    GEMINI_MODEL         Gemini speech model
    GEMINI_VOICE         Gemini voice preset (default: Aoede)
    GEMINI_VISION_MODEL  Periodic vision model (default: gemini-3.1-flash-image-preview)
    OPENAI_MODEL         OpenAI speech model (default: gpt-realtime-1.5)
    OPENAI_VOICE         OpenAI voice preset (default: alloy)
    OPENAI_VISION_MODEL  OpenAI tool model (default: gpt-4o)
    SCALE                Capture scale 0.0-1.0 (default: 0.75)
    AEC                  Echo cancellation: webrtc | speex | 0 (default: webrtc)
    DENOISE              Noise suppression: 1 | 0 (default: 1)
    DENOISE_MODEL        ONNX model file (default: gtcrn_simple.onnx)
    MUTE_MIC             Mute mic during AI playback: 1 | 0 (default: 0)
    LANG_VOICE           Language (default: en)
    MONITOR              Monitor index: 1=primary (default: 1)
    VISION_INTERVAL      Vision analysis interval in ms (default: 10000)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time

from roomkit import (
    DescribeScreenTool,
    GeminiVisionConfig,
    GeminiVisionProvider,
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
    """Pick openai or gemini based on available keys and env override.

    Args:
        env_var: Name of the env var that forces the choice
                 (e.g. ``VOICE_PROVIDER``, ``VISION_TOOL``).
        label: Human label for the interactive prompt.
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_system_prompt(lang: str) -> str:
    """Build system prompt, optionally in a specific language."""
    lang_name = LANG_NAMES.get(lang, lang)
    lang_instruction = (
        f"\nYou MUST speak and respond ONLY in {lang_name}. Never switch to another language."
        if lang != "en"
        else ""
    )

    return f"""\
You are a professional IT support assistant. You can see the user's screen \
in real time, control the mouse, type on the keyboard, and you are having a \
voice conversation with them.{lang_instruction}

## Your first action

When the conversation starts, immediately call describe_screen to see what is \
on the user's screen. Then introduce yourself:
"Hello! I'm your IT support assistant. I can see your screen. \
I'm going to help you navigate to the RoomKit website at roomkit.live. \
Let's get started!"

Then guide the user step by step: open a browser, go to roomkit.live.

## How you see the screen

You have TWO ways to see the screen:
1. **Periodic updates**: You receive automatic screen descriptions every \
~10 seconds as text messages. These give you general context. Do NOT reply \
to every update — only speak when the user talks to you or when you need \
to give the next step.
2. **describe_screen tool**: Call this tool at ANY time to ask a specific \
question about what is currently visible on the screen.

## CRITICAL RULES for using describe_screen

- **NEVER guess or assume** what is on the screen. Always call \
describe_screen first, then answer based on the result.
- **When the user cannot find something**, immediately call describe_screen \
with a fresh, specific query — do NOT rely on old periodic updates.
- Before giving a precise instruction ("click on the third icon"), call \
describe_screen first to verify what you're about to say is correct.
- Ask precise questions. Bad: "describe the screen". \
Good: "List all icons in the taskbar from left to right with their names."
- After calling the tool, speak naturally: "I can see that..." — never \
mention the tool itself.

## Asking permission before taking control

**You MUST ask for permission before every mouse or keyboard action.** \
Never move the mouse, click, type, or press keys without the user's \
explicit consent first. Examples:
- "I can see the Chrome icon in your taskbar. Would you like me to click \
on it for you?"
- "I see the address bar. Can I type the URL for you?"
- "Would you like me to press Enter to go to the page?"

Only proceed after the user says yes, sure, go ahead, or similar. \
If the user says no or wants to do it themselves, give them clear \
verbal instructions instead.

## Mouse and keyboard control

When the user gives permission, you can:
- **locate_element(element)** — find the exact pixel coordinates of a UI \
element. Returns {{x, y}}. You MUST call this before every click or \
move_mouse action.
- **click(x, y)** — click at the coordinates returned by locate_element.
- **type_text(text)** — type text into the currently focused field.
- **press_key(key)** — press keys like 'enter', 'tab', 'escape', \
or combos like 'ctrl+a', 'ctrl+c'.
- **scroll(clicks)** — scroll up (positive) or down (negative).
- **move_mouse(x, y)** — move cursor to coordinates from locate_element.

CRITICAL workflow for clicking:
1. Call locate_element("the Chrome icon in the taskbar") → get {{x, y}}
2. Call click(x, y) with the EXACT coordinates returned
NEVER invent or guess coordinates. ALWAYS use locate_element first.

## Guiding the user

- Give short, clear voice directions one step at a time.
- Wait for the user to complete each step before moving to the next.
- Use describe_screen to confirm progress after each instruction.
- The user can interrupt you at any time — stop immediately and listen.
- Be professional, patient, and reassuring — like a real support agent.
- Speak naturally — you are a voice assistant, not a text bot.\
"""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_aec(sample_rate: int, block_ms: int) -> object | None:
    """Build AEC provider based on AEC env var."""
    aec_mode = os.environ.get("AEC", "webrtc").lower()
    if aec_mode in ("1", "webrtc"):
        try:
            from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

            logger.info("AEC enabled (WebRTC AEC3)")
            return WebRTCAECProvider(sample_rate=sample_rate)
        except ImportError:
            print("\n" + "=" * 60)
            print("WARNING: WebRTC AEC not installed!")
            print("Echo cancellation is CRITICAL to avoid interruptions.")
            print("Install with: pip install aec-audio-processing")
            print("=" * 60 + "\n")
            return None
    if aec_mode == "speex":
        try:
            from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

            frame_size = sample_rate * block_ms // 1000
            logger.info("AEC enabled (Speex)")
            return SpeexAECProvider(
                frame_size=frame_size,
                filter_length=frame_size * 10,
                sample_rate=sample_rate,
            )
        except ImportError:
            print("\n" + "=" * 60)
            print("WARNING: Speex AEC not installed!")
            print("Install with: apt install libspeexdsp1")
            print("=" * 60 + "\n")
            return None
    return None


def _build_denoiser() -> object | None:
    """Build sherpa-onnx GTCRN denoiser based on DENOISE env var (default: on)."""
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
        logger.warning(
            "sherpa-onnx not available — pip install roomkit[sherpa-onnx] (DENOISE=0 to skip)"
        )
        return None


def _build_periodic_vision(google_api_key: str) -> GeminiVisionProvider:
    """Build Gemini vision provider for periodic screen analysis."""
    return GeminiVisionProvider(
        GeminiVisionConfig(
            api_key=google_api_key,
            model=os.environ.get("GEMINI_VISION_MODEL", "gemini-3.1-flash-image-preview"),
            prompt=(
                "Describe what is shown on this screen in 2-3 sentences. "
                "Focus on the FOREGROUND application (ignore terminal/log windows). "
                "Include the application name, visible text, URLs, and what "
                "the user appears to be doing. Be concise."
            ),
        )
    )


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


def _build_screen_tool(
    tool_choice: str,
    google_api_key: str,
    monitor: int,
) -> DescribeScreenTool:
    """Build DescribeScreenTool with the chosen vision provider."""
    if tool_choice == "openai":
        vision = OpenAIVisionProvider(
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
            )
        )
    return DescribeScreenTool(vision, monitor=monitor)


def _get_voice_name(voice_choice: str) -> str:
    """Get the voice preset for the chosen provider."""
    if voice_choice == "openai":
        return os.environ.get("OPENAI_VOICE", "alloy")
    return os.environ.get("GEMINI_VOICE", "Aoede")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        print("GOOGLE_API_KEY is required (periodic screen vision).")
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
    system_prompt = _build_system_prompt(lang)

    kit = RoomKit()

    # --- Screen capture ------------------------------------------------------
    monitor = int(os.environ.get("MONITOR", "1"))
    vision_interval = int(os.environ.get("VISION_INTERVAL", "10000"))

    scale = float(os.environ.get("SCALE", "0.75"))
    screen_backend = ScreenCaptureBackend(
        monitor=monitor,
        fps=2,
        scale=scale,
        diff_threshold=0.02,
    )

    vision = _build_periodic_vision(google_api_key)

    video_channel = VideoChannel(
        "video-screen",
        backend=screen_backend,
        vision=vision,
        vision_interval_ms=vision_interval,
    )
    kit.register_channel(video_channel)

    # --- On-demand vision + input tools ---------------------------------------
    screen_tool = _build_screen_tool(tool_choice, google_api_key, monitor)
    input_tools = ScreenInputTools()

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

    # Full-duplex by default: mic stays open while AI speaks so the user
    # can interrupt at any time.  AEC handles echo cancellation.
    # Set MUTE_MIC=1 to force half-duplex if echo is a problem.
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
    all_tools = [*screen_tool.definitions, *input_tools.definitions]

    async def tool_handler(session: object, name: str, arguments: dict[str, object]) -> str:
        """Route tool calls to the right handler."""
        if name in ("describe_screen", "locate_element"):
            return await screen_tool.handler(session, name, arguments)  # type: ignore[arg-type]
        return await input_tools.handler(session, name, arguments)  # type: ignore[arg-type]

    voice_channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=audio_backend,
        system_prompt=system_prompt,
        voice=voice_name,
        input_sample_rate=sample_rate,
        tools=all_tools,
        tool_handler=tool_handler,
        mute_on_tool_call=True,
    )
    kit.register_channel(voice_channel)

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="screen-assistant")
    await kit.attach_channel("screen-assistant", "video-screen")
    await kit.attach_channel("screen-assistant", "voice")

    # --- Wire vision results → voice session via inject_text -----------------
    frame_count = 0
    last_inject_time = 0.0
    pending_vision: str | None = None
    min_inject_interval = 15.0

    @kit.on("video_vision_result")
    async def on_vision(event: object) -> None:
        nonlocal frame_count, pending_vision
        data = event.data  # type: ignore[attr-defined]
        if event.room_id != "screen-assistant":  # type: ignore[attr-defined]
            return

        description = data.get("description", "")
        if not description:
            return

        frame_count += 1
        labels = data.get("labels", [])
        text = data.get("text")
        elapsed = data.get("elapsed_ms", 0)

        desc_short = description[:200] + "..." if len(description) > 200 else description
        logger.info("[Vision %d] (%dms) %s", frame_count, elapsed, desc_short)
        if text:
            logger.info("[Vision %d] OCR: %s", frame_count, text[:200])

        parts = [f"[Screen update] {description}"]
        if labels:
            parts.append(f"Elements: {', '.join(labels)}")
        if text:
            parts.append(f"Visible text: {text}")
        pending_vision = "\n".join(parts)

        await _try_inject_vision()

    async def _try_inject_vision() -> None:
        nonlocal last_inject_time, pending_vision
        if pending_vision is None:
            return
        now = time.monotonic()
        if now - last_inject_time < min_inject_interval:
            logger.debug(
                "Deferring vision inject — too soon (%.0fs left)",
                min_inject_interval - (now - last_inject_time),
            )
            return

        sessions = voice_channel.get_room_sessions("screen-assistant")
        for session in sessions:
            try:
                await voice_channel.inject_text(session, pending_vision, role="user")
                last_inject_time = time.monotonic()
                logger.info("Injected vision context into voice session")
            except Exception:
                logger.exception("Failed to inject vision text")

        pending_vision = None

    async def _vision_flush_loop() -> None:
        while True:
            await asyncio.sleep(5.0)
            await _try_inject_vision()

    flush_task = asyncio.create_task(_vision_flush_loop())

    # --- Start sessions ------------------------------------------------------
    video_session = await kit.connect_video("screen-assistant", "local-user", "video-screen")
    await screen_backend.start_capture(video_session)

    # Gemini-specific VAD config (ignored by OpenAI provider)
    provider_config = {}
    if voice_choice == "gemini":
        provider_config = {
            "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
            "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
            "silence_duration_ms": 1500,
        }

    voice_session = await voice_channel.start_session(
        "screen-assistant",
        "local-user",
        connection=None,
        metadata={"provider_config": provider_config} if provider_config else None,
    )

    # --- Banner --------------------------------------------------------------
    print()
    voice_label = "OpenAI" if voice_choice == "openai" else "Gemini"
    tool_label = "OpenAI" if tool_choice == "openai" else "Gemini"
    print(f"Screen Assistant ({voice_label} Voice + {tool_label} Vision Tool)")
    print("=" * 60)
    lang_label = LANG_NAMES.get(lang, lang)
    print(f"Monitor: {monitor} | Vision every {vision_interval}ms | Scale {scale}")
    print(f"Language: {lang_label}")
    print(f"Voice: {voice_name} ({voice_label})")
    print(f"Vision tool: {tool_label}")
    print(f"AEC: {'enabled' if aec else 'disabled'}")
    print(f"Denoiser: {'enabled' if denoiser else 'disabled'}")
    print(f"Mic mute during playback: {'yes (half-duplex)' if mute_mic else 'no (full-duplex)'}")
    if not aec:
        print()
        print("  >>> Install AEC: pip install aec-audio-processing <<<")
    print()
    print("Task: Open Chrome → search 'roomkit python conversation ai' → go to roomkit.live")
    print()
    print("Speak into your microphone — the AI can see your screen!")
    print()
    print("TIP: Keep the target app (Chrome) visible — if this terminal covers")
    print("     it, the AI will see the terminal logs instead of Chrome.")
    print("     Use a second monitor, or tile windows side by side.")
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
    flush_task.cancel()
    await screen_backend.stop_capture(video_session)
    await voice_channel.end_session(voice_session)
    await kit.close()
    logger.info("Done. Vision analyzed %d frames.", frame_count)


if __name__ == "__main__":
    asyncio.run(main())
