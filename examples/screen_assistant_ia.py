"""RoomKit — AI screen assistant with Gemini speech-to-speech.

Talk to Gemini while it sees your screen in real time. The AI observes
what's on your display and guides you through tasks step by step using
natural voice conversation.

Uses Gemini Live for voice + periodic screen vision, and OpenAI GPT-4o
for the on-demand ``describe_screen`` tool (higher accuracy for text/icons).

Demo task: open Google Chrome, search "roomkit python conversation ai",
and navigate to the RoomKit website.

Requirements:
    pip install roomkit[screen-capture,realtime-gemini,local-audio,gemini,openai,sherpa-onnx]
    pip install aec-audio-processing    # WebRTC echo cancellation (CRITICAL)

Run with:
    GOOGLE_API_KEY=... OPENAI_API_KEY=... uv run python examples/screen_assistant_ia.py

Environment variables:
    GOOGLE_API_KEY       (required) Google API key (voice + periodic vision)
    OPENAI_API_KEY       (required) OpenAI API key (describe_screen tool)
    GEMINI_MODEL         Speech model (default: gemini-2.5-flash-native-audio-preview-12-2025)
    GEMINI_VOICE         Voice preset (default: Aoede)
    GEMINI_VISION_MODEL  Periodic vision model (default: gemini-3.1-flash-image-preview)
    OPENAI_VISION_MODEL  Tool vision model (default: gpt-4o)
    SCALE                Capture scale 0.0-1.0 (default: 0.75, higher = sharper text)
    AEC                  Echo cancellation: webrtc | speex | 0 (default: webrtc)
    DENOISE              Noise suppression: 1 | 0 (default: 1, uses sherpa-onnx GTCRN)
    DENOISE_MODEL        ONNX model file (default: gtcrn_simple.onnx)
    MUTE_MIC             Mute mic during AI playback: 1 | 0 (default: 1)
    LANG_VOICE           Language for voice conversation (default: en)
                         Examples: fr, es, de, it, pt, ja, ko, zh, ar, ru
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
    VideoChannel,
)
from roomkit.providers.gemini.realtime import GeminiLiveProvider
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


def _build_system_prompt(lang: str) -> str:
    """Build system prompt, optionally in a specific language."""
    lang_name = LANG_NAMES.get(lang, lang)
    lang_instruction = (
        f"\nYou MUST speak and respond ONLY in {lang_name}. Never switch to another language."
        if lang != "en"
        else ""
    )

    return f"""\
You are a helpful screen assistant. You can see the user's screen in real time \
and you are having a voice conversation with them.{lang_instruction}

Your current task: guide the user to open Google Chrome, type \
"roomkit python conversation ai" in the search bar, and navigate to the \
RoomKit website (roomkit.live).

## How you see the screen

You have TWO ways to see the screen:
1. **Periodic updates**: You receive automatic screen descriptions every ~10 seconds \
as text messages. These give you general context. Do NOT reply to every update — \
only speak when the user talks to you or when you need to give the next step.
2. **describe_screen tool**: You can call this tool at ANY time to ask a specific \
question about what is currently visible on the screen.

## CRITICAL RULES for using describe_screen

- **NEVER guess or assume** what is on the screen. If the user asks about something \
visual (an icon, a button, a menu, text, a position), you MUST call describe_screen \
first, then answer based on the result.
- When the user asks "where is X?", call describe_screen with a query like \
"Where is the X icon located? Describe its exact position relative to other elements."
- When the user asks "is X next to Y?" or "is X before Y?", call describe_screen \
with a query like "List all visible icons/elements in order from left to right \
(or top to bottom). Include their positions."
- When you need to give a precise instruction ("click on the third icon"), call \
describe_screen first to verify what you're about to say is correct.
- Ask precise, detailed questions in the tool. Bad: "describe the screen". \
Good: "List all icons in the taskbar from left to right with their names and positions."
- After calling the tool, relay the information naturally in conversation. \
Don't say "I used my tool" — just say "I can see that..." or "Looking at your screen...".

## Guiding the user

- Give short, clear voice directions one step at a time.
- Wait for the user to complete each step before moving to the next.
- Use describe_screen to confirm progress (e.g. verify Chrome is open, verify the \
search query is correct, verify the right website loaded).
- If the user asks for help finding something, ALWAYS use describe_screen to look \
for it, then give precise directions (e.g. "Chrome is the third icon from the left \
in your taskbar, right after the file manager").
- Be encouraging and patient.
- Speak naturally — you are a voice assistant, not a text bot.\
"""


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


def _build_vision(api_key: str) -> GeminiVisionProvider:
    """Build Gemini vision provider for screen analysis."""
    return GeminiVisionProvider(
        GeminiVisionConfig(
            api_key=api_key,
            model=os.environ.get("GEMINI_VISION_MODEL", "gemini-3.1-flash-image-preview"),
            prompt=(
                "Describe what is shown on this screen in 2-3 sentences. "
                "Focus on the FOREGROUND application (ignore terminal/log windows). "
                "Include the application name, visible text, URLs, and what "
                "the user appears to be doing. Be concise."
            ),
        )
    )


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    run_cmd = "GOOGLE_API_KEY=... OPENAI_API_KEY=... uv run python examples/screen_assistant_ia.py"
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")
        print(f"  {run_cmd}")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY for the describe_screen tool.")
        print(f"  {run_cmd}")
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

    vision = _build_vision(api_key)

    video_channel = VideoChannel(
        "video-screen",
        backend=screen_backend,
        vision=vision,
        vision_interval_ms=vision_interval,
    )
    kit.register_channel(video_channel)

    # --- On-demand vision tool -----------------------------------------------
    # The voice agent can call describe_screen(query="...") to get a detailed
    # analysis of the current screen with a specific question.
    #
    # IMPORTANT: The tool captures a FRESH full-resolution screenshot (not the
    # downscaled periodic frames) so the vision model can read text accurately.
    # A fresh provider is created per call so the query becomes the vision prompt.

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_vision_model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")

    screen_tool = DescribeScreenTool(
        lambda query: OpenAIVisionProvider(
            OpenAIVisionConfig(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1",
                model=openai_vision_model,
                prompt=query,
                max_tokens=4096,
                detail="high",
            )
        ),
        monitor=monitor,
    )

    # --- Gemini speech-to-speech ---------------------------------------------
    sample_rate = 24000
    block_ms = 20

    provider = GeminiLiveProvider(
        api_key=api_key,
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
    )

    aec = _build_aec(sample_rate, block_ms)
    denoiser = _build_denoiser()

    pipeline = None
    if aec or denoiser:
        from roomkit.voice.pipeline.config import AudioPipelineConfig

        pipeline = AudioPipelineConfig(aec=aec, denoiser=denoiser)

    # Always mute mic while AI is speaking.  Even with AEC, residual echo
    # is enough to trigger Gemini's server-side VAD → constant barge-in.
    # This makes conversation half-duplex (user waits for AI to finish)
    # but eliminates false interruptions.  Set MUTE_MIC=0 to try full-duplex.
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else True
    audio_backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
        pipeline=pipeline,
    )

    voice_channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=audio_backend,
        system_prompt=system_prompt,
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
        input_sample_rate=sample_rate,
        tools=[screen_tool.definition],
        tool_handler=screen_tool.handler,
        mute_on_tool_call=True,  # Mute mic during vision API call
    )
    kit.register_channel(voice_channel)

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="screen-assistant")
    await kit.attach_channel("screen-assistant", "video-screen")
    await kit.attach_channel("screen-assistant", "voice")

    # --- Wire vision results → voice session via inject_text -----------------
    # Instead of reconfigure_session (which disconnects/reconnects and kills
    # ongoing speech), we inject vision updates as text messages.  The system
    # prompt tells the AI to use these silently as context.
    #
    # Debounce: at most one injection per min_inject_interval seconds.
    # If the AI is generating audio, we defer the injection until later.
    frame_count = 0
    last_inject_time = 0.0
    pending_vision: str | None = None
    min_inject_interval = 15.0  # seconds — don't flood the conversation

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

        # Log vision result
        desc_short = description[:200] + "..." if len(description) > 200 else description
        logger.info("[Vision %d] (%dms) %s", frame_count, elapsed, desc_short)
        if text:
            logger.info("[Vision %d] OCR: %s", frame_count, text[:200])

        # Build screen context message
        parts = [f"[Screen update] {description}"]
        if labels:
            parts.append(f"Elements: {', '.join(labels)}")
        if text:
            parts.append(f"Visible text: {text}")
        pending_vision = "\n".join(parts)

        # Try to inject now if enough time has passed
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

    # Periodically flush any pending vision that was deferred
    async def _vision_flush_loop() -> None:
        while True:
            await asyncio.sleep(5.0)
            await _try_inject_vision()

    flush_task = asyncio.create_task(_vision_flush_loop())

    # --- Start sessions ------------------------------------------------------
    video_session = await kit.connect_video("screen-assistant", "local-user", "video-screen")
    await screen_backend.start_capture(video_session)

    voice_session = await voice_channel.start_session(
        "screen-assistant",
        "local-user",
        connection=None,
        metadata={
            "provider_config": {
                # Reduce VAD sensitivity — ambient noise won't trigger barge-in.
                # Full enum values: START_SENSITIVITY_LOW, END_SENSITIVITY_LOW
                "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
                "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
                # Wait 1500ms of silence before considering the turn done.
                "silence_duration_ms": 1500,
            },
        },
    )

    # --- Banner --------------------------------------------------------------
    print()
    print("Screen Assistant (Gemini Speech-to-Speech + OpenAI Vision Tool)")
    print("=" * 60)
    lang_label = LANG_NAMES.get(lang, lang)
    print(f"Monitor: {monitor} | Vision every {vision_interval}ms | Scale {scale}")
    print(f"Language: {lang_label}")
    print(f"Voice: {os.environ.get('GEMINI_VOICE', 'Aoede')}")
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
