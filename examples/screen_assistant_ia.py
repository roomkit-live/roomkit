"""RoomKit — AI screen assistant with speech-to-speech voice.

Talk to an AI while it sees your screen. The AI checks the screen
on demand (via the describe_screen tool) and can type/press keys
on your keyboard. It guides you verbally and asks permission
before taking any action.

Supports **OpenAI Realtime** or **Gemini Live** for voice, and
**OpenAI** or **Gemini** for the ``describe_screen`` vision tool.

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
    AEC                  Echo cancellation: webrtc | speex | 0 (default: webrtc)
    DENOISE              Noise suppression: 1 | 0 (default: 1)
    MUTE_MIC             Mute mic during AI playback: 1 | 0 (default: 0)
    LANG_VOICE           Language (default: en)
    MONITOR              Monitor index: 1=primary (default: 1)

Press Ctrl+C to stop.
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
    OpenAIVisionConfig,
    OpenAIVisionProvider,
    RealtimeVoiceChannel,
    RoomKit,
    ScreenInputTools,
)
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


def _build_system_prompt(lang: str) -> str:
    """Build system prompt, optionally in a specific language."""
    lang_name = LANG_NAMES.get(lang, lang)
    lang_instruction = f"\nIMPORTANT: You MUST speak ONLY in {lang_name}." if lang != "en" else ""

    return f"""\
You are a professional IT support assistant helping users via voice. \
You can check the user's screen and type on their keyboard.{lang_instruction}

## Your mission

Help the user navigate to roomkit.live. Start by checking their screen, \
then give short step-by-step instructions.

## Rules

- **Be concise.** One short sentence per step. No repetition.
- **Do NOT speak unless needed.** Only talk when giving the next step, \
answering a question, or confirming progress.
- **Use describe_screen only when you need to check progress** or when \
the user asks about something on screen. Do NOT call it after every step.
- **Ask permission once** before typing. Example: "Can I type the URL \
for you?" — then proceed without re-asking for each keystroke.
- **Guide with keyboard shortcuts**, not mouse. Example workflow:
  1. "Please open your browser."
  2. (user confirms) → press_key('ctrl+l') to focus address bar
  3. type_text('roomkit.live')
  4. press_key('enter')
  5. describe_screen to verify the page loaded
- **The user can interrupt you at any time.** Stop and listen.
- **Never repeat what you just said.** If the user didn't respond, wait.\
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
            print("\n  >>> Install AEC: pip install aec-audio-processing <<<\n")
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
            print("\n  >>> Install Speex: apt install libspeexdsp1 <<<\n")
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
    system_prompt = _build_system_prompt(lang)

    kit = RoomKit()

    # --- Tools ---------------------------------------------------------------
    monitor = int(os.environ.get("MONITOR", "1"))
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

    # Full-duplex: mic stays open so user can interrupt.
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
    all_tools = [screen_tool.definition, *input_tools.definitions]

    async def tool_handler(session: object, name: str, arguments: dict[str, object]) -> str:
        """Route tool calls to the right handler."""
        if name == "describe_screen":
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

    # --- Room + session ------------------------------------------------------
    await kit.create_room(room_id="screen-assistant")
    await kit.attach_channel("screen-assistant", "voice")

    # Gemini-specific VAD config
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
    print()
    print(f"Screen Assistant ({voice_label} Voice + {tool_label} Vision)")
    print("=" * 60)
    print(f"Voice: {voice_name} | Language: {LANG_NAMES.get(lang, lang)}")
    print(f"AEC: {'on' if aec else 'off'} | Denoiser: {'on' if denoiser else 'off'}")
    print(f"Interruption: {'off (half-duplex)' if mute_mic else 'on (full-duplex)'}")
    print()
    print("The AI checks your screen on demand (no periodic injection).")
    print("Tools: describe_screen, type_text, press_key")
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
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
