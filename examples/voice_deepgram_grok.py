"""RoomKit -- Voice assistant with Deepgram STT + Claude Haiku + Grok TTS.

Talk to Claude through your microphone using Deepgram for speech-to-text
and xAI Grok for text-to-speech:

  Mic -> [Pipeline] -> Deepgram STT -> Claude Haiku -> Grok TTS -> Speaker

Requirements:
    pip install roomkit[local-audio,anthropic] deepgram-sdk websockets

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    XAI_API_KEY         (required) xAI API key for Grok TTS

    --- Voice (optional) ---
    LANGUAGE            Language for both STT and TTS (default: en)
    GROK_VOICE          Grok voice: eve | ara | rex | sal | leo (default: eve)
    SAMPLE_RATE         Audio sample rate in Hz (default: 16000)

    --- Pipeline (optional) ---
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)
    DENOISE             Enable RNNoise denoiser: 1 | 0 (default: 1)

    --- AI (optional) ---
    SYSTEM_PROMPT       Custom system prompt for Claude

Run with:
    ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... XAI_API_KEY=... \\
        uv run python examples/voice_deepgram_grok.py

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from roomkit import (
    AnthropicAIProvider,
    AnthropicConfig,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.grok import GrokTTSConfig, GrokTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_deepgram_grok")


def check_env() -> dict[str, str]:
    """Check required environment variables and return them."""
    keys = {
        "ANTHROPIC_API_KEY": "Anthropic (Claude)",
        "DEEPGRAM_API_KEY": "Deepgram (STT)",
        "XAI_API_KEY": "xAI (Grok TTS)",
    }
    values = {}
    missing = []
    for key, label in keys.items():
        val = os.environ.get(key, "")
        if not val:
            missing.append(f"  {key:24s} — {label}")
        values[key] = val

    if missing:
        print("Missing required environment variables:\n")
        print("\n".join(missing))
        print("\nExample:\n")
        print(
            "  ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... XAI_API_KEY=... \\\n"
            "    uv run python examples/voice_deepgram_grok.py"
        )
        sys.exit(1)

    return values


async def main() -> None:
    env = check_env()

    # --- Audio settings -------------------------------------------------------
    sample_rate = int(os.environ.get("SAMPLE_RATE", "16000"))
    block_ms = 20

    # --- AEC (echo cancellation) ----------------------------------------------
    aec = None
    aec_mode = os.environ.get("AEC", "webrtc").lower()
    if aec_mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        aec = WebRTCAECProvider(sample_rate=sample_rate)
        logger.info("AEC enabled (WebRTC AEC3)")
    elif aec_mode == "speex":
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        frame_size = sample_rate * block_ms // 1000
        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 10,
            sample_rate=sample_rate,
        )
        logger.info("AEC enabled (Speex)")

    # --- Denoiser (optional) --------------------------------------------------
    denoiser = None
    if os.environ.get("DENOISE", "1") == "1":
        from roomkit.voice.pipeline.denoiser.rnnoise import RNNoiseDenoiserProvider

        denoiser = RNNoiseDenoiserProvider(sample_rate=sample_rate)
        logger.info("Denoiser: RNNoise")

    # --- Backend: local mic + speakers ----------------------------------------
    mute_mic = aec is None  # mute mic during playback when no AEC
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        channels=1,
        block_duration_ms=block_ms,
        aec=aec,
        mute_mic_during_playback=mute_mic,
    )
    logger.info("Backend: LocalAudio (%dHz, mute_mic=%s)", sample_rate, mute_mic)

    kit = RoomKit()

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(aec=aec, denoiser=denoiser)

    # --- Language (shared by STT + TTS) ----------------------------------------
    language = os.environ.get("LANGUAGE", "en")

    # --- Deepgram STT ---------------------------------------------------------
    stt = DeepgramSTTProvider(
        config=DeepgramConfig(
            api_key=env["DEEPGRAM_API_KEY"],
            model="nova-2",
            language=language,
            punctuate=True,
            smart_format=True,
        )
    )
    logger.info("STT: Deepgram (nova-2, lang=%s)", language)

    # --- Grok TTS -------------------------------------------------------------
    grok_voice = os.environ.get("GROK_VOICE", "eve")
    tts = GrokTTSProvider(
        config=GrokTTSConfig(
            api_key=env["XAI_API_KEY"],
            voice_id=grok_voice,
            language=language,
            codec="pcm",
            sample_rate=sample_rate,
        )
    )
    logger.info("TTS: Grok (voice=%s, lang=%s, %dHz PCM)", grok_voice, language, sample_rate)

    # --- Claude Haiku AI ------------------------------------------------------
    ai_provider = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0.7,
        )
    )
    logger.info("AI: Claude Haiku (claude-haiku-4-5-20251001)")

    lang_instruction = f"\n\nAlways respond in {language}." if language != "en" else ""
    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly voice assistant. Keep your responses "
        "short and conversational — one or two sentences at most.\n\n"
        "Your TTS engine supports expressive speech tags. Use them "
        "naturally to make your speech lively and human:\n"
        "- Inline effects: [pause], [laugh], [sigh], [cough]\n"
        "- Wrapping tags: <whisper>...</whisper>, <soft>...</soft>, "
        "<loud>...</loud>, <slow>...</slow>, <fast>...</fast>, "
        "<high-pitch>...</high-pitch>, <low-pitch>...</low-pitch>\n\n"
        "Example: 'Oh [laugh] that's a great question! "
        "<soft>Let me think...</soft> [pause] Here's what I found.'\n"
        "Don't overuse them — sprinkle them in where they feel natural." + lang_instruction,
    )

    # --- Media recorder (optional) --------------------------------------------
    recording_dir = os.environ.get("RECORDING_DIR", "")
    recorders = None
    if recording_dir:
        from roomkit.recorder.base import MediaRecordingConfig, RoomRecorderBinding

        try:
            from roomkit.recorder.pyav import PyAVMediaRecorder

            recorder = PyAVMediaRecorder()
            logger.info("Recording: PyAV → MP4 in %s/", recording_dir)
        except ImportError:
            from roomkit.recorder.mock import MockMediaRecorder

            recorder = MockMediaRecorder()
            logger.info("Recording: Mock (pip install roomkit[video] for MP4)")
        recorders = [
            RoomRecorderBinding(
                recorder=recorder,
                config=MediaRecordingConfig(
                    storage=recording_dir,
                    audio_sample_rate=sample_rate,
                ),
                name="main",
            ),
        ]

    # --- Channels -------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="voice-demo", recorders=recorders)
    await kit.attach_channel("voice-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks: log speech lifecycle ------------------------------------------

    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You said: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Claude says: %s", text)
        return HookResult.allow()

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("voice-demo", "voice")

    logger.info("")
    logger.info("Speak into your microphone!")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
