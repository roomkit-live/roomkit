"""RoomKit -- Full-stack voice assistant with local microphone.

Talk to Claude through your microphone with real audio processing:
  - Deepgram for speech-to-text (streaming)
  - Claude (Anthropic) for AI responses
  - ElevenLabs for text-to-speech
  - SpeexAEC for echo cancellation (strips speaker echo from mic)
  - RNNoise for noise suppression
  - WavFileRecorder for debug audio capture

Audio flows through the full pipeline:

  Mic → [Resampler] → [Recorder tap] → [AEC] → [Denoiser] → VAD
  → Deepgram STT → Claude → ElevenLabs TTS → [Recorder tap] → Speaker

Requirements:
    pip install roomkit[local-audio,anthropic]
    System: libspeexdsp (apt install libspeexdsp1 / brew install speexdsp)
    System (optional): librnnoise (apt install librnnoise0)

Run with:
    ANTHROPIC_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    uv run python examples/voice_full_stack.py

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    ELEVENLABS_API_KEY  (required) ElevenLabs API key
    ELEVENLABS_VOICE_ID Voice ID (default: Rachel)
    LANGUAGE            Language code for STT (default: en)
    SYSTEM_PROMPT       Custom system prompt for Claude
    RECORDING_DIR       Directory for WAV recordings (default: ./recordings)
    RECORDING_MODE      Channel mode: mixed | separate | stereo (default: stereo)
    AEC                 Enable echo cancellation: 1 | 0 (default: 1)
    DENOISE             Enable noise suppression: 1 | 0 (default: 1)

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
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    RecordingChannelMode,
    RecordingConfig,
    WavFileRecorder,
)
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_full_stack")

# Channel mode mapping
CHANNEL_MODES = {
    "mixed": RecordingChannelMode.MIXED,
    "separate": RecordingChannelMode.SEPARATE,
    "stereo": RecordingChannelMode.STEREO,
}


def check_env() -> dict[str, str]:
    """Check required environment variables and return them."""
    keys = {
        "ANTHROPIC_API_KEY": "Anthropic (Claude)",
        "DEEPGRAM_API_KEY": "Deepgram (STT)",
        "ELEVENLABS_API_KEY": "ElevenLabs (TTS)",
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
            "  ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... "
            "ELEVENLABS_API_KEY=... \\\n"
            "    uv run python examples/voice_full_stack.py"
        )
        sys.exit(1)

    return values


async def main() -> None:
    env = check_env()

    kit = RoomKit()

    # --- Audio settings -------------------------------------------------------
    sample_rate = 16000
    block_ms = 20
    frame_size = sample_rate * block_ms // 1000  # 320 samples

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=24000,  # ElevenLabs default output rate
        channels=1,
        block_duration_ms=block_ms,
    )

    # --- AEC (Speex echo cancellation) ----------------------------------------
    use_aec = os.environ.get("AEC", "1") == "1"
    aec = None
    if use_aec:
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 10,  # 200ms echo tail
            sample_rate=sample_rate,
        )
        logger.info("AEC enabled (Speex, filter=%d samples)", frame_size * 10)

    # --- Denoiser (RNNoise noise suppression) ---------------------------------
    use_denoise = os.environ.get("DENOISE", "1") == "1"
    denoiser = None
    if use_denoise:
        from roomkit.voice.pipeline.denoiser.rnnoise import (
            RNNoiseDenoiserProvider,
        )

        denoiser = RNNoiseDenoiserProvider(sample_rate=sample_rate)
        logger.info("Denoiser enabled (RNNoise)")

    # --- WAV recorder (debug audio capture) -----------------------------------
    recording_dir = os.environ.get("RECORDING_DIR", "./recordings")
    rec_mode_name = os.environ.get("RECORDING_MODE", "stereo").lower()
    rec_channel_mode = CHANNEL_MODES.get(
        rec_mode_name, RecordingChannelMode.STEREO
    )

    recorder = WavFileRecorder()
    recording_config = RecordingConfig(
        storage=recording_dir,
        channels=rec_channel_mode,
    )
    logger.info(
        "Recording to %s (mode=%s)", recording_dir, rec_mode_name
    )

    # --- VAD (energy-based) ---------------------------------------------------
    vad = EnergyVADProvider(
        energy_threshold=300.0,
        silence_threshold_ms=600,
        min_speech_duration_ms=200,
    )

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        aec=aec,
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
    )

    # --- Deepgram STT ---------------------------------------------------------
    language = os.environ.get("LANGUAGE", "en")
    stt = DeepgramSTTProvider(
        config=DeepgramConfig(
            api_key=env["DEEPGRAM_API_KEY"],
            model="nova-2",
            language=language,
            punctuate=True,
            smart_format=True,
            endpointing=300,
        )
    )
    logger.info("STT: Deepgram nova-2 (language=%s)", language)

    # --- ElevenLabs TTS -------------------------------------------------------
    tts = ElevenLabsTTSProvider(
        config=ElevenLabsConfig(
            api_key=env["ELEVENLABS_API_KEY"],
            voice_id=os.environ.get(
                "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
            ),
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",  # Raw PCM at 24kHz for local playback
            optimize_streaming_latency=3,
        )
    )
    logger.info("TTS: ElevenLabs (voice=%s)", tts._config.voice_id)

    # --- Claude AI ------------------------------------------------------------
    ai_provider = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            temperature=0.7,
        )
    )
    logger.info("AI: Claude (claude-sonnet-4-20250514)")

    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly voice assistant. Keep your responses "
        "short and conversational — one or two sentences at most.",
    )

    # --- Voice channel --------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=system_prompt,
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="voice-demo")
    await kit.attach_channel("voice-demo", "voice")
    await kit.attach_channel(
        "voice-demo", "ai", category=ChannelCategory.INTELLIGENCE
    )

    # --- Hooks ----------------------------------------------------------------

    @kit.hook(
        HookTrigger.ON_SPEECH_START,
        execution=HookExecution.ASYNC,
    )
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(
        HookTrigger.ON_SPEECH_END,
        execution=HookExecution.ASYNC,
    )
    async def on_speech_end(session, ctx):
        logger.info("Speech ended")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text, ctx):
        from roomkit import HookResult

        logger.info("You said: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        from roomkit import HookResult

        logger.info("Claude says: %s", text)
        return HookResult.allow()

    @kit.hook(
        HookTrigger.ON_RECORDING_STARTED,
        execution=HookExecution.ASYNC,
    )
    async def on_rec_started(event, ctx):
        logger.info("Recording started: %s", event.id)

    @kit.hook(
        HookTrigger.ON_RECORDING_STOPPED,
        execution=HookExecution.ASYNC,
    )
    async def on_rec_stopped(event, ctx):
        logger.info(
            "Recording stopped: %s (%.1fs, %d bytes, files=%s)",
            event.id,
            event.duration_seconds,
            event.size_bytes,
            event.urls,
        )

    # --- Start voice session --------------------------------------------------
    session = await backend.connect("voice-demo", "local-user", "voice")
    binding = ChannelBinding(
        room_id="voice-demo",
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "voice-demo", binding)

    logger.info("")
    logger.info("Speak into your microphone!")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    await backend.start_listening(session)

    # --- Keep running until Ctrl+C --------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    await backend.stop_listening(session)
    voice.unbind_session(session)
    await asyncio.sleep(0.1)
    await backend.disconnect(session)
    await kit.close()
    logger.info("Done. Recordings saved to: %s", recording_dir)


if __name__ == "__main__":
    asyncio.run(main())
