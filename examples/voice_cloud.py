"""RoomKit -- Cloud voice assistant with local microphone.

Talk to Claude through your microphone with cloud AI services:
  - Deepgram for speech-to-text (streaming)
  - Claude (Anthropic) for AI responses
  - ElevenLabs for text-to-speech
  - WebRTC or Speex AEC for echo cancellation
  - RNNoise or sherpa-onnx GTCRN for noise suppression
  - sherpa-onnx neural VAD (TEN-VAD or Silero) for speech detection
  - WavFileRecorder for debug audio capture

Audio flows through the full pipeline:

  Mic -> [Resampler] -> [Recorder tap] -> [AEC] -> [Denoiser] -> VAD
  -> Deepgram STT -> Claude -> ElevenLabs TTS -> [Recorder tap] -> Speaker

Requirements:
    pip install roomkit[local-audio,anthropic,sherpa-onnx]
    System (optional): libspeexdsp (apt install libspeexdsp1) for Speex AEC
    System (optional): librnnoise (apt install librnnoise0) -- or use DENOISE_MODEL

Run with:
    ANTHROPIC_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    VAD_MODEL=path/to/ten-vad.onnx \\
    uv run python examples/voice_cloud.py

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    ELEVENLABS_API_KEY  (required) ElevenLabs API key
    ELEVENLABS_VOICE_ID Voice ID (default: Rachel)
    LANGUAGE            Language code for STT (default: en)
    SYSTEM_PROMPT       Custom system prompt for Claude

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           Path to sherpa-onnx VAD .onnx model file
    VAD_MODEL_TYPE      Model type: ten | silero (default: ten)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)
                        Lower values improve sensitivity for short utterances.
                        The GTCRN denoiser slightly alters spectral features,
                        which reduces TEN-VAD confidence -- 0.35 compensates.
                        Without denoiser you can raise to 0.5 for fewer false
                        positives.

    --- Pipeline (optional) ---
    AEC                 Echo cancellation: webrtc | speex | 1 (=webrtc) | 0
                        (default: webrtc)
    DENOISE             Enable RNNoise noise suppression: 1 | 0 (default: 1)
    DENOISE_MODEL       Path to GTCRN .onnx model (sherpa-onnx denoiser,
                        overrides DENOISE)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto,
                        off with AEC)
    RECORDING_DIR       Directory for WAV recordings (default: ./recordings)
    RECORDING_MODE      Channel mode: mixed | separate | stereo (default: stereo)
    DEBUG_TAPS_DIR      Directory for pipeline debug taps (disabled if unset)
    DEBUG_TAPS_STAGES   Comma-separated stages to capture (default: all)

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
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    PipelineDebugTaps,
    RecordingChannelMode,
    RecordingConfig,
    WavFileRecorder,
)
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_cloud")

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
            "    uv run python examples/voice_cloud.py"
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
    output_rate = 24000  # ElevenLabs native rate

    # --- AEC (echo cancellation) ----------------------------------------------
    # Unified format: AEC=webrtc | speex | 1 (=webrtc) | 0
    aec = None
    aec_mode = os.environ.get("AEC", "webrtc").lower()
    if aec_mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        aec = WebRTCAECProvider(sample_rate=sample_rate)
        logger.info("AEC enabled (WebRTC AEC3)")
    elif aec_mode == "speex":
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 10,  # 200ms echo tail
            sample_rate=sample_rate,
        )
        logger.info("AEC enabled (Speex, filter=%d samples)", frame_size * 10)

    # --- Backend: local mic + speakers ----------------------------------------
    # When AEC is active it removes speaker echo from the mic signal, so we
    # can keep the mic open during playback and allow barge-in interruption.
    # Without AEC the mic is muted during playback to prevent feedback loops.
    # Override with MUTE_MIC=0|1 for testing.
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=output_rate,
        channels=1,
        block_duration_ms=block_ms,
        aec=aec,
        mute_mic_during_playback=mute_mic,
    )
    logger.info(
        "Backend: LocalAudio (in=%dHz, out=%dHz, mute_mic=%s)",
        sample_rate,
        output_rate,
        mute_mic,
    )

    # --- Denoiser (RNNoise or sherpa-onnx GTCRN) ------------------------------
    denoiser = None
    denoise_model = os.environ.get("DENOISE_MODEL", "")
    if denoise_model:
        from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
            SherpaOnnxDenoiserConfig,
            SherpaOnnxDenoiserProvider,
        )

        denoiser = SherpaOnnxDenoiserProvider(SherpaOnnxDenoiserConfig(model=denoise_model))
        logger.info("Denoiser: sherpa-onnx GTCRN (model=%s)", denoise_model)
    elif os.environ.get("DENOISE", "1") == "1":
        from roomkit.voice.pipeline.denoiser.rnnoise import (
            RNNoiseDenoiserProvider,
        )

        denoiser = RNNoiseDenoiserProvider(sample_rate=sample_rate)
        logger.info("Denoiser: RNNoise")

    # --- WAV recorder (debug audio capture) -----------------------------------
    recording_dir = os.environ.get("RECORDING_DIR", "./recordings")
    rec_mode_name = os.environ.get("RECORDING_MODE", "stereo").lower()
    rec_channel_mode = CHANNEL_MODES.get(rec_mode_name, RecordingChannelMode.STEREO)

    recorder = WavFileRecorder()
    recording_config = RecordingConfig(
        storage=recording_dir,
        channels=rec_channel_mode,
    )
    logger.info("Recording to %s (mode=%s)", recording_dir, rec_mode_name)

    # --- VAD (sherpa-onnx neural VAD or energy fallback) ----------------------
    vad_model = os.environ.get("VAD_MODEL", "")
    if vad_model:
        vad_model_type = os.environ.get("VAD_MODEL_TYPE", "ten")
        vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))
        vad = SherpaOnnxVADProvider(
            SherpaOnnxVADConfig(
                model=vad_model,
                model_type=vad_model_type,
                threshold=vad_threshold,
                silence_threshold_ms=600,
                min_speech_duration_ms=200,
                speech_pad_ms=300,
                sample_rate=sample_rate,
            )
        )
        logger.info(
            "VAD: sherpa-onnx (type=%s, threshold=%.2f, model=%s)",
            vad_model_type,
            vad_threshold,
            vad_model,
        )
    else:
        vad = EnergyVADProvider(
            energy_threshold=300.0,
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
        )
        logger.info("VAD: EnergyVAD (set VAD_MODEL for neural VAD)")

    # --- Debug taps (pipeline stage audio capture) ----------------------------
    debug_taps = None
    debug_taps_dir = os.environ.get("DEBUG_TAPS_DIR", "")
    if debug_taps_dir:
        stages_env = os.environ.get("DEBUG_TAPS_STAGES", "all")
        stages = [s.strip() for s in stages_env.split(",")]
        debug_taps = PipelineDebugTaps(
            output_dir=debug_taps_dir,
            stages=stages,
        )
        logger.info("Debug taps: %s (stages=%s)", debug_taps_dir, stages)

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        aec=aec,
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
        debug_taps=debug_taps,
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
    tts_format = f"pcm_{output_rate}"
    tts = ElevenLabsTTSProvider(
        config=ElevenLabsConfig(
            api_key=env["ELEVENLABS_API_KEY"],
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model_id="eleven_multilingual_v2",
            output_format=tts_format,
            optimize_streaming_latency=3,
        )
    )
    logger.info("TTS: ElevenLabs (voice=%s, format=%s)", tts._config.voice_id, tts_format)

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
    logger.info(
        "Interruption: strategy=%s, barge_in=%s",
        voice._interruption_handler._config.strategy.value,
        voice._enable_barge_in,
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
    await kit.attach_channel("voice-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------

    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text, ctx):
        logger.info("You said: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Claude says: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
    async def on_rec_started(event, ctx):
        logger.info("Recording started: %s", event.id)

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
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
