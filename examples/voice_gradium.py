"""RoomKit -- Cloud voice assistant with Gradium STT + TTS.

Talk to Claude through your microphone using Gradium for both
speech-to-text and text-to-speech:
  - Gradium for speech-to-text (streaming WebSocket)
  - Claude (Anthropic) for AI responses
  - Gradium for text-to-speech (streaming)
  - WebRTC or Speex AEC for echo cancellation
  - RNNoise or sherpa-onnx GTCRN for noise suppression
  - sherpa-onnx neural VAD (TEN-VAD or Silero) for speech detection
  - WavFileRecorder for debug audio capture

Audio flows through the full pipeline:

  Mic -> [Resampler] -> [Recorder tap] -> [AEC] -> [Denoiser] -> VAD
  -> Gradium STT -> Claude -> Gradium TTS -> [Recorder tap] -> Speaker

Requirements:
    pip install roomkit[local-audio,anthropic,gradium,sherpa-onnx]
    System (optional): libspeexdsp (apt install libspeexdsp1) for Speex AEC
    System (optional): librnnoise (apt install librnnoise0) -- or use DENOISE_MODEL

Run with:
    ANTHROPIC_API_KEY=... \\
    GRADIUM_API_KEY=... \\
    VAD_MODEL=path/to/ten-vad.onnx \\
    uv run python examples/voice_gradium.py

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    GRADIUM_API_KEY     (required) Gradium API key
    GRADIUM_REGION      API region (default: us)
    GRADIUM_STT_MODEL   STT model name (default: default)
    GRADIUM_TTS_MODEL   TTS model name (default: default)
    GRADIUM_VOICE_ID    Voice ID for TTS (default: default)
    LANGUAGE            Language code for STT (default: en)
    SYSTEM_PROMPT       Custom system prompt for Claude

    --- VAD (sherpa-onnx or none) ---
    VAD_MODEL           Path to sherpa-onnx VAD .onnx model file.
                        When unset AND VAD=0, no local VAD is used and
                        Gradium handles speech detection (continuous STT).
    VAD                 Set to 0 to disable local VAD entirely. Audio is
                        streamed continuously to Gradium STT which handles
                        endpointing server-side. (default: 1 if VAD_MODEL set)
    VAD_MODEL_TYPE      Model type: ten | silero (default: ten)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)
                        Lower values improve sensitivity for short utterances.
                        The GTCRN denoiser slightly alters spectral features,
                        which reduces TEN-VAD confidence -- 0.35 compensates.
                        Without denoiser you can raise to 0.5 for fewer false
                        positives.

    --- TTS (optional) ---
    TTS_SPEED           Speech speed: -4.0 (fastest) to 4.0 (slowest).
                        Default: unset (Gradium default ~0.0).
                        Try -1.0 for slightly faster conversational speech.

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
from roomkit.voice.stt.gradium import GradiumSTTConfig, GradiumSTTProvider
from roomkit.voice.tts.gradium import GradiumTTSConfig, GradiumTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_gradium")

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
        "GRADIUM_API_KEY": "Gradium (STT + TTS)",
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
            "  ANTHROPIC_API_KEY=... GRADIUM_API_KEY=... \\\n"
            "    uv run python examples/voice_gradium.py"
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
    # Gradium TTS with pcm_16000 outputs at 16kHz — no resampling needed
    output_rate = 16000

    # --- AEC (echo cancellation) ----------------------------------------------
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

    # --- VAD (sherpa-onnx neural VAD, energy fallback, or none) ---------------
    vad_disabled = os.environ.get("VAD", "").lower() == "0"
    vad = None
    vad_model = os.environ.get("VAD_MODEL", "")
    if vad_disabled:
        logger.info("VAD: disabled (continuous STT — Gradium handles endpointing)")
    elif vad_model:
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

    # --- Gradium STT ----------------------------------------------------------
    region = os.environ.get("GRADIUM_REGION", "us")
    vad_turn_steps = int(os.environ.get("VAD_TURN_STEPS", "6"))
    stt = GradiumSTTProvider(
        config=GradiumSTTConfig(
            api_key=env["GRADIUM_API_KEY"],
            region=region,
            model_name=os.environ.get("GRADIUM_STT_MODEL", "default"),
            input_format="pcm",
            vad_turn_steps=vad_turn_steps,
        )
    )
    logger.info(
        "STT: Gradium (region=%s, model=%s, vad_steps=%d)",
        region,
        stt._config.model_name,
        vad_turn_steps,
    )

    # --- Gradium TTS ----------------------------------------------------------
    padding_bonus_env = os.environ.get("TTS_SPEED", "")
    padding_bonus = float(padding_bonus_env) if padding_bonus_env else None
    language = os.environ.get("LANGUAGE", "en")
    tts = GradiumTTSProvider(
        config=GradiumTTSConfig(
            api_key=env["GRADIUM_API_KEY"],
            voice_id=os.environ.get("GRADIUM_VOICE_ID", "default"),
            region=region,
            model_name=os.environ.get("GRADIUM_TTS_MODEL", "default"),
            output_format=f"pcm_{output_rate}",
            padding_bonus=padding_bonus,
            rewrite_rules=language if language != "en" else None,
        )
    )
    logger.info(
        "TTS: Gradium (voice=%s, format=pcm_%d, speed=%s, rewrite=%s)",
        tts._config.voice_id,
        output_rate,
        padding_bonus,
        language if language != "en" else None,
    )

    # --- Claude AI ------------------------------------------------------------
    ai_provider = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0.7,
        )
    )
    logger.info("AI: Claude (claude-haiku-4-5-20251001)")

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
        ai_provider=ai_provider,
        ai_channel_id="ai",
        ai_system_prompt=system_prompt,
        ai_temperature=0.7,
        ai_max_tokens=256,
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
