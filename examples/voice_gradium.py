"""RoomKit -- Cloud voice assistant with Gradium STT + TTS.

Talk to Claude through your microphone using Gradium for both
speech-to-text and text-to-speech:
  - Gradium for speech-to-text (streaming WebSocket, server-side VAD)
  - Claude (Anthropic) for AI responses
  - Gradium for text-to-speech (streaming)
  - WebRTC or Speex AEC for echo cancellation
  - RNNoise or sherpa-onnx GTCRN for noise suppression
  - WavFileRecorder for debug audio capture (opt-in)

Audio flows through the full pipeline:

  Mic -> [Resampler] -> [Recorder tap] -> [AEC] -> [Denoiser]
  -> Gradium STT (continuous, server-side endpointing)
  -> Claude -> Gradium TTS -> [Recorder tap] -> Speaker

No local VAD is needed — Gradium handles speech detection and
turn-taking server-side via semantic VAD (inactivity_prob).

Requirements:
    pip install roomkit[local-audio,anthropic,gradium,webrtc-aec]
    System (optional): libspeexdsp (apt install libspeexdsp1) for Speex AEC
    System (optional): librnnoise (apt install librnnoise0) for the default
                       RNNoise denoiser -- without it the denoiser is skipped

Run with:
    ANTHROPIC_API_KEY=... \\
    GRADIUM_API_KEY=... \\
    uv run python examples/voice_gradium.py

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    GRADIUM_API_KEY     (required) Gradium API key
    GRADIUM_REGION      API region (default: us)
    GRADIUM_STT_MODEL   STT model name (default: default)
    GRADIUM_TTS_MODEL   TTS model name (default: default)
    GRADIUM_VOICE_ID    Voice ID for TTS (default: default)
    VOICE_LANGUAGE      Language code for STT (default: en); any other
                        language also turns on the TTS rewrite rules for it
    SYSTEM_PROMPT       Custom system prompt for Claude
    CONSOLE             1 shows the RoomKit console dashboard (default: 0)

    --- TTS (optional) ---
    TTS_SPEED           Speech speed: -4.0 (fastest) to 4.0 (slowest).
                        Default: unset (Gradium default ~0.0).
                        Try -1.0 for slightly faster conversational speech.

    --- Pipeline (optional) ---
    AEC                 Echo cancellation: webrtc | speex | 1 (=webrtc) | 0
                        (default: webrtc)
    DENOISE             Noise suppression: rnnoise | sherpa | webrtc |
                        1 (=rnnoise) | 0 (default: rnnoise; skipped with a
                        warning when its library is missing)
    DENOISE_MODEL       GTCRN .onnx model for DENOISE=sherpa
                        (default: gtcrn_simple.onnx)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto,
                        off with AEC)
    RECORDING_DIR       Record the call as WAV files into this directory
                        (default: unset, no recording)
    RECORDING_ENCRYPTED_AT_REST
                        Required with RECORDING_DIR: set it to 1 to state
                        that RECORDING_DIR is on encrypted storage. RoomKit
                        refuses plaintext recordings (RFC 17.6) and cannot
                        check the claim itself.
    RECORDING_MODE      Channel mode: mixed | separate | stereo (default: stereo)
    DEBUG_TAPS_DIR      Directory for pipeline debug taps (disabled if unset)
    DEBUG_TAPS_STAGES   Comma-separated stages to capture (default: all)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    build_denoiser,
    env_bool,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
    voice_language,
)

from roomkit import ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    DenoiserProvider,
    PipelineDebugTaps,
    RecordingChannelMode,
    RecordingConfig,
    WavFileRecorder,
)
from roomkit.voice.stt.gradium import GradiumSTTConfig, GradiumSTTProvider
from roomkit.voice.tts.gradium import GradiumTTSConfig, GradiumTTSProvider

logger = setup_logging("voice_gradium")
logging.getLogger("roomkit.voice.stt.gradium").setLevel(logging.DEBUG)

# Channel mode mapping
CHANNEL_MODES = {
    "mixed": RecordingChannelMode.MIXED,
    "separate": RecordingChannelMode.SEPARATE,
    "stereo": RecordingChannelMode.STEREO,
}


def build_recording() -> tuple[WavFileRecorder | None, RecordingConfig | None]:
    """WAV recording of the call, off unless RECORDING_DIR is set.

    RFC 17.6 makes encryption at rest a MUST, so WavFileRecorder refuses to
    start without a RecordingEncryption or a statement that the storage
    encrypts at rest. RECORDING_ENCRYPTED_AT_REST=1 is that statement: yours,
    about RECORDING_DIR, which the example cannot verify.
    """
    recording_dir = os.environ.get("RECORDING_DIR", "")
    if not recording_dir:
        return None, None
    if not env_bool("RECORDING_ENCRYPTED_AT_REST", default=False):
        print(
            "Error: RECORDING_DIR needs RECORDING_ENCRYPTED_AT_REST=1, stating that "
            f"{recording_dir} is on encrypted storage (RFC 17.6 refuses plaintext recordings)"
        )
        sys.exit(1)
    mode_name = os.environ.get("RECORDING_MODE", "stereo").lower()
    config = RecordingConfig(
        storage=recording_dir,
        storage_encrypted_at_rest=True,  # stated by RECORDING_ENCRYPTED_AT_REST=1
        channels=CHANNEL_MODES.get(mode_name, RecordingChannelMode.STEREO),
    )
    logger.info("Recording to %s (mode=%s)", recording_dir, mode_name)
    return WavFileRecorder(), config


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY", "GRADIUM_API_KEY")

    kit = RoomKit()
    console_cleanup = setup_console(kit)

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
    # The backend owns the AEC: it feeds the speaker signal as the echo
    # reference block-aligned with playback, and reports NATIVE_AEC so the
    # pipeline does not run a second one.
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

    # --- Denoiser (RNNoise by default, or sherpa-onnx GTCRN / WebRTC NS) ------
    # The shared builder returns `object` to keep its imports lazy.
    denoiser = cast("DenoiserProvider | None", build_denoiser(sample_rate, default="rnnoise"))

    # --- WAV recorder (opt-in) ------------------------------------------------
    recorder, recording_config = build_recording()

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
    # No aec= here: the backend runs it (see above).
    pipeline_config = AudioPipelineConfig(
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
        debug_taps=debug_taps,
    )

    # --- Gradium STT ----------------------------------------------------------
    region = os.environ.get("GRADIUM_REGION", "us")
    language = voice_language("en")
    stt_model = os.environ.get("GRADIUM_STT_MODEL", "default")
    stt = GradiumSTTProvider(
        config=GradiumSTTConfig(
            api_key=env["GRADIUM_API_KEY"],
            region=region,
            model_name=stt_model,
            input_format="pcm",
            language=language,
            connect_buffer_ms=0,
        )
    )
    logger.info(
        "STT: Gradium (region=%s, model=%s, lang=%s)",
        region,
        stt_model,
        language,
    )

    # --- Gradium TTS ----------------------------------------------------------
    padding_bonus_env = os.environ.get("TTS_SPEED", "")
    padding_bonus = float(padding_bonus_env) if padding_bonus_env else None
    voice_id = os.environ.get("GRADIUM_VOICE_ID", "default")
    tts = GradiumTTSProvider(
        config=GradiumTTSConfig(
            api_key=env["GRADIUM_API_KEY"],
            voice_id=voice_id,
            region=region,
            model_name=os.environ.get("GRADIUM_TTS_MODEL", "default"),
            output_format=f"pcm_{output_rate}",
            padding_bonus=padding_bonus,
            rewrite_rules=language if language != "en" else None,
        )
    )
    logger.info(
        "TTS: Gradium (voice=%s, format=pcm_%d, speed=%s, rewrite=%s)",
        voice_id,
        output_rate,
        padding_bonus,
        language if language != "en" else None,
    )

    # --- Claude AI ------------------------------------------------------------
    ai_provider = AnthropicAIProvider(
        AnthropicConfig(
            api_key=env["ANTHROPIC_API_KEY"],
            model="claude-haiku-5-5",
            max_tokens=256,
        )
    )
    logger.info("AI: Claude (claude-haiku-5-5)")

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
    logger.info("Interruption: channel default (immediate)")
    kit.register_channel(voice)

    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=system_prompt,
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="voice-demo")
    await kit.attach_channel("voice-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------

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

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
    async def on_rec_started(event, ctx):
        logger.info("Recording started: %s", event.id)

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
    async def on_rec_stopped(event, ctx):
        logger.info(
            "Recording stopped: %s (%.1fs, files=%s)",
            event.id,
            event.duration_seconds,
            event.urls,
        )

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("voice-demo", "voice")

    logger.info("")
    logger.info("Speak into your microphone!")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        if recording_config is not None:
            logger.info("Recordings saved to: %s", recording_config.storage)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
