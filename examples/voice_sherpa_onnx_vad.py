"""RoomKit -- Neural VAD + Denoiser with sherpa-onnx.

Demonstrates SherpaOnnxVADProvider and (optionally) SherpaOnnxDenoiserProvider
with a local microphone. The neural VAD replaces simple energy thresholding
with a proper speech detection model, giving much better accuracy in noisy
environments. The GTCRN denoiser cleans up the audio before VAD processing.

Supported VAD models:
  - TEN-VAD:  fast, low latency, good for real-time voice assistants
  - Silero VAD: widely used, slightly heavier

Prerequisites:
    pip install roomkit[local-audio,sherpa-onnx]

Download models:
    # VAD (required)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx
    # Denoiser (optional)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

Run with VAD only:
    VAD_MODEL=ten-vad.onnx uv run python examples/voice_sherpa_onnx_vad.py

Run with VAD + Denoiser:
    VAD_MODEL=ten-vad.onnx DENOISE_MODEL=gtcrn_simple.onnx \\
        uv run python examples/voice_sherpa_onnx_vad.py

Or with Silero:
    VAD_MODEL=silero_vad.onnx VAD_MODEL_TYPE=silero \\
        uv run python examples/voice_sherpa_onnx_vad.py

Environment variables:
    VAD_MODEL       (required) Path to VAD .onnx model file
    VAD_MODEL_TYPE  Model type: ten | silero (default: ten)
    VAD_THRESHOLD   Speech probability threshold 0-1 (default: 0.35)
    DENOISE_MODEL   Path to GTCRN denoiser .onnx model (optional)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
    MockAIProvider,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_sherpa_onnx_vad")


def check_env() -> str:
    """Verify VAD_MODEL is set and return the path."""
    model = os.environ.get("VAD_MODEL", "")
    if not model:
        print("VAD_MODEL environment variable is required.\n")
        print("Download a model first:")
        print(
            "  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx\n"
        )
        print("Then run:")
        print("  VAD_MODEL=ten-vad.onnx uv run python examples/voice_sherpa_onnx_vad.py")
        sys.exit(1)
    return model


async def main() -> None:
    model_path = check_env()
    model_type = os.environ.get("VAD_MODEL_TYPE", "ten")
    threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))

    kit = RoomKit()

    sample_rate = 16000

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=24000,
        channels=1,
        block_duration_ms=20,
    )

    # --- Neural VAD -----------------------------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=model_path,
            model_type=model_type,
            threshold=threshold,
            silence_threshold_ms=500,
            min_speech_duration_ms=250,
            speech_pad_ms=300,
            sample_rate=sample_rate,
        )
    )
    logger.info(
        "VAD: sherpa-onnx (model_type=%s, threshold=%.2f, model=%s)",
        model_type,
        threshold,
        model_path,
    )

    # --- Denoiser (optional, GTCRN via sherpa-onnx) ----------------------------
    denoise_model = os.environ.get("DENOISE_MODEL", "")
    denoiser = None
    if denoise_model:
        from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
            SherpaOnnxDenoiserConfig,
            SherpaOnnxDenoiserProvider,
        )

        denoiser = SherpaOnnxDenoiserProvider(SherpaOnnxDenoiserConfig(model=denoise_model))
        logger.info("Denoiser: sherpa-onnx GTCRN (model=%s)", denoise_model)

    # --- Pipeline config ------------------------------------------------------
    pipeline = AudioPipelineConfig(vad=vad, denoiser=denoiser)

    # --- STT + TTS (mock — swap with real providers for production) -----------
    stt = MockSTTProvider(transcripts=["Hello from sherpa-onnx VAD!"])
    tts = MockTTSProvider()

    # --- Channels -------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    ai = AIChannel(
        "ai",
        provider=MockAIProvider(responses=["Neural VAD detected your speech!"]),
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="vad-demo")
    await kit.attach_channel("vad-demo", "voice")
    await kit.attach_channel("vad-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started (neural VAD)")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended (neural VAD)")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text, ctx):
        from roomkit import HookResult

        logger.info("Transcription: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        from roomkit import HookResult

        logger.info("AI says: %s", text)
        return HookResult.allow()

    # --- Start voice session --------------------------------------------------
    session = await backend.connect("vad-demo", "local-user", "voice")
    binding = ChannelBinding(
        room_id="vad-demo",
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "vad-demo", binding)

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
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
