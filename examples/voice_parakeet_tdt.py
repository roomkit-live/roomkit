"""RoomKit -- Local voice assistant with NVIDIA Parakeet TDT STT.

Uses the NVIDIA Parakeet TDT 0.6B v3 model for high-accuracy offline
speech-to-text via sherpa-onnx.  Parakeet TDT is an offline model —
VAD segments speech, then each utterance is transcribed in a single
batch call.  It supports 25 European languages with automatic language
detection, punctuation, and capitalization.

Audio pipeline:

  Mic → [Denoiser] → VAD → Parakeet TDT STT → Local LLM → TTS → Speaker

Prerequisites:
    pip install roomkit[local-audio,openai,sherpa-onnx]

Download models:
    # STT — Parakeet TDT 0.6B v3 (int8 quantized, ~640 MB)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
    tar xf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2

    # VAD — TEN-VAD
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

    # TTS — VITS (Piper voices)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xf vits-piper-en_US-amy-low.tar.bz2

    # Denoiser — GTCRN (optional)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

Run:
    LLM_MODEL=qwen3:8b \
    LLM_BASE_URL=http://localhost:11434/v1 \
    VAD_MODEL=ten-vad.onnx \
    STT_ENCODER=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    STT_DECODER=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    STT_JOINER=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    STT_TOKENS=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
    TTS_MODEL=vits-piper-en_US-amy-low/en_US-amy-low.onnx \
    TTS_TOKENS=vits-piper-en_US-amy-low/tokens.txt \
    TTS_DATA_DIR=vits-piper-en_US-amy-low/espeak-ng-data \
    uv run python examples/voice_parakeet_tdt.py

Environment variables:
    LLM_MODEL       (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL    Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY     API key if server requires auth (default: none)
    VAD_MODEL       (required) Path to VAD .onnx model
    STT_ENCODER     (required) Path to Parakeet encoder .onnx
    STT_DECODER     (required) Path to Parakeet decoder .onnx
    STT_JOINER      (required) Path to Parakeet joiner .onnx
    STT_TOKENS      (required) Path to Parakeet tokens.txt
    TTS_MODEL       (required) Path to TTS VITS/Piper .onnx model
    TTS_TOKENS      (required) Path to TTS tokens.txt
    TTS_DATA_DIR    Path to espeak-ng data dir (Piper models)
    DENOISE_MODEL   Path to GTCRN denoiser .onnx (optional)
    ONNX_PROVIDER   ONNX provider for STT/TTS: cpu | cuda (default: cpu)

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
    HookResult,
    HookTrigger,
    RoomKit,
    VLLMConfig,
    VoiceChannel,
    create_vllm_provider,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_parakeet_tdt")


def check_env() -> None:
    """Check required environment variables."""
    required = {
        "LLM_MODEL": "Model name (e.g. qwen3:8b for Ollama)",
        "VAD_MODEL": "Path to VAD .onnx model (e.g. ten-vad.onnx)",
        "STT_ENCODER": "Path to Parakeet encoder .onnx",
        "STT_DECODER": "Path to Parakeet decoder .onnx",
        "STT_JOINER": "Path to Parakeet joiner .onnx",
        "STT_TOKENS": "Path to Parakeet tokens.txt",
        "TTS_MODEL": "Path to TTS .onnx model",
        "TTS_TOKENS": "Path to TTS tokens.txt",
    }
    missing = [
        f"  {key:20s} — {desc}" for key, desc in required.items() if not os.environ.get(key)
    ]
    if missing:
        print("Missing required environment variables:\n")
        print("\n".join(missing))
        print("\nSee docstring at the top of this file for download links and usage.")
        sys.exit(1)


async def main() -> None:
    check_env()

    kit = RoomKit()

    onnx_provider = os.environ.get("ONNX_PROVIDER", "cpu")
    sample_rate = 16000
    tts_sample_rate = 22050

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_sample_rate,
        channels=1,
        block_duration_ms=20,
        mute_mic_during_playback=True,
    )

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=os.environ["VAD_MODEL"],
            model_type=os.environ.get("VAD_MODEL_TYPE", "ten"),
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.35")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            speech_pad_ms=300,
            sample_rate=sample_rate,
        )
    )

    # --- Denoiser (optional) --------------------------------------------------
    denoiser = None
    denoise_model = os.environ.get("DENOISE_MODEL", "")
    if denoise_model:
        from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
            SherpaOnnxDenoiserConfig,
            SherpaOnnxDenoiserProvider,
        )

        denoiser = SherpaOnnxDenoiserProvider(SherpaOnnxDenoiserConfig(model=denoise_model))
        logger.info("Denoiser: GTCRN (%s)", denoise_model)

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(vad=vad, denoiser=denoiser)

    # --- STT: Parakeet TDT (offline, via sherpa-onnx) -------------------------
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig(
            mode="transducer",
            model_type="nemo_transducer",
            encoder=os.environ["STT_ENCODER"],
            decoder=os.environ["STT_DECODER"],
            joiner=os.environ["STT_JOINER"],
            tokens=os.environ["STT_TOKENS"],
            sample_rate=sample_rate,
            provider=onnx_provider,
        )
    )
    logger.info("STT: Parakeet TDT 0.6B v3 (offline, provider=%s)", onnx_provider)

    # --- TTS (sherpa-onnx VITS/Piper) -----------------------------------------
    tts = SherpaOnnxTTSProvider(
        SherpaOnnxTTSConfig(
            model=os.environ["TTS_MODEL"],
            tokens=os.environ["TTS_TOKENS"],
            data_dir=os.environ.get("TTS_DATA_DIR", ""),
            speaker_id=int(os.environ.get("TTS_SPEAKER_ID", "0")),
            sample_rate=tts_sample_rate,
            provider=onnx_provider,
        )
    )

    # --- LLM (local via OpenAI-compatible API) --------------------------------
    ai_provider = create_vllm_provider(
        VLLMConfig(
            model=os.environ["LLM_MODEL"],
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("LLM_API_KEY", "none"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
        )
    )
    logger.info("LLM: %s", os.environ["LLM_MODEL"])

    # --- Channels -------------------------------------------------------------
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
        system_prompt=(
            "You are a friendly voice assistant. Keep your responses "
            "short and conversational — one or two sentences at most. "
            "Answer directly without thinking, reasoning, or internal monologue. "
            "/no_think"
        ),
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="parakeet-demo")
    await kit.attach_channel("parakeet-demo", "voice")
    await kit.attach_channel("parakeet-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session, ctx):
        logger.info("Speech ended — transcribing with Parakeet TDT...")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text, ctx):
        logger.info("You said: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Warmup ---------------------------------------------------------------
    logger.info("Loading Parakeet TDT model (~640 MB)...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded — ready!")

    # --- Start voice session --------------------------------------------------
    session = await backend.connect("parakeet-demo", "local-user", "voice")
    binding = ChannelBinding(
        room_id="parakeet-demo",
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "parakeet-demo", binding)

    logger.info("")
    logger.info("Speak into your microphone. Press Ctrl+C to stop.")
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
