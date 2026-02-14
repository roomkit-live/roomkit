"""RoomKit -- Voice assistant with Qwen3-ASR speech recognition.

Uses Qwen3-ASR for high-quality local speech recognition with automatic
language detection. Combine with any TTS + LLM backend for a complete
voice assistant with state-of-the-art local ASR.

Audio flow:
    Mic -> [Pipeline] -> VAD -> Qwen3-ASR (STT) -> LLM -> TTS -> Speaker

Requirements:
    pip install roomkit[local-audio,openai,sherpa-onnx,qwen-asr]

    System dependencies:
    - CUDA GPU with 3-5GB VRAM (for Qwen3-ASR-0.6B model)
    - For the 4B model, 8GB+ VRAM is recommended

    A local LLM server:
      Ollama: ollama pull qwen3:8b && ollama serve

    VAD model:
      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

Run:
    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    VAD_MODEL=ten-vad.onnx \\
    uv run python examples/voice_qwen3_asr.py

Environment variables:
    --- Qwen3-ASR ---
    ASR_MODEL_ID        HuggingFace model ID (default: Qwen/Qwen3-ASR-0.6B)
    ASR_BACKEND         Inference backend: transformers | vllm (default: transformers)
    ASR_DEVICE_MAP      Torch device mapping (default: auto)
    ASR_DTYPE           Model dtype: bfloat16, float16, float32 (default: bfloat16)
    ASR_LANGUAGE        Language code, e.g. en, zh (default: auto-detect)

    --- LLM ---
    LLM_MODEL           (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL        Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY         API key if needed (default: none)
    LLM_MAX_TOKENS      Max response tokens (default: 256)
    SYSTEM_PROMPT       Custom system prompt

    --- TTS (sherpa-onnx) ---
    TTS_MODEL           Path to TTS .onnx model
    TTS_TOKENS          Path to TTS tokens.txt
    TTS_DATA_DIR        Path to TTS data directory (espeak-ng-data)

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           (required) Path to VAD .onnx model
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)

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
from roomkit.voice.stt.qwen3 import Qwen3ASRConfig, Qwen3ASRProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_qwen3_asr")


def check_env() -> None:
    """Check required environment variables."""
    required = {
        "LLM_MODEL": "Model name (e.g. qwen3:8b for Ollama)",
        "VAD_MODEL": "Path to VAD .onnx model (e.g. ten-vad.onnx)",
    }
    missing = [
        f"  {key:20s} — {desc}" for key, desc in required.items() if not os.environ.get(key)
    ]
    if missing:
        print("Missing required environment variables:\n")
        print("\n".join(missing))
        print("\nSee docstring at the top of this file for usage.")
        sys.exit(1)


async def main() -> None:
    check_env()

    kit = RoomKit()

    sample_rate = 16000

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        channels=1,
        block_duration_ms=20,
    )

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad_model = os.environ["VAD_MODEL"]
    vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=vad_model,
            threshold=vad_threshold,
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            speech_pad_ms=300,
            sample_rate=sample_rate,
            provider="cpu",
        )
    )
    logger.info("VAD: sherpa-onnx (threshold=%.2f, model=%s)", vad_threshold, vad_model)

    pipeline_config = AudioPipelineConfig(vad=vad)

    # --- STT (Qwen3-ASR) -----------------------------------------------------
    asr_language = os.environ.get("ASR_LANGUAGE") or None
    stt_config = Qwen3ASRConfig(
        model_id=os.environ.get("ASR_MODEL_ID", "Qwen/Qwen3-ASR-0.6B"),
        backend=os.environ.get("ASR_BACKEND", "transformers"),
        device_map=os.environ.get("ASR_DEVICE_MAP", "auto"),
        dtype=os.environ.get("ASR_DTYPE", "bfloat16"),
        language=asr_language,
    )
    stt = Qwen3ASRProvider(stt_config)
    logger.info(
        "STT: Qwen3-ASR (model=%s, backend=%s, language=%s)",
        stt_config.model_id,
        stt_config.backend,
        stt_config.language or "auto-detect",
    )

    # --- LLM (local via OpenAI-compatible API) --------------------------------
    llm_model = os.environ["LLM_MODEL"]
    llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    ai_provider = create_vllm_provider(
        VLLMConfig(
            model=llm_model,
            base_url=llm_base_url,
            api_key=os.environ.get("LLM_API_KEY", "none"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
        )
    )
    logger.info("LLM: %s (base_url=%s)", llm_model, llm_base_url)

    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly voice assistant. Keep your responses "
        "short and conversational — one or two sentences at most. "
        "Answer directly without thinking, reasoning, or internal monologue. "
        "/no_think",
    )

    # --- Channels -------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="qwen3-asr")
    await kit.attach_channel("qwen3-asr", "voice")
    await kit.attach_channel("qwen3-asr", "ai", category=ChannelCategory.INTELLIGENCE)

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
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Warmup: pre-load model -----------------------------------------------
    logger.info("Loading Qwen3-ASR model (may take a moment on first run)...")
    await stt.warmup()
    logger.info("Model loaded — ready!")

    # --- Start voice session --------------------------------------------------
    session = await backend.connect("qwen3-asr", "local-user", "voice")
    binding = ChannelBinding(
        room_id="qwen3-asr",
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "qwen3-asr", binding)

    logger.info("")
    logger.info(
        "Qwen3-ASR active — model=%s, language=%s",
        stt_config.model_id,
        stt_config.language or "auto-detect",
    )
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
    await stt.close()
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
