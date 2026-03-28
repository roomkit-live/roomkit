"""RoomKit -- Voice assistant with NeuTTS French voice cloning.

Uses NeuTTS for LLM-based text-to-speech with zero-shot voice cloning from a
short reference audio clip (3-15 seconds). NeuTTS uses a Qwen2.5 backbone with
NeuCodec for high-quality 24kHz audio, with native streaming via GGUF models.

Audio flow:
    Mic -> [Pipeline] -> VAD -> STT -> LLM -> NeuTTS (voice clone) -> Speaker

Requirements:
    pip install roomkit[local-audio,openai,sherpa-onnx] neutts

    System dependencies:
    - espeak-ng (required by neutts for phonemization)
      Ubuntu: sudo apt install espeak-ng
      macOS:  brew install espeak-ng

    A local LLM server:
      Ollama: ollama pull qwen3:8b && ollama serve

    STT models (sherpa-onnx):
      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
      tar xf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

    VAD model:
      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

    Reference audio for voice cloning:
      Record 3-15 seconds of clean speech and provide the transcript.
      Save as a WAV file (mono, 16-44 kHz). Use a same-language reference
      for best results (French for the default model).

Run:
    REF_AUDIO=reference_fr.wav \\
    REF_TEXT="Bonjour, je suis un assistant vocal." \\
    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    VAD_MODEL=ten-vad.onnx \\
    STT_ENCODER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \\
    STT_DECODER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \\
    STT_JOINER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \\
    STT_TOKENS=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \\
    uv run python examples/voice_neutts.py

Environment variables:
    --- NeuTTS ---
    REF_AUDIO           (required) Path to reference audio for voice cloning (3-15s)
    REF_TEXT            (required) Transcript of the reference audio
    TTS_BACKBONE        HF repo or local path (default: neuphonic/neutts-nano-french-q8-gguf)
    TTS_CODEC           HF repo or local path (default: neuphonic/neucodec)
    TTS_DEVICE          Inference device: cpu | cuda (default: cpu)

    --- LLM ---
    LLM_MODEL           (required) Model name (e.g. qwen3:8b)
    LLM_BASE_URL        Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY         API key if needed (default: none)
    LLM_MAX_TOKENS      Max response tokens (default: 256)
    SYSTEM_PROMPT       Custom system prompt

    --- STT (sherpa-onnx) ---
    STT_ENCODER         (required) Path to encoder .onnx
    STT_DECODER         (required) Path to decoder .onnx
    STT_JOINER          Path to joiner .onnx (transducer mode)
    STT_TOKENS          (required) Path to tokens.txt

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           (required) Path to VAD .onnx model
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, run_until_stopped, setup_console, setup_logging

from roomkit import ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.neutts import NeuTTSConfig, NeuTTSProvider, NeuTTSVoiceConfig

logger = setup_logging("voice_neutts")
# Suppress espeak-ng language-switch warnings (harmless, fires on mixed-language text)
logging.getLogger("phonemizer").setLevel(logging.ERROR)


async def main() -> None:
    env = require_env(
        "REF_AUDIO",
        "REF_TEXT",
        "LLM_MODEL",
        "VAD_MODEL",
        "STT_ENCODER",
        "STT_DECODER",
        "STT_TOKENS",
    )

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    sample_rate = 16000
    # NeuTTS outputs at 24kHz natively
    tts_sample_rate = 24000

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_sample_rate,
        channels=1,
        block_duration_ms=20,
    )

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad_model = env["VAD_MODEL"]
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

    # --- STT (sherpa-onnx) ----------------------------------------------------
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig(
            mode="transducer",
            encoder=env["STT_ENCODER"],
            decoder=env["STT_DECODER"],
            joiner=os.environ.get("STT_JOINER", ""),
            tokens=env["STT_TOKENS"],
            sample_rate=sample_rate,
            provider="cpu",
        )
    )
    logger.info("STT: sherpa-onnx (encoder=%s)", os.environ["STT_ENCODER"])

    # --- TTS (NeuTTS with voice cloning) --------------------------------------
    ref_audio = env["REF_AUDIO"]
    ref_text = env["REF_TEXT"]

    # Auto-detect CUDA if available, override with TTS_DEVICE env var
    tts_device = os.environ.get("TTS_DEVICE")
    if tts_device is None:
        try:
            import torch

            tts_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            tts_device = "cpu"

    tts_config = NeuTTSConfig(
        backbone_repo=os.environ.get("TTS_BACKBONE", "neuphonic/neutts-nano-french-q8-gguf"),
        codec_repo=os.environ.get("TTS_CODEC", "neuphonic/neucodec"),
        device=tts_device,
        voices={
            "default": NeuTTSVoiceConfig(ref_audio=ref_audio, ref_text=ref_text),
        },
        # GPU is fast enough for real-time streaming; skip pre-buffer
        streaming_pre_buffer=0 if tts_device == "cuda" else 2,
    )
    tts = NeuTTSProvider(tts_config)
    logger.info("TTS: NeuTTS (backbone=%s, ref_audio=%s)", tts_config.backbone_repo, ref_audio)

    # --- LLM (local via OpenAI-compatible API) --------------------------------
    llm_model = env["LLM_MODEL"]
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
        "Tu es un assistant vocal sympathique. Réponds de manière concise "
        "en une ou deux phrases maximum. Réponds directement sans réflexion "
        "ni monologue interne. /no_think",
    )

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
    await kit.create_room(room_id="neutts-voice")
    await kit.attach_channel("neutts-voice", "ai", category=ChannelCategory.INTELLIGENCE)

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
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Warmup: pre-load models ----------------------------------------------
    logger.info("Loading models (NeuTTS downloads on first run)...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded — ready!")

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("neutts-voice", "voice")

    logger.info("")
    logger.info("Voice cloning active — speaking as '%s'", ref_audio)
    logger.info("Speak into your microphone. Press Ctrl+C to stop.")
    logger.info("")

    # --- Keep running until Ctrl+C --------------------------------------------
    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())
