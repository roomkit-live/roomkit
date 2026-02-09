"""RoomKit -- Audio-native turn detection with smart-turn.

Demonstrates SmartTurnDetector for natural conversational turn-taking.
Instead of relying on silence duration to decide when the user is done
speaking, the smart-turn model analyzes prosody and intonation from raw
audio to distinguish mid-utterance pauses from actual turn completions.

This example uses a fully local stack:
  - sherpa-onnx for STT (streaming Zipformer) + TTS (VITS/Piper)
  - sherpa-onnx neural VAD (TEN-VAD)
  - smart-turn ONNX model for turn detection
  - Local LLM via any OpenAI-compatible server (Ollama, vLLM, etc.)

How it works:
  1. VAD detects speech start/end and collects the audio segment
  2. STT transcribes the speech to text
  3. SmartTurnDetector receives the raw audio (not text) and predicts
     whether the user's turn is complete based on prosodic cues
  4. If complete → route to AI. If incomplete → wait for more speech.
     Accumulated audio and text are combined when the turn completes.

Requirements:
    pip install roomkit[local-audio,openai,sherpa-onnx,smart-turn]

Download models:
    # smart-turn (required for this example)
    wget https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx

    # VAD — TEN-VAD
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

    # STT — Zipformer transducer (streaming)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
    tar xf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

    # TTS — VITS (Piper voices)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xf vits-piper-en_US-amy-low.tar.bz2

Run:
    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    SMART_TURN_MODEL=smart-turn-v3.2-cpu.onnx \\
    VAD_MODEL=ten-vad.onnx \\
    STT_ENCODER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \\
    STT_DECODER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \\
    STT_JOINER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \\
    STT_TOKENS=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \\
    TTS_MODEL=vits-piper-en_US-amy-low/en_US-amy-low.onnx \\
    TTS_TOKENS=vits-piper-en_US-amy-low/tokens.txt \\
    TTS_DATA_DIR=vits-piper-en_US-amy-low/espeak-ng-data \\
    uv run python examples/voice_smart_turn.py

Environment variables:
    --- Smart turn detection ---
    SMART_TURN_MODEL      (required) Path to smart-turn .onnx model
    SMART_TURN_THRESHOLD  Completion probability threshold 0-1 (default: 0.5)
                          Lower values make the detector more eager to complete turns.
                          Higher values wait longer for natural turn endings.

    --- LLM ---
    LLM_MODEL             (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL          Server endpoint (default: http://localhost:11434/v1)

    --- STT (sherpa-onnx) ---
    STT_ENCODER           (required) Path to encoder .onnx
    STT_DECODER           (required) Path to decoder .onnx
    STT_JOINER            Path to joiner .onnx (transducer mode)
    STT_TOKENS            (required) Path to tokens.txt

    --- TTS (sherpa-onnx) ---
    TTS_MODEL             (required) Path to VITS/Piper .onnx model
    TTS_TOKENS            (required) Path to tokens.txt
    TTS_DATA_DIR          Path to espeak-ng data dir (Piper models)

    --- VAD (sherpa-onnx) ---
    VAD_MODEL             (required) Path to VAD .onnx model
    VAD_THRESHOLD         Speech probability threshold 0-1 (default: 0.35)

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
from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_smart_turn")


def check_env() -> None:
    """Check required environment variables."""
    required = {
        "SMART_TURN_MODEL": "Path to smart-turn .onnx model",
        "LLM_MODEL": "Model name (e.g. qwen3:8b for Ollama)",
        "VAD_MODEL": "Path to VAD .onnx model (e.g. ten-vad.onnx)",
        "STT_ENCODER": "Path to STT encoder .onnx",
        "STT_DECODER": "Path to STT decoder .onnx",
        "STT_TOKENS": "Path to STT tokens.txt",
        "TTS_MODEL": "Path to TTS VITS/Piper .onnx model",
        "TTS_TOKENS": "Path to TTS tokens.txt",
    }
    missing = [
        f"  {key:22s} — {desc}" for key, desc in required.items() if not os.environ.get(key)
    ]
    if missing:
        print("Missing required environment variables:\n")
        print("\n".join(missing))
        print("\nSee docstring at the top of this file for download links and usage.")
        sys.exit(1)


async def main() -> None:
    check_env()

    kit = RoomKit()
    sample_rate = 16000
    tts_sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "22050"))

    # --- Backend: local mic + speakers ----------------------------------------
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_sample_rate,
        channels=1,
        block_duration_ms=20,
    )

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=os.environ["VAD_MODEL"],
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.35")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            speech_pad_ms=300,
            sample_rate=sample_rate,
            provider="cpu",
        )
    )

    # --- Smart turn detector --------------------------------------------------
    smart_turn_model = os.environ["SMART_TURN_MODEL"]
    smart_turn_threshold = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))
    turn_detector = SmartTurnDetector(
        SmartTurnConfig(
            model_path=smart_turn_model,
            threshold=smart_turn_threshold,
        )
    )
    logger.info(
        "Turn detection: smart-turn (model=%s, threshold=%.2f)",
        smart_turn_model,
        smart_turn_threshold,
    )

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        turn_detector=turn_detector,
    )

    # --- STT (sherpa-onnx) ----------------------------------------------------
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig(
            mode=os.environ.get("STT_MODE", "transducer"),
            encoder=os.environ["STT_ENCODER"],
            decoder=os.environ["STT_DECODER"],
            joiner=os.environ.get("STT_JOINER", ""),
            tokens=os.environ["STT_TOKENS"],
            sample_rate=sample_rate,
        )
    )

    # --- TTS (sherpa-onnx) ----------------------------------------------------
    tts = SherpaOnnxTTSProvider(
        SherpaOnnxTTSConfig(
            model=os.environ["TTS_MODEL"],
            tokens=os.environ["TTS_TOKENS"],
            data_dir=os.environ.get("TTS_DATA_DIR", ""),
            sample_rate=tts_sample_rate,
        )
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
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="smart-turn-demo")
    await kit.attach_channel("smart-turn-demo", "voice")
    await kit.attach_channel("smart-turn-demo", "ai", category=ChannelCategory.INTELLIGENCE)

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

    @kit.hook(HookTrigger.ON_TURN_COMPLETE, execution=HookExecution.ASYNC)
    async def on_turn_complete(event, ctx):
        logger.info(
            "Turn COMPLETE (confidence=%.2f): %s",
            event.confidence,
            event.text,
        )

    @kit.hook(HookTrigger.ON_TURN_INCOMPLETE, execution=HookExecution.ASYNC)
    async def on_turn_incomplete(event, ctx):
        logger.info(
            "Turn INCOMPLETE (confidence=%.2f) — waiting for more: %s",
            event.confidence,
            event.text,
        )

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Warmup ---------------------------------------------------------------
    logger.info("Loading models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded — ready!")

    # --- Start voice session --------------------------------------------------
    session = await backend.connect("smart-turn-demo", "local-user", "voice")
    binding = ChannelBinding(
        room_id="smart-turn-demo",
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "smart-turn-demo", binding)

    logger.info("")
    logger.info("Smart turn detection active!")
    logger.info("Try speaking with natural pauses — the detector will wait")
    logger.info("for you to finish your thought before responding.")
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
    turn_detector.close()
    await asyncio.sleep(0.1)
    await backend.disconnect(session)
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
