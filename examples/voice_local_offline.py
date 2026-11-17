"""RoomKit -- A voice agent that runs with no network: every model on this machine.

Speech-to-text and text-to-speech run in sherpa-onnx (a streaming zipformer,
Kokoro), the LLM in llama.cpp (SmolLM2-135M), noise suppression in RNNoise,
voice activity detection in TEN-VAD and end-of-turn detection in Smart Turn
v3.2. The sherpa-onnx and Smart Turn models are named, not located: RoomKit
knows their files, pinned to their SHA-256, and downloads them once.

    Mic → RNNoise → TEN-VAD → zipformer STT → Smart Turn → SmolLM2 → Kokoro → Speaker

Requirements:
    pip install roomkit[sherpa-onnx,smart-turn,llamacpp,rnnoise,local-audio]

Once, while online (about 220 MB):
    python -m roomkit.local_models download \\
        zipformer-en-2023-06-26 kokoro-int8-multi-lang-v1_0 ten-vad smart-turn-v3.2-cpu

The first start also downloads llama.cpp and the GGUF model; from then on the
loop needs no network. For a model you already hold, set LLM_MODEL to the path
of its .gguf file.

Run with:
    uv run python examples/voice_local_offline.py

Environment variables:
    LLM_MODEL     GGUF model: a Hugging Face "repo:quant" or a local .gguf path
                  (default: bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0, the
                  smallest model that holds a conversation; try
                  unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M for better answers)
    VOICE         Kokoro voice (default: af_heart; ff_siwis with VOICE_LANG=fr)
    VOICE_LANG    espeak-ng language for text that is not English or Chinese
    ROOMKIT_MODELS_DIR  Where the downloaded models live
                  (default: ~/.cache/roomkit/models)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_local_models, run_until_stopped, setup_logging

from roomkit import Agent, ChannelCategory, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    RNNoiseDenoiserProvider,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)
from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector
from roomkit.voice.stt import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logger = setup_logging("voice_local_offline")

INSTRUCTIONS = (
    "You are a friendly voice assistant that runs fully offline on the user's computer. "
    "Answer in one or two short sentences. Use plain spoken English: "
    "no emojis, markdown, lists or special characters."
)


def build_speech() -> tuple[SherpaOnnxSTTProvider, SherpaOnnxTTSProvider]:
    """Speech-to-text and text-to-speech, their models found by name."""
    stt = SherpaOnnxSTTProvider(SherpaOnnxSTTConfig.from_local_model("zipformer-en-2023-06-26"))
    tts = SherpaOnnxTTSProvider(
        SherpaOnnxTTSConfig.from_local_model(
            "kokoro-int8-multi-lang-v1_0",
            voice=os.environ.get("VOICE", "af_heart"),
            lang=os.environ.get("VOICE_LANG", ""),
        )
    )
    return stt, tts


def build_pipeline() -> AudioPipelineConfig:
    return AudioPipelineConfig(
        denoiser=RNNoiseDenoiserProvider(),
        vad=SherpaOnnxVADProvider(SherpaOnnxVADConfig.from_local_model("ten-vad")),
        turn_detector=SmartTurnDetector(SmartTurnConfig.from_local_model("smart-turn-v3.2-cpu")),
    )


async def main() -> None:
    require_local_models(
        "zipformer-en-2023-06-26", "kokoro-int8-multi-lang-v1_0", "ten-vad", "smart-turn-v3.2-cpu"
    )
    stt, tts = build_speech()
    pipeline = build_pipeline()
    # CPU on purpose: the point is a loop that needs no GPU anywhere.
    llm = LlamaCppAIProvider(
        LlamaCppConfig(
            model=os.environ.get("LLM_MODEL", "bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0"),
            max_tokens=80,
            gpu_layers=0,
        )
    )

    kit = RoomKit()
    # Without an echo canceller the backend mutes the mic while the agent speaks.
    kit.register_channel(
        VoiceChannel("voice", stt=stt, tts=tts, backend=LocalAudioBackend(), pipeline=pipeline)
    )
    kit.register_channel(
        Agent(
            "agent",
            provider=llm,
            system_prompt=INSTRUCTIONS,
            greeting="Hi, I run entirely on this computer. What can I do for you?",
        )
    )
    await kit.create_room(room_id="offline")
    await kit.attach_channel("offline", "agent", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Agent: %s", text)
        return HookResult.allow()

    logger.info("Loading the models...")
    await asyncio.gather(stt.warmup(), tts.warmup(), llm.start())
    await kit.attach_channel("offline", "voice")  # opens the mic; the agent greets
    logger.info("Ready: speak into the microphone. Ctrl+C to stop.")
    try:
        await run_until_stopped(kit)
    finally:
        await llm.close()  # stops llama-server: the provider is ours, not the kit's


if __name__ == "__main__":
    asyncio.run(main())
