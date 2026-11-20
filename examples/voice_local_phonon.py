"""RoomKit -- Local French voice assistant with Gradium's Phonon TTS, on the CPU.

Everything runs on this machine, microphone included:
  - sherpa-onnx neural VAD and French speech-to-text (CPU)
  - a local LLM that RoomKit runs itself through llama.cpp (or Ollama)
  - Gradium's Phonon, a 40M-parameter French TTS, on the CPU through its Rust runtime

    Mic → [AEC] → VAD → sherpa-onnx STT (fr) → local LLM → [StripEmoji] → Phonon (fr) → Speaker

Phonon needs neither torch nor a GPU. Measured on a desktop CPU with two
threads, ``phonon-40m-fr`` runs about ten times faster than real time and
starts speaking ~50 ms after it gets a sentence.

Requirements:
    Nothing to run beside it: the LLM, Qwen3-4B-Instruct (Q4_K_M), is downloaded
    on first run with the llama.cpp build for this machine, and RoomKit starts
    and stops llama-server itself (~4 GB of VRAM on a GPU, or the CPU, slower).
    Headphones are recommended: echo cancellation is never perfect on
    speakers, and the assistant hearing itself reads as a barge-in.

    The Phonon 40m models are gated: accept their terms on
    https://huggingface.co/gradium/phonon-40m-fr (its licence is stated
    there), then run ``hf auth login`` or set HF_TOKEN. The weights (75 MB)
    and the voice you pick download on first run.

Models (download once, while online): TEN-VAD, and Kroko, a French streaming
Zipformer transducer (model license: huggingface.co/Banafo/Kroko-ASR):
    python -m roomkit.local_models download ten-vad zipformer-fr-kroko-2025-08-06
    (with TURN_DETECTOR=smart-turn, add smart-turn-v3.2-cpu)

Run (from the repository root):
    uv run --extra local-audio --extra webrtc-aec --extra llamacpp \\
        --extra sherpa-onnx --extra phonon \\
        python examples/voice_local_phonon.py

    With the console dashboard (add --extra console):
    CONSOLE=1 uv run ... --extra console python examples/voice_local_phonon.py

Environment variables:
    --- Phonon ---
    PHONON_MODEL        Hugging Face repo or local checkpoint
                        (default: hf://gradium/phonon-40m-fr; also
                        hf://gradium/phonon-40m-multilingual)
    PHONON_LANGUAGE     How numbers, dates and symbols are spelled out:
                        en | fr | de | es | pt (default: fr)
    PHONON_VOICE        A voice from huggingface.co/gradium/phonon-voices, by
                        name (Noémie, Lucien...) or id (default: the model's own)
    PHONON_SPEED        Speaking rate: negative is faster, positive slower,
                        e.g. -0.2 (default: the model's)
    PHONON_THREADS      CPU threads of the runtime (default: 2)

    --- LLM ---
    LLM_BACKEND         llamacpp | ollama (default: llamacpp)
    LLM_MODEL           llamacpp: GGUF "repo:quant" or .gguf path
                        (default: unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M);
                        ollama: model name (default: qwen3:4b-instruct; add
                        --extra ollama to the run command)
    OLLAMA_HOST         Ollama server (default: http://localhost:11434)
    LLM_MAX_TOKENS      Max response tokens (default: 200)
    SYSTEM_PROMPT       Custom system prompt

    --- Barge-in and turns ---
    INTERRUPTION        semantic | confirmed | immediate | disabled (default: semantic:
                        an acknowledgement like "d'accord" does not stop the voice)
    TURN_DETECTOR       smart-turn | 0 (default: 0; add --extra smart-turn).
                        Runs the smart-turn-v3.2-cpu local model; TURN_MODEL
                        sets an .onnx file instead.
    VOICE_DEBUG         1 to log turn-taking decisions

    --- STT and VAD (sherpa-onnx, CPU) ---
    ROOMKIT_MODELS_DIR  Where the downloaded models live (default: ~/.cache/roomkit/models)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.5)

    --- Audio ---
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)

    --- Other ---
    CONSOLE             1 shows the RoomKit console dashboard (default: 0)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import ptts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    build_aec,
    build_interruption,
    build_turn_detector,
    enable_voice_debug,
    require_local_models,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.providers.ai import AIProvider
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig, SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts import PhononTTSConfig, PhononTTSProvider, StripEmoji

logger = setup_logging("voice_local_phonon")
# Request lines: httpx2 under the model client (openai 3.x), httpx under the server check.
for _noisy in ("httpx", "httpx2"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

MIC_RATE = 16000
BLOCK_MS = 20

VAD_MODEL = "ten-vad"
STT_MODEL = "zipformer-fr-kroko-2025-08-06"

SYSTEM_PROMPT = (
    "You are a warm voice assistant having a spoken conversation. "
    "Always answer in French, in one or two short, natural sentences, the way people talk. "
    "Never use lists, markdown or emojis."
)


def build_llm() -> AIProvider:
    """The local LLM: llama.cpp run by RoomKit, or an Ollama server."""
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "200"))
    if os.environ.get("LLM_BACKEND", "llamacpp") == "ollama":
        return OllamaAIProvider(
            OllamaConfig(
                host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                model=os.environ.get("LLM_MODEL", "qwen3:4b-instruct"),
                max_tokens=max_tokens,
                think=False,
            )
        )
    return LlamaCppAIProvider(
        LlamaCppConfig(
            model=os.environ.get("LLM_MODEL", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"),
            max_tokens=max_tokens,
            enable_thinking=False,
        )
    )


async def start_llm(provider: AIProvider) -> None:
    """Load a llama.cpp model before the first word; an Ollama server is already up."""
    if isinstance(provider, LlamaCppAIProvider):
        await provider.start()


def build_tts() -> PhononTTSProvider:
    """Phonon in French; the runtime's thread pool is sized before the model loads."""
    ptts.set_num_threads(int(os.environ.get("PHONON_THREADS", "2")))
    speed = os.environ.get("PHONON_SPEED")
    config = PhononTTSConfig(
        model=os.environ.get("PHONON_MODEL", "hf://gradium/phonon-40m-fr"),
        language=os.environ.get("PHONON_LANGUAGE", "fr"),
        voice=os.environ.get("PHONON_VOICE") or None,
        duration_delta=float(speed) if speed else None,
    )
    logger.info("Phonon: %s, voice %s", config.model, config.voice or "of the model")
    return PhononTTSProvider(config)


async def main() -> None:
    require_local_models(VAD_MODEL, STT_MODEL)
    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # --- Microphone and speaker ----------------------------------------------
    # With echo cancellation the mic stays open while the assistant speaks, so
    # you can interrupt it; without, it is muted during playback.
    aec = build_aec(MIC_RATE, BLOCK_MS)
    backend = LocalAudioBackend(
        input_sample_rate=MIC_RATE,
        output_sample_rate=24000,  # Phonon speaks 24 kHz
        channels=1,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=aec is None,
        aec=aec,
    )

    # --- VAD and STT (sherpa-onnx, CPU) ----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig.from_local_model(
            VAD_MODEL,
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            sample_rate=MIC_RATE,
            provider="cpu",
        )
    )
    # The local model carries Kroko French's 1.5 s tail padding: below it the
    # last word is dropped ("midi" -> "mid").
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig.from_local_model(STT_MODEL, sample_rate=MIC_RATE, provider="cpu")
    )

    tts = build_tts()
    ai_provider = build_llm()
    logger.info("LLM: %s (%s)", ai_provider.model_name, ai_provider.name)

    # --- Channels and room -------------------------------------------------------
    turn_detector = build_turn_detector()
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        # No aec= here: the backend feeds the echo reference itself and
        # reports NATIVE_AEC, so a pipeline copy would never run.
        pipeline=AudioPipelineConfig(vad=vad, turn_detector=turn_detector),
        # The LLM adds emoji despite the prompt; spoken, they sound wrong.
        tts_filter=StripEmoji(),
        interruption=build_interruption(),
    )
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=ai_provider,
            system_prompt=os.environ.get("SYSTEM_PROMPT", SYSTEM_PROMPT),
        )
    )
    await kit.create_room(room_id="local-phonon")
    await kit.attach_channel("local-phonon", "ai", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
    async def on_barge_in(event, ctx):
        logger.info("Barge-in: the assistant stops speaking")

    if os.environ.get("VOICE_DEBUG") == "1":
        enable_voice_debug(kit)

    # --- Load everything before the first word -----------------------------------
    logger.info("Loading the LLM, Phonon, STT and VAD models...")
    try:
        await asyncio.gather(stt.warmup(), tts.warmup(), start_llm(ai_provider))
        await kit.attach_channel("local-phonon", "voice")  # opens the mic
        logger.info("Ready: speak French into the microphone. Ctrl+C to stop.")
        await run_until_stopped(kit, cleanup=console_cleanup)
    finally:
        await ai_provider.close()  # stops llama-server


if __name__ == "__main__":
    asyncio.run(main())
