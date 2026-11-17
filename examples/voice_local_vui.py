"""RoomKit -- Local voice assistant with Vui Nano: it hears the conversation.

Everything runs on this machine, microphone included:
  - sherpa-onnx neural VAD and speech-to-text (CPU)
  - a local LLM that RoomKit runs itself through llama.cpp (or Ollama)
  - Vui Nano text-to-speech (CUDA GPU), which generates each reply inside the
    dialogue: the TTS conversation context (RFC §12.2.2) hands it every turn,
    the audio of what you said included, and a barge-in cuts its reply back to
    what you actually heard.

    Mic → [AEC] → VAD → sherpa-onnx STT → local LLM → Vui → Speaker

Vui speaks English only: the STT model, the LLM prompt and the voice are
English here.

Requirements:
    Python 3.12 (a vui-tts requirement), a CUDA GPU: Vui takes ~3.5 GB of VRAM
    and the default LLM ~4 GB, which fit together on a 12 GB card.

    Nothing to run beside it: the LLM, Qwen3-4B-Instruct (Q4_K_M), is downloaded
    on first run with the llama.cpp build for this machine, and RoomKit starts
    and stops llama-server itself. Pick an instruct model, not a thinking one:
    Vui would read the reasoning aloud.

    Headphones are recommended: echo cancellation is never perfect on
    speakers, and the assistant hearing itself reads as a barge-in.

Models (download once, while online): TEN-VAD, and Kroko, an English streaming
Zipformer transducer that holds up on microphone speech (model license:
huggingface.co/Banafo/Kroko-ASR):
    python -m roomkit.local_models download ten-vad zipformer-en-kroko-2025-08-06
    # Turn detection (optional, used once downloaded): Smart Turn v3 hears
    # whether a sentence is over; without it, a turn ends at every 0.6 s pause
    python -m roomkit.local_models download smart-turn-v3.2-cpu

    Vui's weights and voice presets download from Hugging Face on first run.

Run (from the repository root; .venv-vui keeps the Python 3.12 environment
apart from the project's .venv):
    UV_PROJECT_ENVIRONMENT=.venv-vui uv run --python 3.12 \\
        --extra local-audio --extra webrtc-aec --extra llamacpp \\
        --extra sherpa-onnx --extra smart-turn --extra vui \\
        python examples/voice_local_vui.py

Environment variables:
    --- LLM ---
    LLM_BACKEND         llamacpp | ollama (default: llamacpp)
    LLM_MODEL           llamacpp: GGUF "repo:quant" or .gguf path
                        (default: unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M);
                        ollama: model name (default: qwen3:4b-instruct; add
                        --extra ollama to the run command)
    OLLAMA_HOST         Ollama server (default: http://localhost:11434)
    LLM_MAX_TOKENS      Max response tokens (default: 200)
    SYSTEM_PROMPT       Custom system prompt

    --- STT and VAD (sherpa-onnx, CPU) ---
    ROOMKIT_MODELS_DIR  Where the downloaded models live (default: ~/.cache/roomkit/models)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.5)

    --- Vui ---
    VUI_VOICE           Preset voice: maeve | abraham | rhian | harry (default: maeve)
    VUI_INCLUDE_AUDIO   Let Vui hear your voice, not only your words: 1 | 0 (default: 1)
    VUI_DISABLE_CUDNN   1 if the codec fails with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
                        (a system cuDNN shadowing PyTorch's)

    --- Audio ---
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)
    MUTE_MIC            Mute the mic while Vui speaks: 1 | 0 (default: 0 with AEC).
                        Muting disables barge-in.

    --- Barge-in ---
    INTERRUPTION        semantic | confirmed | immediate | disabled (default: semantic:
                        an acknowledgement like "okay" does not stop the voice)
    INTERRUPTION_WAIT_MS
                        How long semantic waits for the first words before
                        judging on duration (default: 2000)

    --- Debugging ---
    VOICE_DEBUG         1 to log turn-taking decisions (speech start/end,
                        suppressed segments, barge-in evaluation, the Vui cache,
                        AI turns)
    DEBUG_AUDIO         1 to record every pipeline stage as WAV in ./debug_audio/

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    build_debug_taps,
    build_interruption,
    enable_voice_debug,
    require_local_models,
    run_until_stopped,
    setup_logging,
)

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    TTSContextConfig,
    VoiceChannel,
)
from roomkit.local_models import is_downloaded
from roomkit.providers.ai import AIProvider
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import (
    AECProvider,
    AudioPipelineConfig,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)
from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector
from roomkit.voice.stt import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts import (
    VUI_TAGS,
    StripBrackets,
    StripEmoji,
    TTSFilterChain,
    TTSStreamFilter,
    VuiTTSConfig,
    VuiTTSProvider,
    VuiVoice,
)

logger = setup_logging("voice_local_vui")

MIC_RATE = 16000
BLOCK_MS = 20

VAD_MODEL = "ten-vad"
STT_MODEL = "zipformer-en-kroko-2025-08-06"
SMART_TURN_MODEL = "smart-turn-v3.2-cpu"

SYSTEM_PROMPT = (
    "You are a friendly voice assistant having a spoken conversation in English. "
    "Keep every reply short and natural, one or two sentences, the way people talk. "
    "You remember everything said earlier in this conversation and build on it. "
    "Your voice can make these sounds, written exactly so, where a person would: "
    + ", ".join(f"[{tag}]" for tag in VUI_TAGS)
    + ". Never write any other word in square brackets: no [nod], [smiles] or "
    "stage directions. Never use emojis, lists or markdown."
)


class StripEmphasis(TTSStreamFilter):
    """Remove the ``*`` and backticks a model writes for emphasis or code."""

    def reset(self) -> None:
        pass

    def feed(self, chunk: str) -> str:
        return chunk.replace("*", "").replace("`", "")

    def flush(self) -> str:
        return ""


# What the LLM writes despite the prompt: emoji, emphasis marks, and bracketed
# words Vui cannot render. Vui's own tags pass through.
VUI_TEXT_FILTER = TTSFilterChain(StripEmoji(), StripEmphasis(), StripBrackets(keep=VUI_TAGS))


def build_turn_detector() -> SmartTurnDetector | None:
    """Smart Turn v3 once its model is downloaded: it hears whether a sentence is over.

    The VAD alone ends a turn at every 0.6 s pause, so "I'm just making [pause]
    a test with you" became two turns and two answers. A turn Smart Turn judges
    unfinished waits for more speech, and is still answered after 1.5 s.
    """
    if not is_downloaded(SMART_TURN_MODEL):
        logger.info("Turn detection: VAD pauses only (%s not downloaded)", SMART_TURN_MODEL)
        return None
    try:
        detector = SmartTurnDetector(SmartTurnConfig.from_local_model(SMART_TURN_MODEL))
    except ImportError as exc:
        logger.warning("Smart Turn needs its extra (--extra smart-turn): %s", exc)
        return None
    logger.info("Turn detection: Smart Turn v3 (%s)", SMART_TURN_MODEL)
    return detector


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


def build_aec() -> AECProvider | None:
    mode = os.environ.get("AEC", "webrtc").lower()
    if mode in ("1", "webrtc"):
        from roomkit.voice.pipeline.aec import WebRTCAECProvider

        return WebRTCAECProvider(sample_rate=MIC_RATE)
    if mode == "speex":
        from roomkit.voice.pipeline import SpeexAECProvider

        frame = MIC_RATE * BLOCK_MS // 1000
        return SpeexAECProvider(frame_size=frame, filter_length=frame * 10, sample_rate=MIC_RATE)
    return None


async def main() -> None:
    require_local_models(VAD_MODEL, STT_MODEL)
    if os.environ.get("VUI_DISABLE_CUDNN") == "1":
        import torch

        torch.backends.cudnn.enabled = False

    kit = RoomKit()

    # --- Microphone and speaker ----------------------------------------------
    # With echo cancellation the mic stays open while Vui speaks, so you can
    # interrupt it; without, it is muted during playback to avoid feedback.
    aec = build_aec()
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env == "1" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=MIC_RATE,
        output_sample_rate=24000,  # Vui speaks 24 kHz
        channels=1,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )
    logger.info("Audio: AEC=%s, mic muted during playback=%s", type(aec).__name__, mute_mic)

    # --- VAD and STT (sherpa-onnx, CPU: the GPU is Vui's and the LLM's) --------
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
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig.from_local_model(
            STT_MODEL,
            sample_rate=MIC_RATE,
            provider="cpu",
            # Kroko drops the last word with less (measured on Kroko FR).
            tail_padding_s=1.5,
        )
    )
    logger.info("STT: sherpa-onnx streaming transducer, %s (CPU)", STT_MODEL)

    # --- Vui: the reply is generated inside the dialogue ------------------------
    voice_name = os.environ.get("VUI_VOICE", "maeve")
    tts = VuiTTSProvider(VuiTTSConfig(voices={voice_name: VuiVoice(voice_name)}))
    include_audio = os.environ.get("VUI_INCLUDE_AUDIO", "1") == "1"

    # --- LLM: thinking off, a spoken reply cannot wait ----------------------------
    ai_provider = build_llm()
    logger.info(
        "LLM: %s (%s), Vui voice: %s", ai_provider.model_name, ai_provider.name, voice_name
    )

    # --- Channels and room -------------------------------------------------------
    turn_detector = build_turn_detector()
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        # No aec= here: the backend feeds the echo reference itself and
        # reports NATIVE_AEC, so a pipeline copy would never run.
        pipeline=AudioPipelineConfig(
            vad=vad,
            turn_detector=turn_detector,
            turn_incomplete_wait_ms=1500,
            debug_taps=build_debug_taps(),
        ),
        tts_filter=VUI_TEXT_FILTER,
        # Vui keeps talking through an "okay" and stops for anything else.
        interruption=build_interruption(),
        # Vui hears the dialogue: your words, and your voice when include_audio.
        tts_context=TTSContextConfig(include_audio=include_audio),
    )
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=ai_provider,
            system_prompt=os.environ.get("SYSTEM_PROMPT", SYSTEM_PROMPT),
        )
    )
    await kit.create_room(room_id="local-vui")
    await kit.attach_channel("local-vui", "ai", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Vui: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
    async def on_barge_in(event, ctx):
        logger.info("Barge-in: Vui stops and keeps only what you heard")

    if os.environ.get("VOICE_DEBUG") == "1":
        enable_voice_debug(kit)

    # --- Load everything before the first word -----------------------------------
    logger.info("Loading the LLM, Vui, STT and VAD models...")
    warmups = [stt.warmup(), tts.warmup(), start_llm(ai_provider)]
    if turn_detector is not None:
        warmups.append(asyncio.to_thread(turn_detector.warmup))
    await asyncio.gather(*warmups)
    await kit.attach_channel("local-vui", "voice")  # opens the mic
    logger.info("Ready: speak English into the microphone. Ctrl+C to stop.")

    try:
        await run_until_stopped(kit)
    finally:
        await ai_provider.close()  # the provider is ours: the kit's close leaves it open


if __name__ == "__main__":
    asyncio.run(main())
