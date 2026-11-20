"""RoomKit -- Local French voice assistant with Pocket TTS, on CPU or GPU.

Everything runs on this machine, microphone included:
  - sherpa-onnx neural VAD and French speech-to-text (CPU)
  - a local LLM that RoomKit runs itself through llama.cpp (or Ollama)
  - optional tools from an MCP server started as a command (stdio)
  - Kyutai / Gradium's Pocket TTS, French model, on the CPU (default) or a CUDA GPU

    Mic → [AEC] → VAD → sherpa-onnx STT (fr) → local LLM (+ MCP tools) → [StripEmoji] → Pocket TTS (fr) → Speaker

Pocket TTS is a 100M-parameter model that streams faster than real time on
two CPU cores: no GPU is needed for the voice. Measured on a desktop CPU, the
``french`` model starts speaking ~80 ms after it gets a sentence.

Requirements:
    Nothing to run beside it: the LLM, Qwen3-4B-Instruct (Q4_K_M), is downloaded
    on first run with the llama.cpp build for this machine, and RoomKit starts
    and stops llama-server itself (~4 GB of VRAM on a GPU, or the CPU, slower).
    Headphones are recommended: echo cancellation is never perfect on
    speakers, and the assistant hearing itself reads as a barge-in.

Models (download once, while online):
    # VAD: TEN-VAD; STT: Kroko, a Zipformer transducer, French, streaming
    # (model license: huggingface.co/Banafo/Kroko-ASR)
    python -m roomkit.local_models download ten-vad zipformer-fr-kroko-2025-08-06
    # Turn detection (optional, multilingual): a pause no longer splits a sentence
    python -m roomkit.local_models download smart-turn-v3.2-cpu

    Pocket TTS weights and its pre-made voices download from Hugging Face on
    first run (licences per voice: huggingface.co/kyutai/tts-voices).

Run (from the repository root):
    uv run --extra local-audio --extra webrtc-aec --extra llamacpp \\
        --extra sherpa-onnx --extra pocket-tts --extra smart-turn \\
        python examples/voice_local_pocket_fr.py

    With tools from an MCP server started as a command (add --extra mcp):
    MCP_COMMAND="uvx mcp-server-time" uv run ... --extra mcp \\
        python examples/voice_local_pocket_fr.py

    On Linux this installs the CUDA build of PyTorch (~3 GB). For CPU only,
    install torch from https://download.pytorch.org/whl/cpu first (see the
    Pocket TTS guide).

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

    --- Barge-in ---
    INTERRUPTION        semantic | confirmed | immediate | disabled (default: semantic:
                        an acknowledgement like "okay" does not stop the voice)
    INTERRUPTION_WAIT_MS
                        How long semantic waits for the first words before
                        judging on duration (default: 2000)

    --- Debugging ---
    VOICE_DEBUG         1 to log turn-taking decisions (speech start/end,
                        suppressed segments, barge-in evaluation, AI turns)
    DEBUG_AUDIO_DIR     Directory to record the mic audio as WAV: as captured
                        (00_transport_raw), the echo reference the AEC received
                        (08_aec_reference), after echo cancellation (01_raw), and
                        each speech segment as the STT receives it

    --- Turn detection ---
    SMART_TURN_THRESHOLD
                        Probability from which a turn is complete (default: 0.5);
                        raise it (0.7-0.8) if a pause mid-sentence still gets answered,
                        at the cost of waiting TURN_WAIT_MS on more sentences.
    TURN_WAIT_MS        Silence after a turn judged unfinished before it is answered
                        (default: 1500)
    SMART_TURN          0 to turn Smart Turn v3 off (default: on when the
                        smart-turn-v3.2-cpu local model is downloaded; add
                        --extra smart-turn). Without it, every pause of 0.6 s
                        ends the turn.

    --- Tools (MCP, stdio) ---
    MCP_COMMAND         Command line of an MCP server whose tools the assistant
                        may use (default: none). Started with the assistant,
                        stopped with it; it gets a minimal environment.

    --- STT and VAD (sherpa-onnx, CPU) ---
    ROOMKIT_MODELS_DIR  Where the downloaded models live
                        (default: ~/.cache/roomkit/models)
    STT_MODEL           Streaming STT local model (default:
                        zipformer-fr-kroko-2025-08-06). With
                        nemotron-3.5-asr-streaming-0.6b-560ms-int8
                        (multilingual, 475 MB) the stream is told French.
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.5)

    --- Pocket TTS ---
    POCKET_DEVICE       cpu | cuda (default: cpu)
    POCKET_LANGUAGE     french | french_24l (default: french). french_24l
                        sounds better but is ~3x slower, and often stops a
                        long sentence at its first comma.
    POCKET_VOICE        A pre-made voice (default: estelle, the French one) or
                        the path of a clean speech clip to clone
    POCKET_QUANTIZE     1 for int8 quantization, CPU only (default: 0)
    POCKET_DISABLE_CUDNN
                        1 if the GPU run fails with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
                        (a system cuDNN shadowing PyTorch's)

    --- Audio ---
    AEC                 Echo cancellation: webrtc | speex | 0 (default: webrtc)
    MUTE_MIC            Mute the mic while the assistant speaks: 1 | 0
                        (default: 0 with AEC). Muting disables barge-in.

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    build_interruption,
    enable_voice_debug,
    log_tool_call,
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
    VoiceChannel,
)
from roomkit.local_models import is_downloaded
from roomkit.providers.ai import AIProvider
from roomkit.providers.llamacpp import LlamaCppAIProvider, LlamaCppConfig
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.tools import MCPToolProvider
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.pipeline import (
    AECProvider,
    AudioPipelineConfig,
    PipelineDebugTaps,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)
from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector
from roomkit.voice.stt import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts import PocketTTSConfig, PocketTTSProvider, StripEmoji

logger = setup_logging("voice_local_pocket_fr")
# Keep the terminal readable: request lines and Pocket's per-sentence timers.
for _noisy in ("httpx", "httpx2", "pocket_tts"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

MIC_RATE = 16000
BLOCK_MS = 20

STT_MODEL = os.environ.get("STT_MODEL", "zipformer-fr-kroko-2025-08-06")
TURN_MODEL = "smart-turn-v3.2-cpu"

SYSTEM_PROMPT = (
    "You are a warm voice assistant having a spoken conversation. "
    "Always answer in French, in one or two short, natural sentences, the way people talk. "
    "Never use lists, markdown or emojis."
)
TOOLS_PROMPT = (
    " When a question needs a tool, call it at once: never say you are going to look"
    " something up without calling the tool in the same reply. Only state facts a tool"
    " returned; if you do not have them, call the tool again or say you do not know."
    " Then say what it returned in a few spoken words, never as raw data. For more"
    " than five items, say how many there are, name the first three, and offer the rest."
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
            # Tool definitions and results take room: MCP servers are verbose.
            # 24k tokens of Qwen3-4B take ~6 GB of VRAM.
            context_size=24576,
        )
    )


async def start_llm(provider: AIProvider) -> None:
    """Load a llama.cpp model before the first word; an Ollama server is already up."""
    if isinstance(provider, LlamaCppAIProvider):
        await provider.start()


def build_turn_detector() -> SmartTurnDetector | None:
    """Smart Turn v3 when its model was downloaded: it hears whether a sentence is over.

    The VAD alone ends a turn at every 0.6 s pause, so "how many boards do I
    see, [pause] and which has the most cards?" became two turns and two
    stacked answers. A turn Smart Turn judges unfinished waits for more speech,
    and is still answered after 1.5 s of silence (turn_incomplete_wait_ms).
    """
    if os.environ.get("SMART_TURN") == "0":
        logger.info("Smart Turn off (SMART_TURN=0): turns end at every VAD pause")
        return None
    if not is_downloaded(TURN_MODEL):
        logger.info("Smart Turn not downloaded (%s): turns end at every VAD pause", TURN_MODEL)
        return None
    try:
        threshold = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))
        detector = SmartTurnDetector(
            SmartTurnConfig.from_local_model(TURN_MODEL, threshold=threshold)
        )
    except ImportError as exc:
        logger.warning("Smart Turn needs its extra (--extra smart-turn): %s", exc)
        return None
    logger.info("Turn detection: Smart Turn v3 (%s, threshold %.2f)", TURN_MODEL, threshold)
    return detector


def build_debug_taps() -> PipelineDebugTaps | None:
    """Record what the mic heard and what the STT received, to replay it offline.

    The echo canceller runs in LocalAudioBackend, before the pipeline: ``raw`` is
    already echo-cancelled, and the backend taps the mic as captured
    (``transport_raw``) and the reference its AEC received (``aec_reference``),
    aligned with ``raw`` sample for sample."""
    out = os.environ.get("DEBUG_AUDIO_DIR")
    if not out:
        return None
    logger.info("Recording the mic audio to %s", out)
    return PipelineDebugTaps(
        output_dir=out,
        stages=["transport_raw", "aec_reference", "raw", "post_vad_speech"],
    )


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
    # The MCP server, when there is one, lives as long as the assistant.
    async with AsyncExitStack() as stack:
        await run(stack)


async def run(stack: AsyncExitStack) -> None:
    require_local_models("ten-vad", STT_MODEL)
    if os.environ.get("POCKET_DISABLE_CUDNN") == "1":
        import torch

        torch.backends.cudnn.enabled = False

    kit = RoomKit()

    # --- Microphone and speaker ----------------------------------------------
    # With echo cancellation the mic stays open while the assistant speaks, so
    # you can interrupt it; without, it is muted during playback.
    aec = build_aec()
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env == "1" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=MIC_RATE,
        output_sample_rate=24000,  # Pocket TTS speaks 24 kHz
        channels=1,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )
    logger.info("Audio: AEC=%s, mic muted during playback=%s", type(aec).__name__, mute_mic)

    # --- VAD and STT (sherpa-onnx, CPU) ----------------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig.from_local_model(
            "ten-vad",
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            sample_rate=MIC_RATE,
            provider="cpu",
        )
    )
    # Kroko French's local model pads 1.5 s of tail silence: below that it
    # drops the last word ("midi" -> "mid").
    stt_config = SherpaOnnxSTTConfig.from_local_model(
        STT_MODEL, sample_rate=MIC_RATE, provider="cpu"
    )
    if stt_config.stream_language:
        # A multilingual streaming model (Nemotron 3.5) is told French.
        stt_config = replace(stt_config, stream_language="fr")
    stt = SherpaOnnxSTTProvider(stt_config)

    # --- Pocket TTS: French, on the CPU or the GPU -----------------------------
    voice_name = os.environ.get("POCKET_VOICE", "estelle")
    tts = PocketTTSProvider(
        PocketTTSConfig(
            language=os.environ.get("POCKET_LANGUAGE", "french"),
            voices={"assistant": voice_name},
            device=os.environ.get("POCKET_DEVICE", "cpu"),
            quantize=os.environ.get("POCKET_QUANTIZE") == "1",
        )
    )

    # --- LLM: thinking off, a spoken reply cannot wait ----------------------------
    ai_provider = build_llm()
    stack.push_async_callback(ai_provider.close)  # stops llama-server on any exit
    logger.info(
        "LLM: %s (%s), Pocket TTS voice: %s", ai_provider.model_name, ai_provider.name, voice_name
    )

    # --- Tools from an MCP server (optional) -------------------------------------
    system_prompt = os.environ.get("SYSTEM_PROMPT", SYSTEM_PROMPT)
    tool_kwargs: dict = {}
    mcp_command = shlex.split(os.environ.get("MCP_COMMAND", ""))
    if mcp_command:
        mcp = await stack.enter_async_context(
            MCPToolProvider.from_command(mcp_command[0], mcp_command[1:])
        )
        logger.info("MCP tools: %s", ", ".join(mcp.tool_names))
        tool_kwargs = {"tools": mcp.get_tools(), "tool_handler": mcp.as_tool_handler()}
        system_prompt += TOOLS_PROMPT

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
            turn_incomplete_wait_ms=float(os.environ.get("TURN_WAIT_MS", "1500")),
            debug_taps=build_debug_taps(),
        ),
        # The LLM adds emoji despite the prompt; spoken, they sound wrong.
        tts_filter=StripEmoji(),
        # The assistant keeps talking through a "d'accord" and stops for anything else.
        interruption=build_interruption(),
    )
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=ai_provider,
            system_prompt=system_prompt,
            # A result above this is set aside behind a preview that a small
            # model does not page through; keep a whole list of boards in view.
            evict_threshold_tokens=12000,
            **tool_kwargs,
        )
    )
    await kit.create_room(room_id="local-pocket-fr")
    await kit.attach_channel("local-pocket-fr", "ai", category=ChannelCategory.INTELLIGENCE)

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

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def on_tool_call(event, ctx):
        return log_tool_call(event, show_result=True)

    if os.environ.get("VOICE_DEBUG") == "1":
        enable_voice_debug(kit)

    # --- Load everything before the first word -----------------------------------
    logger.info("Loading the LLM, Pocket TTS, STT, VAD and turn models...")
    warmups = [stt.warmup(), tts.warmup(), start_llm(ai_provider)]
    if turn_detector is not None:
        warmups.append(asyncio.to_thread(turn_detector.warmup))
    await asyncio.gather(*warmups)
    await kit.attach_channel("local-pocket-fr", "voice")  # opens the mic
    logger.info("Ready: speak French into the microphone. Ctrl+C to stop.")

    await run_until_stopped(kit)


if __name__ == "__main__":
    asyncio.run(main())
