"""RoomKit -- Fully local voice assistant with sherpa-onnx + local LLM.

Everything runs locally — no cloud APIs required:
  - sherpa-onnx for speech-to-text (Zipformer transducer streaming, or Whisper offline)
  - Local LLM via any OpenAI-compatible server (Ollama, vLLM, LM Studio, etc.)
  - sherpa-onnx for text-to-speech (VITS/Piper models)
  - sherpa-onnx neural VAD (TEN-VAD or Silero)
  - Optional: sherpa-onnx GTCRN denoiser, SpeexAEC echo cancellation
  - All neural models support CUDA acceleration

Audio flows through the full pipeline:

  Mic → [Resampler] → [Recorder tap] → [AEC] → [Denoiser] → VAD
  → sherpa-onnx STT → Local LLM → sherpa-onnx TTS → [Recorder tap] → Speaker

Requirements:
    pip install roomkit[local-audio,openai,sherpa-onnx]
    A local LLM server — pick one:
      Ollama:   ollama pull qwen3:8b && ollama serve
      vLLM:     vllm serve Qwen/Qwen3-8B --port 8000

    Recommended LLM by GPU VRAM:
      24GB+ VRAM  →  qwen3:30b  (MoE, only ~3B active — fast + best quality)
      12GB VRAM   →  qwen3:8b   (best balance for most setups)
      8GB VRAM    →  qwen3:4b   (fast, good quality, leaves room for ONNX CUDA)
      CPU only    →  qwen3:4b   (still usable, ~1-2s latency)

    System (optional): libspeexdsp (apt install libspeexdsp1) for AEC

GPU acceleration (CUDA) for sherpa-onnx:
    The default sherpa-onnx pip package is CPU-only. For GPU support:

    1. Install cuDNN 9 (system library, once):
       # Add NVIDIA repo (Ubuntu 24.04 — change for your version)
       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
       sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
       sudo apt-get -y install cudnn9-cuda-12

    2. Install the CUDA 12 wheel:
       uv pip install sherpa-onnx==1.12.23+cuda12.cudnn9 \
           -f https://k2-fsa.github.io/sherpa/onnx/cuda.html

    3. Set the env var:
       export ONNX_PROVIDER=cuda

    See docs/sherpa-onnx.md for troubleshooting and CUDA 11 instructions.

Models (download once):
    # VAD — TEN-VAD (recommended)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

    # STT — Zipformer transducer (streaming)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
    tar xf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

    # TTS — VITS (Piper voices)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xf vits-piper-en_US-amy-low.tar.bz2

    # Denoiser — GTCRN (optional)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

Run with Ollama (recommended):
    ollama pull qwen3:8b

    LLM_MODEL=qwen3:8b \\
    LLM_BASE_URL=http://localhost:11434/v1 \\
    VAD_MODEL=ten-vad.onnx \\
    STT_ENCODER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \\
    STT_DECODER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \\
    STT_JOINER=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \\
    STT_TOKENS=sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \\
    TTS_MODEL=vits-piper-en_US-amy-low/en_US-amy-low.onnx \\
    TTS_TOKENS=vits-piper-en_US-amy-low/tokens.txt \\
    TTS_DATA_DIR=vits-piper-en_US-amy-low/espeak-ng-data \\
    uv run python examples/voice_local_onnx_vllm.py

Run with vLLM:
    LLM_MODEL=Qwen/Qwen3-8B \\
    LLM_BASE_URL=http://localhost:8000/v1 \\
    VAD_MODEL=ten-vad.onnx \\
    STT_ENCODER=... STT_DECODER=... STT_JOINER=... STT_TOKENS=... \\
    TTS_MODEL=... TTS_TOKENS=... TTS_DATA_DIR=... \\
    uv run python examples/voice_local_onnx_vllm.py

Environment variables:
    --- LLM (any OpenAI-compatible server) ---
    LLM_MODEL           (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL        Server endpoint (default: http://localhost:11434/v1 — Ollama)
    LLM_API_KEY         API key if server requires auth (default: none)
    LLM_MAX_TOKENS      Max response tokens (default: 256)
    SYSTEM_PROMPT       Custom system prompt

    --- STT (sherpa-onnx) ---
    STT_MODE            Recognition mode: transducer | whisper (default: transducer)
    STT_ENCODER         (required) Path to encoder .onnx
    STT_DECODER         (required) Path to decoder .onnx
    STT_JOINER          Path to joiner .onnx (transducer only)
    STT_TOKENS          (required) Path to tokens.txt
    STT_LANGUAGE        Language code for Whisper mode (default: en)

    --- TTS (sherpa-onnx) ---
    TTS_MODEL           (required) Path to VITS/Piper .onnx model
    TTS_TOKENS          (required) Path to tokens.txt
    TTS_DATA_DIR        Path to espeak-ng data dir (Piper models)
    TTS_SPEAKER_ID      Speaker ID for multi-speaker models (default: 0)
    TTS_SPEED           Speech speed multiplier (default: 1.0)
    TTS_SAMPLE_RATE     Output sample rate (default: 22050)

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           (required) Path to VAD .onnx model
    VAD_MODEL_TYPE      Model type: ten | silero (default: ten)
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)
                        Lower values improve sensitivity for short utterances.
                        The GTCRN denoiser slightly alters spectral features,
                        which reduces TEN-VAD confidence — 0.35 compensates.
                        Without denoiser you can raise to 0.5 for fewer false
                        positives.

    --- Pipeline (optional) ---
    DENOISE_MODEL       Path to GTCRN .onnx model (enables denoiser)
    AEC                 Echo cancellation: speex | webrtc | 1 (=speex) | 0 (default: 0)
    MUTE_MIC            Mute mic during playback: 1 | 0 (default: auto, off with AEC)
    RECORDING_DIR       Directory for WAV recordings (default: ./recordings)
    RECORDING_MODE      Channel mode: mixed | separate | stereo (default: stereo)
    DEBUG_TAPS_DIR      Directory for pipeline debug taps (disabled if unset)
    ONNX_PROVIDER       ONNX execution provider for STT/TTS: cpu | cuda (default: cpu)
                        VAD and denoiser always use CPU (per-frame overhead makes
                        CUDA slower for these tiny models)

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
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    PipelineDebugTaps,
    RecordingChannelMode,
    RecordingConfig,
    WavFileRecorder,
)
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_local_onnx_vllm")

CHANNEL_MODES = {
    "mixed": RecordingChannelMode.MIXED,
    "separate": RecordingChannelMode.SEPARATE,
    "stereo": RecordingChannelMode.STEREO,
}


def check_env() -> None:
    """Check required environment variables."""
    required = {
        "LLM_MODEL": "Model name (e.g. qwen2.5:7b for Ollama)",
        "VAD_MODEL": "Path to VAD .onnx model (e.g. ten-vad.onnx)",
        "STT_ENCODER": "Path to STT encoder .onnx",
        "STT_DECODER": "Path to STT decoder .onnx",
        "STT_TOKENS": "Path to STT tokens.txt",
        "TTS_MODEL": "Path to TTS VITS/Piper .onnx model",
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

    # --- ONNX execution providers ----------------------------------------------
    # CUDA accelerates STT and TTS (batch inference on larger models).
    # VAD and denoiser run per-frame (every 20ms) on tiny models — the GPU
    # transfer overhead makes them *slower* on CUDA, so they always use CPU.
    onnx_provider = os.environ.get("ONNX_PROVIDER", "cpu")
    if onnx_provider == "cuda":
        logger.info("ONNX execution provider: CUDA (GPU) for STT/TTS, CPU for VAD/denoiser")
    else:
        logger.info("ONNX execution provider: CPU")

    # --- Audio settings -------------------------------------------------------
    sample_rate = 16000
    block_ms = 20
    frame_size = sample_rate * block_ms // 1000  # 320 samples

    tts_sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "22050"))

    # --- AEC (optional, echo cancellation) ------------------------------------
    aec = None
    aec_mode = os.environ.get("AEC", "0").lower()
    if aec_mode in ("1", "speex"):
        from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

        aec = SpeexAECProvider(
            frame_size=frame_size,
            filter_length=frame_size * 10,
            sample_rate=sample_rate,
        )
        logger.info("AEC enabled (Speex, filter=%d samples)", frame_size * 10)
    elif aec_mode == "webrtc":
        from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

        aec = WebRTCAECProvider(sample_rate=sample_rate)
        logger.info("AEC enabled (WebRTC AEC3)")

    # --- Backend: local mic + speakers ----------------------------------------
    # When AEC is active it removes speaker echo from the mic signal, so we
    # can keep the mic open during playback and allow barge-in interruption.
    # Without AEC the mic is muted during playback to prevent feedback loops.
    # Override with MUTE_MIC=0|1 for testing (e.g. MUTE_MIC=0 to test AEC).
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=tts_sample_rate,
        channels=1,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )

    # --- Denoiser (optional, sherpa-onnx GTCRN) -------------------------------
    denoiser = None
    denoise_model = os.environ.get("DENOISE_MODEL", "")
    if denoise_model:
        from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
            SherpaOnnxDenoiserConfig,
            SherpaOnnxDenoiserProvider,
        )

        denoiser = SherpaOnnxDenoiserProvider(
            SherpaOnnxDenoiserConfig(model=denoise_model, provider="cpu")
        )
        logger.info("Denoiser: sherpa-onnx GTCRN (model=%s)", denoise_model)

    # --- VAD (sherpa-onnx neural VAD) -----------------------------------------
    vad_model = os.environ["VAD_MODEL"]
    vad_model_type = os.environ.get("VAD_MODEL_TYPE", "ten")
    vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.35"))
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=vad_model,
            model_type=vad_model_type,
            threshold=vad_threshold,
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
            speech_pad_ms=300,
            sample_rate=sample_rate,
            provider="cpu",
        )
    )
    logger.info(
        "VAD: sherpa-onnx (type=%s, threshold=%.2f, model=%s)",
        vad_model_type,
        vad_threshold,
        vad_model,
    )

    # --- WAV recorder (optional debug audio capture) --------------------------
    recording_dir = os.environ.get("RECORDING_DIR", "./recordings")
    rec_mode_name = os.environ.get("RECORDING_MODE", "stereo").lower()
    rec_channel_mode = CHANNEL_MODES.get(rec_mode_name, RecordingChannelMode.STEREO)

    recorder = WavFileRecorder()
    recording_config = RecordingConfig(
        storage=recording_dir,
        channels=rec_channel_mode,
    )
    logger.info("Recording to %s (mode=%s)", recording_dir, rec_mode_name)

    # --- Debug taps (optional pipeline stage capture) -------------------------
    debug_taps = None
    debug_taps_dir = os.environ.get("DEBUG_TAPS_DIR", "")
    if debug_taps_dir:
        stages_env = os.environ.get("DEBUG_TAPS_STAGES", "all")
        stages = [s.strip() for s in stages_env.split(",")]
        debug_taps = PipelineDebugTaps(output_dir=debug_taps_dir, stages=stages)
        logger.info("Debug taps: %s (stages=%s)", debug_taps_dir, stages)

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        aec=aec,
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
        debug_taps=debug_taps,
    )

    # --- STT (sherpa-onnx) ----------------------------------------------------
    stt_mode = os.environ.get("STT_MODE", "transducer")
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig(
            mode=stt_mode,
            encoder=os.environ["STT_ENCODER"],
            decoder=os.environ["STT_DECODER"],
            joiner=os.environ.get("STT_JOINER", ""),
            tokens=os.environ["STT_TOKENS"],
            language=os.environ.get("STT_LANGUAGE", "en"),
            sample_rate=sample_rate,
            provider=onnx_provider,
        )
    )
    logger.info(
        "STT: sherpa-onnx (mode=%s, encoder=%s)",
        stt_mode,
        os.environ["STT_ENCODER"],
    )

    # --- TTS (sherpa-onnx) ----------------------------------------------------
    tts = SherpaOnnxTTSProvider(
        SherpaOnnxTTSConfig(
            model=os.environ["TTS_MODEL"],
            tokens=os.environ["TTS_TOKENS"],
            data_dir=os.environ.get("TTS_DATA_DIR", ""),
            speaker_id=int(os.environ.get("TTS_SPEAKER_ID", "0")),
            speed=float(os.environ.get("TTS_SPEED", "1.0")),
            sample_rate=tts_sample_rate,
            provider=onnx_provider,
        )
    )
    logger.info(
        "TTS: sherpa-onnx (model=%s, rate=%d)",
        os.environ["TTS_MODEL"],
        tts_sample_rate,
    )

    # --- LLM (local via OpenAI-compatible API: Ollama, vLLM, etc.) ------------
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
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=system_prompt,
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="local-voice")
    await kit.attach_channel("local-voice", "voice")
    await kit.attach_channel("local-voice", "ai", category=ChannelCategory.INTELLIGENCE)

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

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
    async def on_rec_started(event, ctx):
        logger.info("Recording started: %s", event.id)

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
    async def on_rec_stopped(event, ctx):
        logger.info(
            "Recording stopped: %s (%.1fs, %d bytes, files=%s)",
            event.id,
            event.duration_seconds,
            event.size_bytes,
            event.urls,
        )

    # --- Warmup: pre-load models (avoids delay on first interaction) ----------
    logger.info("Loading models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded — ready!")

    # --- Start voice session --------------------------------------------------
    session = await backend.connect("local-voice", "local-user", "voice")
    binding = ChannelBinding(
        room_id="local-voice",
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "local-voice", binding)

    logger.info("")
    logger.info("All local — no cloud APIs needed!")
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
    logger.info("Done. Recordings saved to: %s", recording_dir)


if __name__ == "__main__":
    asyncio.run(main())
