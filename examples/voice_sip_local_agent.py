#!/usr/bin/env python3
"""RoomKit — SIP calls routed to a fully local AI agent (no cloud APIs).

Incoming SIP calls from a PBX/trunk are answered automatically and processed
through a fully local pipeline:

  Phone → SIP/RTP → [Resampler] → [AEC] → [Denoiser] → VAD
  → sherpa-onnx STT → Local LLM → sherpa-onnx TTS → SIP/RTP → Phone

Everything runs on your server — no cloud APIs, no data leaves the box.
Ideal for privacy-sensitive deployments, air-gapped environments, or
on-premise telephony AI.

The PBX/proxy should set ``X-Room-ID`` and ``X-Session-ID`` SIP headers
before forwarding INVITEs to this server.

Requirements:
    pip install roomkit[sip,openai,sherpa-onnx]
    A local LLM server — pick one:
      Ollama:   ollama pull qwen3:8b && ollama serve
      vLLM:     vllm serve Qwen/Qwen3-8B --port 8000

Models (download once):
    # VAD — TEN-VAD (recommended)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

    # STT — Zipformer transducer (streaming)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
    tar xf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

    # TTS — VITS (Piper voices)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xf vits-piper-en_US-amy-low.tar.bz2

    # Denoiser — GTCRN (optional, recommended for noisy phone lines)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

Usage:
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
    python examples/voice_sip_local_agent.py

Environment variables:
    --- SIP ---
    SIP_HOST            Listening address (default: 0.0.0.0)
    SIP_PORT            SIP port (default: 5060)
    RTP_IP              RTP bind address (default: 0.0.0.0)
    RTP_PORT_START      RTP port range start (default: 10000)
    RTP_PORT_END        RTP port range end (default: 20000)

    --- LLM (any OpenAI-compatible server) ---
    LLM_MODEL           (required) Model name (e.g. qwen3:8b for Ollama)
    LLM_BASE_URL        Server endpoint (default: http://localhost:11434/v1)
    LLM_API_KEY         API key if server requires auth (default: none)
    LLM_MAX_TOKENS      Max response tokens (default: 256)
    SYSTEM_PROMPT       Custom system prompt

    --- STT (sherpa-onnx) ---
    STT_MODE            Recognition mode: transducer | whisper (default: transducer)
    STT_ENCODER         (required) Path to encoder .onnx
    STT_DECODER         (required) Path to decoder .onnx
    STT_JOINER          Path to joiner .onnx (transducer only)
    STT_TOKENS          (required) Path to tokens.txt

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

    --- Pipeline (optional) ---
    DENOISE_MODEL       Path to GTCRN .onnx model (enables denoiser)
    RECORDING_DIR       Directory for WAV recordings (default: ./recordings)
    RECORDING_MODE      Channel mode: mixed | separate | stereo (default: stereo)
    ONNX_PROVIDER       ONNX execution provider: cpu | cuda (default: cpu)
"""

import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("sip_local_agent")

from roomkit import (
    ChannelCategory,
    HookResult,
    HookTrigger,
    RoomKit,
    VLLMConfig,
    VoiceChannel,
    create_vllm_provider,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.models.trace import ProtocolTrace
from roomkit.voice import parse_voice_session
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.pipeline import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
    RecordingChannelMode,
    RecordingConfig,
    WavFileRecorder,
)
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = os.environ.get("SIP_HOST", "0.0.0.0")  # nosec B104
SIP_PORT = int(os.environ.get("SIP_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")  # nosec B104
RTP_PORT_START = int(os.environ.get("RTP_PORT_START", "10000"))
RTP_PORT_END = int(os.environ.get("RTP_PORT_END", "20000"))

CHANNEL_MODES = {
    "mixed": RecordingChannelMode.MIXED,
    "separate": RecordingChannelMode.SEPARATE,
    "stereo": RecordingChannelMode.STEREO,
}


def check_env() -> None:
    """Check required environment variables."""
    required = {
        "LLM_MODEL": "Model name (e.g. qwen3:8b for Ollama)",
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

    # --- ONNX execution providers ----------------------------------------
    onnx_provider = os.environ.get("ONNX_PROVIDER", "cpu")
    if onnx_provider == "cuda":
        logger.info("ONNX: CUDA (GPU) for STT/TTS, CPU for VAD/denoiser")
    else:
        logger.info("ONNX: CPU")

    # --- Audio settings --------------------------------------------------
    sample_rate = 16000
    tts_sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "22050"))

    # --- SIP backend (answers incoming calls) ----------------------------
    sip = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
        user_agent="RoomKit/LocalAgent",
        server_name="RoomKit",
    )

    # --- Denoiser (optional, sherpa-onnx GTCRN) --------------------------
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
        logger.info("Denoiser: sherpa-onnx GTCRN (%s)", denoise_model)

    # --- VAD (sherpa-onnx neural VAD) ------------------------------------
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
    logger.info("VAD: %s (threshold=%.2f)", vad_model_type, vad_threshold)

    # --- WAV recorder (optional) -----------------------------------------
    recording_dir = os.environ.get("RECORDING_DIR", "./recordings")
    rec_mode_name = os.environ.get("RECORDING_MODE", "stereo").lower()
    rec_channel_mode = CHANNEL_MODES.get(rec_mode_name, RecordingChannelMode.STEREO)
    recorder = WavFileRecorder()
    recording_config = RecordingConfig(storage=recording_dir, channels=rec_channel_mode)
    logger.info("Recording to %s (mode=%s)", recording_dir, rec_mode_name)

    # --- Pipeline contract -----------------------------------------------
    # SIP negotiates G.722 (16 kHz) or G.711 (8 kHz). TTS may output at a
    # different rate (e.g. 22050 Hz for Piper). The contract tells the
    # pipeline to resample outbound audio to match the SIP codec rate.
    sip_rate = 16000  # G.722 preferred; change to 8000 if only G.711
    contract = AudioPipelineContract(
        transport_inbound_format=AudioFormat(sample_rate=sip_rate),
        transport_outbound_format=AudioFormat(sample_rate=sip_rate),
        internal_format=AudioFormat(sample_rate=sample_rate),
    )

    # --- Pipeline config -------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        denoiser=denoiser,
        recorder=recorder,
        recording_config=recording_config,
        contract=contract,
    )

    # --- STT (sherpa-onnx) -----------------------------------------------
    stt_mode = os.environ.get("STT_MODE", "transducer")
    stt = SherpaOnnxSTTProvider(
        SherpaOnnxSTTConfig(
            mode=stt_mode,
            encoder=os.environ["STT_ENCODER"],
            decoder=os.environ["STT_DECODER"],
            joiner=os.environ.get("STT_JOINER", ""),
            tokens=os.environ["STT_TOKENS"],
            sample_rate=sample_rate,
            provider=onnx_provider,
        )
    )
    logger.info("STT: sherpa-onnx (%s)", stt_mode)

    # --- TTS (sherpa-onnx) -----------------------------------------------
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
    logger.info("TTS: sherpa-onnx (rate=%d)", tts_sample_rate)

    # --- LLM (local via OpenAI-compatible API) ---------------------------
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
    logger.info("LLM: %s (%s)", llm_model, llm_base_url)

    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a friendly phone assistant. Keep your responses "
        "short and conversational — one or two sentences at most. "
        "Answer directly without thinking, reasoning, or internal monologue. "
        "/no_think",
    )

    # --- Channels --------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=sip,
        pipeline=pipeline_config,
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
    kit.register_channel(ai)

    # -------------------------------------------------------------------
    # Hooks — same hooks work for text AND voice
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def gate_incoming(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Log incoming voice sessions. Could block spam callers here."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            caller = event.content.data.get("caller", "unknown")
            logger.info(
                "BEFORE_BROADCAST — call from %s in room %s",
                caller,
                ctx.room.id,
            )
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text: str, ctx: RoomContext) -> HookResult:
        """Log what the caller says."""
        logger.info("Caller: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text: str, ctx: RoomContext) -> HookResult:
        """Log what the AI responds."""
        logger.info("Agent: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
    async def on_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
        """Log SIP signaling at the room level."""
        logger.info(
            "ON_PROTOCOL_TRACE [room=%s] %s %s: %s",
            ctx.room.id,
            trace.direction,
            trace.protocol,
            trace.summary,
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def log_events(event: RoomEvent, ctx: RoomContext) -> None:
        """Log every event that flows through the room."""
        logger.info(
            "AFTER_BROADCAST — type=%s channel=%s room=%s provider=%s",
            event.type,
            event.source.channel_id,
            ctx.room.id,
            event.source.provider,
        )

    # -------------------------------------------------------------------
    # Incoming call → unified process_inbound (same as text)
    # -------------------------------------------------------------------

    @sip.on_call
    async def handle_call(session):
        room_id = session.metadata.get("room_id")
        result = await kit.process_inbound(
            parse_voice_session(session, channel_id="voice"),
            room_id=room_id,
        )
        if result.blocked:
            logger.warning("Call rejected: %s", result.reason)
        else:
            # Attach AI channel so the room has intelligence
            actual_room_id = room_id or session.room_id
            await kit.attach_channel(actual_room_id, "ai", category=ChannelCategory.INTELLIGENCE)
            logger.info("Call connected — session=%s room=%s", session.id, actual_room_id)

    # -------------------------------------------------------------------
    # Remote hangup → cleanup
    # -------------------------------------------------------------------

    @sip.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("Call ended — session=%s", session.id)
        room_id = session.metadata.get("room_id", session.id)
        await kit.close_room(room_id)

    # -------------------------------------------------------------------
    # Warmup models + start
    # -------------------------------------------------------------------

    logger.info("Loading ONNX models...")
    await asyncio.gather(stt.warmup(), tts.warmup())
    logger.info("Models loaded!")

    await sip.start()
    logger.info(
        "SIP + Local Agent ready — SIP %s:%d, RTP %d-%d",
        SIP_HOST,
        SIP_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )
    logger.info("LLM: %s — STT: sherpa-onnx — TTS: sherpa-onnx — no cloud APIs", llm_model)
    logger.info("Waiting for incoming SIP calls...")

    try:
        await asyncio.Event().wait()
    finally:
        await sip.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
