#!/usr/bin/env python3
"""Voice orchestration: SIP call with multi-agent pipeline + background delegation.

A SIP call arrives, gets triaged by a voice AI agent, then hands off to
an advisor — all on the same SIP session. The caller never hears a
disconnect.

When the caller asks about insurance details, the advisor delegates the
lookup to a background agent via ``kit.delegate()``. The voice conversation
continues while the insurance agent works in a child room. Once the result
is ready, it's injected into the advisor's context and shared on the next turn.

    SIP Call → VoiceChannel → Triage → Advisor → (background: Insurance lookup)

Requirements:
    pip install roomkit[sip,gemini,deepgram,elevenlabs,sherpa-onnx]

Models (download once):
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

Run with:
    GOOGLE_API_KEY=... DEEPGRAM_API_KEY=... ELEVENLABS_API_KEY=... \\
    VAD_MODEL=ten-vad.onnx python examples/orchestration_voice_triage.py

Environment variables:
    SIP_HOST, SIP_PORT, RTP_IP, RTP_PORT_START, RTP_PORT_END
    GOOGLE_API_KEY, GEMINI_MODEL
    DEEPGRAM_API_KEY, STT_LANGUAGE
    ELEVENLABS_API_KEY, VOICE_TRIAGE, VOICE_ADVISOR
    VAD_MODEL, VAD_THRESHOLD
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from shared.env import require_env

from roomkit import (
    Agent,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
    WaitForIdle,
)
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.orchestration.handoff import HandoffMemoryProvider
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import (
    ConversationState,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig
from roomkit.tasks.delegate import DelegateHandler, build_delegate_tool, setup_delegation
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import (
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider
from roomkit.voice.tts.filters import StripInternalTags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_triage")
logging.getLogger("roomkit.core.event_router").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = os.environ.get("SIP_HOST", "0.0.0.0")  # nosec B104
SIP_PORT = int(os.environ.get("SIP_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")  # nosec B104
RTP_PORT_START = int(os.environ.get("RTP_PORT_START", "10000"))
RTP_PORT_END = int(os.environ.get("RTP_PORT_END", "20000"))

VOICE_TRIAGE = os.environ.get("VOICE_TRIAGE", "21m00Tcm4TlvDq8ikWAM")
VOICE_ADVISOR = os.environ.get("VOICE_ADVISOR", "pNInz6obpgDQGcFmaJgB")

# ---------------------------------------------------------------------------
# Pipeline: intake (triage) -> handling (advisor)
# ---------------------------------------------------------------------------

pipeline = ConversationPipeline(
    stages=[
        PipelineStage(phase="intake", agent_id="agent-triage", next="handling"),
        PipelineStage(phase="handling", agent_id="agent-advisor", next="intake"),
    ],
)


async def main() -> None:
    env = require_env("GOOGLE_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "VAD_MODEL")

    kit = RoomKit(delivery_strategy=WaitForIdle())

    # --- SIP backend ---------------------------------------------------------

    sip = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
    )

    # --- VAD -----------------------------------------------------------------

    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=env["VAD_MODEL"],
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.35")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
        )
    )

    # --- STT -----------------------------------------------------------------

    stt = DeepgramSTTProvider(
        DeepgramConfig(
            api_key=env["DEEPGRAM_API_KEY"],
            language=os.environ.get("STT_LANGUAGE", "en"),
        )
    )

    # --- TTS -----------------------------------------------------------------

    tts = ElevenLabsTTSProvider(
        ElevenLabsConfig(
            api_key=env["ELEVENLABS_API_KEY"],
            voice_id=VOICE_TRIAGE,
            output_format="pcm_16000",
        )
    )

    # --- Voice channel -------------------------------------------------------

    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=sip,
        pipeline=AudioPipelineConfig(vad=vad),
        tts_filter=StripInternalTags(),
    )
    kit.register_channel(voice)

    # --- AI agents -----------------------------------------------------------

    gemini_config = GeminiConfig(
        api_key=env["GOOGLE_API_KEY"],
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        max_tokens=150,
    )

    triage = Agent(
        "agent-triage",
        provider=GeminiAIProvider(gemini_config),
        role="Triage receptionist",
        description="Routes callers to the right financial specialist",
        voice=VOICE_TRIAGE,
        language="French",
        auto_greet=False,
        greeting=(
            "[A new caller just connected — greet them warmly "
            "and ask how you can help with financial services]"
        ),
        system_prompt=(
            "Greet callers warmly and identify their need. "
            "If the caller's request matches financial advisory, use the "
            "handoff_conversation tool to transfer them. "
            "If it does NOT match, politely explain that this firm only "
            "handles financial matters. "
            "Keep responses under 30 words. "
            "NEVER write handoff instructions as text — always use the tool. "
            "Wrap internal reasoning in [internal]...[/internal] tags. "
            "Never use markdown or special formatting."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=20)),
    )

    advisor = Agent(
        "agent-advisor",
        provider=GeminiAIProvider(gemini_config),
        role="Portfolio advisor",
        description="Provides financial and investment advice",
        voice=VOICE_ADVISOR,
        language="French",
        system_prompt=(
            "When you first join after a transfer, introduce yourself briefly. "
            "Give concise, conversational financial advice. "
            "When a caller asks about insurance, use the delegate_task tool. "
            "Tell the caller you're checking and keep chatting. "
            "When the result appears in your context, share it. "
            "To re-triage, use handoff_conversation. "
            "NEVER write handoff or delegation as text — use tools. "
            "Wrap internal reasoning in [internal]...[/internal] tags. "
            "Under 50 words per response. No markdown or formatting."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=20)),
    )

    insurance = Agent(
        "agent-insurance",
        provider=GeminiAIProvider(gemini_config),
        role="Insurance data specialist",
        description="Looks up client insurance policy details",
        system_prompt=(
            "You are a background insurance lookup agent. Simulate looking up "
            "their insurance policy and return realistic fake data: policy number, "
            "coverage type, monthly premium, deductible, next renewal date. "
            "Under 50 words. No markdown."
        ),
    )

    for ch in [triage, advisor, insurance]:
        kit.register_channel(ch)

    # --- Orchestration -------------------------------------------------------

    _router, _handler = pipeline.install(
        kit,
        [triage, advisor],
        greet_on_handoff=True,
        voice_channel_id="voice",
    )

    # --- Delegation (background agent) ---------------------------------------

    delegate_tool = build_delegate_tool(
        [("agent-insurance", "Looks up client insurance policy details")]
    )
    delegate_handler = DelegateHandler(kit, delivery_strategy=WaitForIdle())
    setup_delegation(advisor, delegate_handler, tool=delegate_tool)

    # --- Hooks ---------------------------------------------------------------

    _last_transcription_ts: dict[str, float] = {}

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx: RoomContext) -> HookResult:
        _last_transcription_ts[ctx.room.id] = time.monotonic()
        logger.info("[STT] Caller: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx: RoomContext) -> HookResult:
        t0 = _last_transcription_ts.get(ctx.room.id)
        latency = f" ({(time.monotonic() - t0) * 1000:.0f}ms)" if t0 else ""
        agent = get_conversation_state(ctx.room).active_agent_id or "?"
        logger.info("[TTS] agent=%s text=%.60s...%s", agent, text, latency)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event, ctx):
        logger.info("[delegated] %s", event.metadata.get("agent_id"))

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event, ctx):
        logger.info(
            "[completed] %s (%.0fms)",
            event.metadata.get("task_id"),
            event.metadata.get("duration_ms", 0),
        )

    # --- Incoming call handler -----------------------------------------------

    @sip.on_call
    async def handle_call(session):
        room_id = session.metadata.get("room_id") or session.id
        logger.info("Incoming call — session=%s", session.id)

        await kit.create_room(room_id=room_id, orchestration=None)
        await kit.attach_channel(room_id, "voice")
        await kit.join(room_id, "voice", session=session)

        await kit.attach_channel(room_id, "agent-triage", category=ChannelCategory.INTELLIGENCE)
        await kit.attach_channel(room_id, "agent-advisor", category=ChannelCategory.INTELLIGENCE)

        room = await kit.get_room(room_id)
        room = set_conversation_state(
            room,
            ConversationState(phase="intake", active_agent_id="agent-triage"),
        )
        await kit.store.update_room(room)

        await _handler.send_greeting(room_id, channel_id="voice")
        logger.info("Call connected — room=%s agent=triage", room_id)

    @sip.on_call_disconnected
    async def handle_disconnect(session):
        room_id = session.metadata.get("room_id", session.id)
        room = await kit.get_room(room_id)
        state = get_conversation_state(room)
        logger.info("Call ended — phase=%s handoffs=%d", state.phase, state.handoff_count)
        await kit.leave(session)

    # --- Start ---------------------------------------------------------------

    await sip.start()
    logger.info("Voice triage ready — SIP %s:%d", SIP_HOST, SIP_PORT)
    logger.info("Pipeline: intake (triage) -> handling (advisor)")
    logger.info("Waiting for incoming SIP calls...")

    try:
        await asyncio.Event().wait()
    finally:
        await sip.close()
        for ch in [triage, advisor, insurance]:
            await ch.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
