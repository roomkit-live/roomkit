#!/usr/bin/env python3
"""Voice orchestration: SIP call with multi-agent handoff + background delegation.

A SIP call arrives, gets triaged by a voice AI agent, then hands off to
an advisor — all on the same SIP session. The caller never hears a
disconnect.

When the caller asks about insurance details, the advisor delegates the
lookup to a background agent via ``kit.delegate()``. The voice conversation
continues while the insurance agent works in a child room. Once the result
is ready, it's injected into the advisor's context and shared on the next turn.

Architecture:

    SIP Call → VoiceChannel (STT) → transcript (RoomEvent)
                                          │
                                   ConversationRouter (BEFORE_BROADCAST hook)
                                          │
                              ┌───────────┴──────────┐
                              │                      │
                        agent-triage           agent-advisor
                        (phase: intake)        (phase: handling)
                              │                      │
                              └───────────┬──────────┘
                                          │
                                   text response (RoomEvent)
                                          │
                              VoiceChannel (TTS) → SIP audio out

    Background delegation (kit.delegate):

        agent-advisor ──delegate_task──→ child room
                                            │
                                      agent-insurance
                                      (background lookup)
                                            │
                                      result → injected into
                                      agent-advisor context

Requirements:
    pip install roomkit[sip,gemini,deepgram,elevenlabs,sherpa-onnx]

Models (download once):
    # VAD — TEN-VAD (2 MB, runs on CPU)
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

Usage:
    GOOGLE_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    VAD_MODEL=ten-vad.onnx \\
    python examples/orchestration_voice_triage.py

    # From a SIP client or PBX, send an INVITE to port 5060 with:
    #   X-Room-ID: my-room   (optional — auto-generated if absent)

Environment variables:
    --- SIP ---
    SIP_HOST            Listening address (default: 0.0.0.0)
    SIP_PORT            SIP port (default: 5060)
    RTP_IP              RTP bind address (default: 0.0.0.0)
    RTP_PORT_START      RTP port range start (default: 10000)
    RTP_PORT_END        RTP port range end (default: 20000)

    --- AI (Gemini) ---
    GOOGLE_API_KEY      (required) Google API key
    GEMINI_MODEL        Model name (default: gemini-2.0-flash)

    --- STT (Deepgram) ---
    DEEPGRAM_API_KEY    (required) Deepgram API key
    STT_LANGUAGE        STT language code (default: en)

    --- TTS (ElevenLabs) ---
    ELEVENLABS_API_KEY  (required) ElevenLabs API key
    VOICE_TRIAGE        Voice ID for triage agent (default: Rachel)
    VOICE_ADVISOR       Voice ID for advisor agent (default: Adam)

    --- VAD (sherpa-onnx) ---
    VAD_MODEL           (required) Path to VAD .onnx model
    VAD_THRESHOLD       Speech probability threshold 0-1 (default: 0.35)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_triage")

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit.core.event_router").setLevel(logging.ERROR)

from roomkit import (
    Agent,
    ChannelCategory,
    ConversationPipeline,
    ConversationState,
    DelegateHandler,
    GeminiAIProvider,
    GeminiConfig,
    HandoffMemoryProvider,
    HookExecution,
    HookResult,
    HookTrigger,
    PipelineStage,
    RoomKit,
    SlidingWindowMemory,
    VoiceChannel,
    WaitForIdleDelivery,
    build_delegate_tool,
    get_conversation_state,
    set_conversation_state,
    setup_delegation,
)
from roomkit.models.context import RoomContext
from roomkit.voice import parse_voice_session
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import (
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = os.environ.get("SIP_HOST", "0.0.0.0")  # nosec B104
SIP_PORT = int(os.environ.get("SIP_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")  # nosec B104
RTP_PORT_START = int(os.environ.get("RTP_PORT_START", "10000"))
RTP_PORT_END = int(os.environ.get("RTP_PORT_END", "20000"))

# Per-agent ElevenLabs voice IDs (Rachel for triage, Adam for advisor)
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


def check_env() -> None:
    """Check required environment variables."""
    required = {
        "GOOGLE_API_KEY": "Google API key (Gemini)",
        "DEEPGRAM_API_KEY": "Deepgram API key (STT)",
        "ELEVENLABS_API_KEY": "ElevenLabs API key (TTS)",
        "VAD_MODEL": "Path to VAD .onnx model (e.g. ten-vad.onnx)",
    }
    missing = [
        f"  {key:22s} — {desc}" for key, desc in required.items() if not os.environ.get(key)
    ]
    if missing:
        print("Missing required environment variables:\n")
        print("\n".join(missing))
        print("\nSee docstring at the top of this file for setup instructions.")
        sys.exit(1)


async def main() -> None:
    check_env()

    kit = RoomKit()

    # --- SIP backend ---------------------------------------------------------
    sip = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
    )

    # --- VAD (sherpa-onnx, runs locally on CPU) ------------------------------
    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=os.environ["VAD_MODEL"],
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.35")),
            silence_threshold_ms=600,
            min_speech_duration_ms=200,
        )
    )

    # --- STT (Deepgram, cloud) -----------------------------------------------
    stt_language = os.environ.get("STT_LANGUAGE", "en")
    stt = DeepgramSTTProvider(
        DeepgramConfig(api_key=os.environ["DEEPGRAM_API_KEY"], language=stt_language)
    )

    # --- TTS (ElevenLabs, cloud) ---------------------------------------------
    tts = ElevenLabsTTSProvider(
        ElevenLabsConfig(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice_id=VOICE_TRIAGE,  # initial voice; swapped on handoff
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
    )
    kit.register_channel(voice)

    # --- AI agents (Gemini) --------------------------------------------------
    # Both use voice-friendly prompts: short, no markdown, no code blocks.
    gemini_config = GeminiConfig(
        api_key=os.environ["GOOGLE_API_KEY"],
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        max_tokens=150,
    )

    triage = Agent(
        "agent-triage",
        provider=GeminiAIProvider(gemini_config),
        role="Triage receptionist",
        description="Routes callers to the right financial specialist",
        scope="Financial advisory services only",
        voice=VOICE_TRIAGE,
        language="French",
        auto_greet=False,  # handler.send_greeting() handles greeting via AI
        greeting=(
            "[A new caller just connected — greet them warmly "
            "and ask how you can help with financial services]"
        ),
        system_prompt=(
            "Greet callers warmly and identify their need. "
            "If the caller's request matches financial advisory, use the "
            "handoff_conversation tool to transfer them. "
            "If it does NOT match (e.g. mechanical parts, IT support, "
            "medical, legal), politely explain that this firm only handles "
            "financial matters and you cannot help with their request. "
            "Do NOT transfer callers whose needs don't match any available agent. "
            "Keep responses under 30 words — they will be spoken aloud. "
            "NEVER write handoff instructions as text — always use the tool. "
            "Your spoken responses must ONLY contain words meant for the caller. "
            "Never use markdown, bullet points, or special formatting."
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
            "When you first join a call after a transfer, introduce yourself "
            "briefly and warmly — the caller is waiting. "
            "Give concise, conversational financial advice. "
            "When a caller asks about insurance details, policy information, "
            "or coverage, use the delegate_task tool to delegate the lookup "
            "to agent-insurance. Tell the caller you're checking and keep "
            "chatting. When the result appears in your context, share it. "
            "If the caller wants to discuss a different topic or needs to "
            "be re-triaged, use the handoff_conversation tool to transfer "
            "them back. "
            "NEVER write handoff or delegation instructions as text — "
            "always use the appropriate tool. "
            "Your spoken responses must ONLY contain words meant for the caller. "
            "Speak naturally, under 50 words per response. "
            "No tables, no bullet points, no file references. "
            "Never use markdown or special formatting."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=20)),
    )

    # --- Background agent (insurance lookup) ---------------------------------
    insurance = Agent(
        "agent-insurance",
        provider=GeminiAIProvider(gemini_config),
        role="Insurance data specialist",
        description="Looks up client insurance policy details",
        scope="Insurance policy lookups only",
        system_prompt=(
            "You are a background insurance lookup agent. When given a client "
            "query, simulate looking up their insurance policy and return "
            "realistic fake data. Include: policy number, coverage type, "
            "monthly premium, deductible, and next renewal date. "
            "Keep it concise — under 50 words. No markdown or formatting."
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

    # --- Delegation (background agent tasks) ---------------------------------
    delegate_tool = build_delegate_tool(
        [
            ("agent-insurance", "Looks up client insurance policy details"),
        ]
    )
    delegate_handler = DelegateHandler(
        kit,
        delivery_strategy=WaitForIdleDelivery(
            prompt="A background task just completed. Share the result with the caller.",
        ),
    )
    setup_delegation(advisor, delegate_handler, tool=delegate_tool)

    # --- Hooks ---------------------------------------------------------------

    # Latency tracker: room_id → monotonic timestamp of last user transcription
    _last_transcription_ts: dict[str, float] = {}

    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC, name="log_speech")
    async def on_speech_start(event, ctx):
        logger.info("[VAD] speech_start — room=%s", ctx.room.id)

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC, name="log_speech_end")
    async def on_speech_end(event, ctx):
        logger.info("[VAD] speech_end — room=%s", ctx.room.id)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text: str, ctx: RoomContext) -> HookResult:
        _last_transcription_ts[ctx.room.id] = time.monotonic()
        logger.info("[STT] Caller: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx: RoomContext) -> HookResult:
        t0 = _last_transcription_ts.get(ctx.room.id)
        latency_msg = ""
        if t0:
            ai_ms = (time.monotonic() - t0) * 1000
            latency_msg = f" (AI latency: {ai_ms:.0f}ms since STT)"
        agent = get_conversation_state(ctx.room).active_agent_id or "?"
        logger.info("[TTS] start — agent=%s text=%.60s...%s", agent, text, latency_msg)
        return HookResult.allow()

    @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC, name="log_after_tts")
    async def after_tts(event, ctx):
        t0 = _last_transcription_ts.pop(ctx.room.id, None)
        e2e_msg = ""
        if t0:
            e2e_ms = (time.monotonic() - t0) * 1000
            e2e_msg = f" (e2e: {e2e_ms:.0f}ms since STT)"
        logger.info("[TTS] done — room=%s%s", ctx.room.id, e2e_msg)

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_task_delegated(event, ctx):
        logger.info(
            "Task delegated → agent=%s child_room=%s",
            event.metadata.get("agent_id"),
            event.metadata.get("child_room_id"),
        )

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_task_completed(event, ctx):
        logger.info(
            "Task completed → %s (%.0fms)",
            event.metadata.get("task_id"),
            event.metadata.get("duration_ms", 0),
        )

    @kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC, name="log_dtmf")
    async def on_dtmf(event, ctx):
        logger.info("[DTMF] digit=%s", event.digit)
        voice_channel = kit.get_channel("voice")
        if event.digit == "1":
            await voice_channel.play(event.session, "../../olivia.wav")
        else:
            await voice_channel.say(event.session, f"You pressed {event.digit}")

    # --- Incoming call handler -----------------------------------------------

    @sip.on_call
    async def handle_call(session):
        """Called when a SIP INVITE is accepted and RTP is active."""
        room_id = session.metadata.get("room_id")
        caller = session.metadata.get("caller", "unknown")
        logger.info("Incoming call — session=%s caller=%s", session.id, caller)

        result = await kit.process_inbound(
            parse_voice_session(session, channel_id="voice"),
            room_id=room_id,
        )
        if result.blocked:
            logger.warning("Call rejected: %s", result.reason)
            return

        actual_room_id = room_id or session.room_id

        # Attach BOTH agents — only one is active via ConversationRouter
        await kit.attach_channel(
            actual_room_id,
            "agent-triage",
            category=ChannelCategory.INTELLIGENCE,
        )
        await kit.attach_channel(
            actual_room_id,
            "agent-advisor",
            category=ChannelCategory.INTELLIGENCE,
        )

        # Initialize orchestration state to the pipeline's first stage
        room = await kit.get_room(actual_room_id)
        room = set_conversation_state(
            room,
            ConversationState(phase="intake", active_agent_id="agent-triage"),
        )
        await kit.store.update_room(room)

        # Trigger the triage agent's initial greeting
        await _handler.send_greeting(actual_room_id, channel_id="voice")

        logger.info(
            "Call connected — session=%s room=%s phase=intake agent=triage",
            session.id,
            actual_room_id,
        )

    # --- Disconnect handler --------------------------------------------------

    @sip.on_call_disconnected
    async def handle_disconnect(session):
        """Called when the remote party hangs up (BYE)."""
        room_id = session.metadata.get("room_id", session.id)
        room = await kit.get_room(room_id)
        state = get_conversation_state(room)
        logger.info(
            "Call ended — session=%s phase=%s handoffs=%d",
            session.id,
            state.phase,
            state.handoff_count,
        )
        await kit.close_room(room_id)

    # --- Start ---------------------------------------------------------------

    await sip.start()
    logger.info(
        "Voice triage ready — SIP %s:%d, RTP %d-%d",
        SIP_HOST,
        SIP_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )
    logger.info("AI: Gemini — STT: Deepgram — TTS: ElevenLabs — VAD: sherpa-onnx")
    logger.info("Pipeline: intake (triage) -> handling (advisor)")
    logger.info("Delegation: advisor -> agent-insurance (background lookup)")
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
