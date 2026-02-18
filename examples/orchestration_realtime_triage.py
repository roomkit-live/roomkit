#!/usr/bin/env python3
"""Speech-to-speech orchestration: SIP call with Gemini Live multi-agent handoff.

Same triage -> advisor pipeline as orchestration_voice_triage.py, but using
Gemini Live for speech-to-speech AI instead of the STT -> LLM -> TTS chain.
Audio flows directly between the caller and Gemini; on agent handoff the
Gemini session is reconfigured (system prompt, voice, tools) using session
resumption so the conversation context is preserved.

Architecture:

    SIP Call -> SIPRealtimeTransport -> RealtimeVoiceChannel
                                             |
                                      GeminiLiveProvider
                                             |
                                  handoff_conversation tool call
                                             |
                                      _wire_realtime() tool handler
                                             |
                                  reconfigure_session() (voice, prompt, tools)
                                             |
                                  Gemini reconnects with session resumption
                                  (~200-500 ms, conversation preserved)

Requirements:
    pip install roomkit[sip,realtime-gemini]

Usage:
    GOOGLE_API_KEY=... python examples/orchestration_realtime_triage.py

    # From a SIP client or PBX, send an INVITE to port 5060

Environment variables:
    GOOGLE_API_KEY      (required) Google API key
    GEMINI_MODEL        Gemini model (default: gemini-2.5-flash-native-audio-preview-12-2025)
    SIP_HOST            Listening address (default: 0.0.0.0)
    SIP_PORT            SIP port (default: 5060)
    RTP_IP              RTP bind address (default: 0.0.0.0)
    RTP_PORT_START      RTP port range start (default: 10000)
    RTP_PORT_END        RTP port range end (default: 20000)
    VOICE_TRIAGE        Gemini voice for triage (default: Zephyr, female)
    VOICE_ADVISOR       Gemini voice for advisor (default: Fenrir, male)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("realtime_triage")

# Suppress chain-depth warnings (expected in multi-agent setups)
logging.getLogger("roomkit.core.event_router").setLevel(logging.ERROR)

from roomkit import (
    Agent,
    ConversationPipeline,
    ConversationState,
    PipelineStage,
    RealtimeVoiceChannel,
    RoomKit,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.hook import HookResult
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice import parse_voice_session
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.realtime.events import RealtimeTranscriptionEvent
from roomkit.voice.realtime.sip_transport import SIPRealtimeTransport

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = os.environ.get("SIP_HOST", "0.0.0.0")  # nosec B104
SIP_PORT = int(os.environ.get("SIP_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")  # nosec B104
RTP_PORT_START = int(os.environ.get("RTP_PORT_START", "10000"))
RTP_PORT_END = int(os.environ.get("RTP_PORT_END", "20000"))

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")

VOICE_TRIAGE = os.environ.get("VOICE_TRIAGE", "Zephyr")
VOICE_ADVISOR = os.environ.get("VOICE_ADVISOR", "Fenrir")

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
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Missing GOOGLE_API_KEY environment variable.")
        print("See docstring at the top of this file for setup instructions.")
        sys.exit(1)


async def main() -> None:
    check_env()

    kit = RoomKit()

    # --- SIP backend + realtime transport -----------------------------------
    sip = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
    )
    transport = SIPRealtimeTransport(sip)

    # --- Gemini Live provider -----------------------------------------------
    provider = GeminiLiveProvider(
        api_key=os.environ["GOOGLE_API_KEY"],
        model=GEMINI_MODEL,
    )

    # --- RealtimeVoiceChannel -----------------------------------------------
    # system_prompt, voice, and tools will be set by pipeline.install()
    rtv = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
    )
    kit.register_channel(rtv)

    # --- Config-only agents (no AI provider — Gemini handles reasoning) -----
    triage = Agent(
        "agent-triage",
        role="Triage receptionist",
        description="Routes callers to the right financial specialist",
        scope="Financial advisory services only",
        voice=VOICE_TRIAGE,
        system_prompt=(
            "ALWAYS reply in the same language as the caller. "
            "Greet callers warmly and identify their need. "
            "If the caller's request matches financial advisory, use the "
            "handoff_conversation tool to transfer them. "
            "If it does NOT match (e.g. mechanical parts, IT support, "
            "medical, legal), politely explain that this firm only handles "
            "financial matters and you cannot help with their request. "
            "Do NOT transfer callers whose needs don't match any available agent. "
            "Keep responses under 30 words. "
            "NEVER write handoff instructions as text — always use the tool."
        ),
    )

    advisor = Agent(
        "agent-advisor",
        role="Portfolio advisor",
        description="Provides financial and investment advice",
        voice=VOICE_ADVISOR,
        system_prompt=(
            "ALWAYS reply in the same language as the caller. "
            "Give concise, conversational financial advice. "
            "If the caller wants to discuss a different topic or needs to "
            "be re-triaged, use the handoff_conversation tool to transfer "
            "them back. "
            "Speak naturally, under 50 words per response."
        ),
    )

    # --- Orchestration (speech-to-speech mode) ------------------------------
    # install() detects RealtimeVoiceChannel and auto-wires:
    #   - system_prompt + identity block per agent
    #   - voice per agent
    #   - enum-constrained handoff tool per agent
    #   - tool handler that intercepts handoff_conversation
    #   - session reconfiguration on handoff (Gemini session resumption)
    _router, _handler = pipeline.install(
        kit,
        [triage, advisor],
        greet_on_handoff=True,
        voice_channel_id="voice",
    )

    # --- Hooks ---------------------------------------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(
        event: RealtimeTranscriptionEvent,
        ctx: RoomContext,
    ) -> HookResult:
        logger.info("[%s] %s", event.role, event.text)
        return HookResult.allow()

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

        # Initialize orchestration state to the pipeline's first stage
        room = await kit.get_room(actual_room_id)
        room = set_conversation_state(
            room,
            ConversationState(phase="intake", active_agent_id="agent-triage"),
        )
        await kit.store.update_room(room)

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
        "Realtime triage ready — SIP %s:%d, RTP %d-%d",
        SIP_HOST,
        SIP_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )
    logger.info("AI: Gemini Live (%s) — speech-to-speech", GEMINI_MODEL)
    logger.info("Pipeline: intake (triage) -> handling (advisor)")
    logger.info("Voices: triage=%s, advisor=%s", VOICE_TRIAGE, VOICE_ADVISOR)
    logger.info("Waiting for incoming SIP calls...")

    try:
        await asyncio.Event().wait()
    finally:
        await sip.close()
        await provider.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
