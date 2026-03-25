#!/usr/bin/env python3
"""RoomKit — SIP calls routed to ElevenLabs Conversational AI.

Incoming SIP calls from a PBX/trunk are answered automatically and bridged
to an ElevenLabs agent.  The caller talks to the AI in real time over the
phone — ElevenLabs handles STT, LLM, TTS, and turn detection server-side.

The PBX/proxy should set ``X-Room-ID`` and ``X-Session-ID`` SIP headers
before forwarding INVITEs to this server.

Requirements:
    pip install roomkit[sip,realtime-elevenlabs]

Usage:
    ELEVENLABS_API_KEY=... ELEVENLABS_AGENT_ID=... \
        python examples/realtime_voice_sip_elevenlabs.py

Environment variables:
    ELEVENLABS_API_KEY      (required) ElevenLabs API key
    ELEVENLABS_AGENT_ID     (required) Agent ID from the dashboard
    ELEVENLABS_VOICE_ID     Override the agent's default voice
    SYSTEM_PROMPT           Override the agent's default system prompt
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("sip_elevenlabs_example")

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.models.hook import HookResult
from roomkit.models.trace import ProtocolTrace
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.realtime.events import RealtimeTranscriptionEvent
from roomkit.voice.realtime.sip_transport import SIPRealtimeTransport

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = "0.0.0.0"  # nosec B104
SIP_PORT = 5060
RTP_IP = "0.0.0.0"  # nosec B104  — auto-detects per call; set your IP for production
RTP_PORT_START = 10000
RTP_PORT_END = 20000


async def main() -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not api_key or not agent_id:
        logger.error("Set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID to run this example.")
        return

    kit = RoomKit()

    # -- SIP backend (answers incoming calls) --
    sip = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
        user_agent="RoomKit/0.1",
        server_name="RoomKit",
    )

    # -- ElevenLabs Conversational AI provider --
    config = ElevenLabsRealtimeConfig(api_key=api_key, agent_id=agent_id)
    provider = ElevenLabsRealtimeProvider(config)

    # -- Bridge transport: SIP audio ↔ ElevenLabs audio --
    transport = SIPRealtimeTransport(sip)

    # -- Realtime voice channel --
    # ElevenLabs uses 16 kHz PCM; SIP is 8 kHz — resampling is automatic
    realtime = RealtimeVoiceChannel(
        "realtime-voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get("SYSTEM_PROMPT"),
        voice=os.environ.get("ELEVENLABS_VOICE_ID"),
        input_sample_rate=16000,
        output_sample_rate=16000,
    )
    kit.register_channel(realtime)

    # -------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def gate_incoming(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Log incoming calls."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            caller = event.content.data.get("caller")
            logger.info(
                "BEFORE_BROADCAST — session_started from %s in room %s",
                caller,
                ctx.room.id,
            )
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: RealtimeTranscriptionEvent, ctx: RoomContext) -> HookResult:
        """Log every transcription — both user speech and AI responses."""
        tag = "USER" if event.role == "user" else "AI"
        final = "final" if event.is_final else "interim"
        logger.info("ON_TRANSCRIPTION [%s] (%s): %s", tag, final, event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
    async def on_protocol_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
        """Log protocol-level traces (SIP INVITE, BYE, etc.)."""
        logger.info(
            "ON_PROTOCOL_TRACE [room=%s] %s %s: %s",
            ctx.room.id,
            trace.direction,
            trace.protocol,
            trace.summary,
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def log_events(event: RoomEvent, ctx: RoomContext) -> None:
        """Log every event that passes through the room."""
        logger.info(
            "AFTER_BROADCAST — type=%s channel=%s room=%s content=%s",
            event.type,
            event.source.channel_id,
            ctx.room.id,
            type(event.content).__name__,
        )

    # -------------------------------------------------------------------
    # Incoming call → process through framework (room routing + hooks)
    # -------------------------------------------------------------------

    @sip.on_call
    async def handle_call(session):
        room_id = session.metadata.get("room_id", session.id)
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "realtime-voice")
        await kit.join(
            room_id,
            "realtime-voice",
            participant_id=session.participant_id or session.id,
            connection=session,
        )
        logger.info("SIP call connected — session=%s room=%s", session.id, room_id)

    # -------------------------------------------------------------------
    # Remote hangup → end realtime session + close room
    # -------------------------------------------------------------------

    @sip.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("SIP call ended — session=%s", session.id)
        await kit.leave(session)

    # -------------------------------------------------------------------
    # Start
    # -------------------------------------------------------------------

    await sip.start()
    logger.info(
        "SIP + ElevenLabs ready — SIP %s:%d, RTP %d-%d",
        SIP_HOST,
        SIP_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )
    logger.info("Waiting for incoming SIP calls...")

    try:
        await asyncio.Event().wait()
    finally:
        await sip.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
