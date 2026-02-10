#!/usr/bin/env python3
"""RoomKit — SIP calls routed to Gemini Live for speech-to-speech AI.

Incoming SIP calls from a PBX/trunk are answered automatically and bridged
to Google Gemini Live.  The caller talks to the AI in real time over the
phone — audio is resampled transparently between 8 kHz (telephony) and
16/24 kHz (Gemini).

The PBX/proxy should set ``X-Room-ID`` and ``X-Session-ID`` SIP headers
before forwarding INVITEs to this server.

Requirements:
    pip install roomkit[sip,realtime-gemini]

Usage:
    GOOGLE_API_KEY=... python examples/realtime_voice_sip_gemini.py
"""

import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("sip_gemini_example")

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.models.hook import HookResult
from roomkit.models.trace import ProtocolTrace
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice import parse_voice_session
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

GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
SYSTEM_PROMPT = "You are a friendly phone assistant. Be concise and helpful."
VOICE = "Aoede"

# Set of blocked caller IDs (example: block known spam numbers)
BLOCKED_CALLERS = {"+15550000000"}


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")
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

    # -- Gemini Live provider --
    gemini = GeminiLiveProvider(api_key=api_key, model=GEMINI_MODEL)

    # -- Bridge transport: SIP audio ↔ Gemini audio --
    transport = SIPRealtimeTransport(sip)

    # -- Realtime voice channel --
    realtime = RealtimeVoiceChannel(
        "realtime-voice",
        provider=gemini,
        transport=transport,
        system_prompt=SYSTEM_PROMPT,
        voice=VOICE,
        input_sample_rate=16000,
        output_sample_rate=24000,
    )
    kit.register_channel(realtime)

    # -------------------------------------------------------------------
    # Hooks — same hooks work for text AND voice, that's the point
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def gate_incoming(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Block calls from spam numbers. Works identically to blocking
        a text message — same hook, same interface."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            caller = event.content.data.get("caller")
            if caller in BLOCKED_CALLERS:
                logger.warning("Blocked call from %s", caller)
                return HookResult.block(f"caller_blocked:{caller}")
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
        logger.info(
            "ON_TRANSCRIPTION [%s] (%s): %s",
            tag,
            final,
            event.text,
        )
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
        """Log every event that passes through the room — voice or text."""
        logger.info(
            "AFTER_BROADCAST — type=%s channel=%s room=%s content=%s provider=%s",
            event.type,
            event.source.channel_id,
            ctx.room.id,
            type(event.content).__name__,
            event.source.provider,
        )

    # -------------------------------------------------------------------
    # Incoming call → process through framework (room routing + hooks)
    # -------------------------------------------------------------------

    @sip.on_call
    async def handle_call(session):
        result = await kit.process_inbound(
            parse_voice_session(session, channel_id="realtime-voice"),
            room_id=session.metadata.get("room_id"),
        )
        if result.blocked:
            logger.warning("Call rejected by hooks: %s", result.reason)
        else:
            logger.info("SIP call connected — session=%s", session.id)

    # -------------------------------------------------------------------
    # Remote hangup → end realtime session + close room
    # -------------------------------------------------------------------

    @sip.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("SIP call ended — session=%s", session.id)
        room_id = session.metadata.get("room_id", session.id)
        for rt_session in realtime._get_room_sessions(room_id):
            await realtime.end_session(rt_session)
        await kit.close_room(room_id)

    # -------------------------------------------------------------------
    # Start
    # -------------------------------------------------------------------

    await sip.start()
    logger.info(
        "SIP + Gemini Live ready — SIP %s:%d, RTP %d-%d",
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
