#!/usr/bin/env python3
"""SIP voice backend example.

Demonstrates how to use the SIPVoiceBackend to accept incoming SIP calls
from a PBX/SIP trunk, route them to rooms using X-headers, and wire up
the full voice AI pipeline (STT -> AI -> TTS).

Shows that voice sessions flow through the exact same hooks as text
messages — BEFORE_BROADCAST can block a call just like it blocks a
text message.

The PBX/proxy (Kamailio, OpenSIPS, Asterisk) sets X-Room-ID and
X-Session-ID headers before forwarding INVITEs to this server.

Requirements:
    pip install roomkit[sip]

Usage:
    python examples/voice_sip.py

    # From a SIP client or PBX, send an INVITE to port 5060 with:
    #   X-Room-ID: my-room
    #   X-Session-ID: caller-123
"""

import asyncio
import logging
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_sip_example")

from roomkit import RoomKit, VoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.models.hook import HookResult
from roomkit.models.trace import ProtocolTrace
from roomkit.voice import parse_voice_session
from roomkit.voice.backends.sip import SIPVoiceBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = "0.0.0.0"
SIP_PORT = 5060
RTP_IP = "0.0.0.0"  # Use your actual IP for real deployments
RTP_PORT_START = 10000
RTP_PORT_END = 20000

# Simple in-memory call log (demonstrates AFTER_BROADCAST observability)
call_log: list[dict] = []


async def main() -> None:
    kit = RoomKit()

    # Create the SIP backend
    backend = SIPVoiceBackend(
        local_sip_addr=(SIP_HOST, SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
    )

    # Create voice channel with backend (no STT/TTS in this example)
    voice = VoiceChannel("voice", backend=backend)
    kit.register_channel(voice)

    # -------------------------------------------------------------------
    # Protocol trace — channel-level (all SIP traces, no room needed)
    # -------------------------------------------------------------------

    voice.on_trace(
        lambda t: logger.info("[TRACE] %s %s: %s", t.direction, t.protocol, t.summary),
        protocols=["sip"],
    )

    # -------------------------------------------------------------------
    # Hooks — identical to what you'd write for text channels
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def log_and_gate(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Log every inbound event. Block outside business hours as an example."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            caller = event.content.data.get("caller", "unknown")
            hour = datetime.now(UTC).hour

            logger.info(
                "BEFORE_BROADCAST — new voice session from %s (hour=%d UTC)",
                caller,
                hour,
            )

            # Example: reject calls outside 8:00-20:00 UTC
            if not (8 <= hour < 20):
                logger.warning("Rejecting call outside business hours (hour=%d)", hour)
                return HookResult.block("outside_business_hours")

        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text: str, ctx: RoomContext) -> HookResult:
        """Log what the caller says. Can also modify or block the text."""
        logger.info("ON_TRANSCRIPTION: %s", text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
    async def on_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
        """Room-level protocol trace — only traces for channels in the room."""
        logger.info(
            "ON_PROTOCOL_TRACE [room=%s] %s %s: %s",
            ctx.room.id,
            trace.direction,
            trace.protocol,
            trace.summary,
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def track_calls(event: RoomEvent, ctx: RoomContext) -> None:
        """Record every voice session that passes through — observability hook."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            entry = {
                "session_id": event.content.data.get("session_id"),
                "caller": event.content.data.get("caller"),
                "room_id": ctx.room.id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            call_log.append(entry)
            logger.info(
                "AFTER_BROADCAST — call logged: %s (provider=%s)",
                entry,
                event.source.provider,
            )

    # -----------------------------------------------------------------------
    # Incoming call → process through framework
    # -----------------------------------------------------------------------

    @backend.on_call
    async def handle_call(session):
        """Called when a SIP INVITE is accepted and RTP is active."""
        caller = session.metadata.get("caller")
        logger.info("Incoming call — session=%s caller=%s", session.id, caller)

        result = await kit.process_inbound(
            parse_voice_session(session, channel_id="voice"),
            room_id=session.metadata.get("room_id"),
        )
        if result.blocked:
            logger.warning("Call rejected: %s", result.reason)

    # -----------------------------------------------------------------------
    # Disconnect handler
    # -----------------------------------------------------------------------

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        """Called when the remote party hangs up (BYE)."""
        logger.info("Call ended — session=%s", session.id)
        room_id = session.metadata.get("room_id", session.id)
        await kit.close_room(room_id)

    # -----------------------------------------------------------------------
    # Start
    # -----------------------------------------------------------------------

    await backend.start()
    logger.info(
        "SIP voice backend ready — listening on %s:%d, RTP ports %d-%d",
        SIP_HOST,
        SIP_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )

    # Run forever
    try:
        await asyncio.Event().wait()
    finally:
        await backend.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
