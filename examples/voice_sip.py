#!/usr/bin/env python3
"""SIP voice backend example.

Demonstrates how to use the SIPVoiceBackend to accept incoming SIP calls
from a PBX/SIP trunk, route them to rooms using X-headers, and wire up
the full voice AI pipeline (STT → AI → TTS).

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

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_sip_example")

from roomkit import RoomKit, VoiceChannel
from roomkit.voice import VoiceSession
from roomkit.voice.backends.sip import SIPVoiceBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_HOST = "0.0.0.0"
SIP_PORT = 5060
RTP_IP = "0.0.0.0"  # Use your actual IP for real deployments
RTP_PORT_START = 10000
RTP_PORT_END = 20000


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
    VoiceChannel("voice", backend=backend)

    # -----------------------------------------------------------------------
    # Incoming call handler
    # -----------------------------------------------------------------------

    def on_call(session: VoiceSession) -> None:
        """Called when a SIP INVITE is accepted and RTP is active."""
        room_id = session.metadata.get("room_id", session.id)
        logger.info("Incoming call — session=%s, room=%s", session.id, room_id)
        logger.info("  Caller: %s", session.metadata.get("caller"))
        logger.info("  X-Headers: %s", session.metadata.get("x_headers"))

        # In a real application you would:
        # 1. Create or find the room
        # 2. Attach voice + AI channels
        # 3. Bind the session to the room
        #
        # Example:
        #   asyncio.get_running_loop().create_task(_setup_room(kit, session, room_id))

    # -----------------------------------------------------------------------
    # Disconnect handler
    # -----------------------------------------------------------------------

    def on_disconnect(session: VoiceSession) -> None:
        """Called when the remote party hangs up (BYE)."""
        logger.info("Call ended — session=%s", session.id)

        # In a real application you would:
        # 1. Unbind the session from the voice channel
        # 2. Close the room if no more participants
        #
        # Example:
        #   asyncio.get_running_loop().create_task(_teardown_room(kit, session))

    backend.on_call(on_call)
    backend.on_call_disconnected(on_disconnect)

    # -----------------------------------------------------------------------
    # Audio callback (for debugging / pipeline integration)
    # -----------------------------------------------------------------------

    def on_audio(session: VoiceSession, frame: object) -> None:
        """Called for every inbound audio frame (20ms PCM)."""
        # In a real application the AudioPipeline handles this via
        # VoiceChannel, but you can also tap into raw frames here.
        pass

    backend.on_audio_received(on_audio)

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
