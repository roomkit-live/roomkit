#!/usr/bin/env python3
"""Outbound SIP call example.

Demonstrates how to use SIPVoiceBackend.dial() to initiate an outbound
SIP call through a proxy, with optional digest authentication.  Once
the call is answered, the full voice AI pipeline is active — the remote
party's speech is transcribed and an AI agent responds via TTS.

Requirements:
    pip install roomkit[sip]

Usage:
    python examples/voice_sip_dial.py

    # Environment variables (all optional):
    #   SIP_PROXY_HOST  — SIP proxy/PBX IP   (default: 127.0.0.1)
    #   SIP_PROXY_PORT  — SIP proxy port      (default: 5060)
    #   SIP_FROM_URI    — caller SIP URI       (default: sip:bot@example.com)
    #   SIP_TO_URI      — callee SIP URI       (default: sip:alice@example.com)
    #   SIP_AUTH_USER   — digest auth username (optional)
    #   SIP_AUTH_PASS   — digest auth password (optional)
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_sip_dial")

from roomkit import RoomKit, VoiceChannel
from roomkit.voice import parse_voice_session
from roomkit.voice.backends.sip import SIPVoiceBackend

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

SIP_PROXY_HOST = os.environ.get("SIP_PROXY_HOST", "127.0.0.1")
SIP_PROXY_PORT = int(os.environ.get("SIP_PROXY_PORT", "5060"))
FROM_URI = os.environ.get("SIP_FROM_URI", "sip:bot@example.com")
TO_URI = os.environ.get("SIP_TO_URI", "sip:alice@example.com")
AUTH_USER = os.environ.get("SIP_AUTH_USER", "")
AUTH_PASS = os.environ.get("SIP_AUTH_PASS", "")


async def main() -> None:
    kit = RoomKit()

    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", 5070),  # local SIP port for this UA
        local_rtp_ip="0.0.0.0",
        rtp_port_start=10000,
        rtp_port_end=20000,
    )

    voice = VoiceChannel("voice", backend=backend)
    kit.register_channel(voice)

    # Optional: log SIP protocol traces
    voice.on_trace(
        lambda t: logger.info("[TRACE] %s %s: %s", t.direction, t.protocol, t.summary),
        protocols=["sip"],
    )

    # Route incoming audio (from the answered call) through the framework
    @backend.on_call
    async def handle_call(session):
        callee = session.metadata.get("callee")
        logger.info("Call active — session=%s callee=%s", session.id, callee)
        await kit.process_inbound(
            parse_voice_session(session, channel_id="voice"),
            room_id=session.room_id,
        )

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("Call ended — session=%s", session.id)
        await kit.close_room(session.room_id)

    # Start the backend (opens SIP transport)
    await backend.start()

    # Build optional digest auth
    auth = None
    if AUTH_USER:
        from aiosipua import SipDigestAuth

        auth = SipDigestAuth(username=AUTH_USER, password=AUTH_PASS)

    # Initiate the outbound call
    logger.info(
        "Dialing %s from %s via %s:%d ...",
        TO_URI,
        FROM_URI,
        SIP_PROXY_HOST,
        SIP_PROXY_PORT,
    )
    try:
        session = await backend.dial(
            to_uri=TO_URI,
            from_uri=FROM_URI,
            proxy_addr=(SIP_PROXY_HOST, SIP_PROXY_PORT),
            auth=auth,
            timeout=30.0,
        )
        logger.info("Call answered! session=%s", session.id)
    except TimeoutError:
        logger.error("Call timed out — no answer within 30s")
        await backend.close()
        return
    except RuntimeError as exc:
        logger.error("Call rejected: %s", exc)
        await backend.close()
        return

    # Keep running until the call ends
    try:
        await asyncio.Event().wait()
    finally:
        await backend.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
