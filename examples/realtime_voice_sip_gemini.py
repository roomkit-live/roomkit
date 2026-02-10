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

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("sip_gemini_example")

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.base import VoiceSession
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


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")
        return

    kit = RoomKit()

    # -- SIP backend (answers incoming calls) --
    sip_backend = SIPVoiceBackend(
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
    transport = SIPRealtimeTransport(sip_backend)

    # -- Realtime voice channel --
    # The SIP transport auto-detects the codec sample rate per call
    # (16 kHz for G.722, 8 kHz for G.711) and resamples automatically.
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
    # Incoming call → create room + start realtime session
    # -------------------------------------------------------------------

    async def handle_call(voice_session: VoiceSession) -> None:
        room_id = voice_session.metadata.get("room_id", voice_session.id)
        logger.info(
            "Incoming SIP call — session=%s, room=%s, caller=%s",
            voice_session.id,
            room_id,
            voice_session.metadata.get("caller"),
        )

        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "realtime-voice")

        # Start the realtime session, passing the SIP VoiceSession as
        # the "connection" — SIPRealtimeTransport knows how to bridge it.
        rt_session = await realtime.start_session(
            room_id,
            voice_session.participant_id,
            connection=voice_session,
        )
        logger.info(
            "Realtime session %s started for SIP call %s",
            rt_session.id,
            voice_session.id,
        )

    def on_call(voice_session: VoiceSession) -> None:
        asyncio.get_running_loop().create_task(handle_call(voice_session))

    sip_backend.on_call(on_call)

    # -------------------------------------------------------------------
    # Remote hangup → end realtime session + close room
    # -------------------------------------------------------------------

    async def handle_disconnect(voice_session: VoiceSession) -> None:
        logger.info("SIP call ended — session=%s", voice_session.id)
        room_id = voice_session.metadata.get("room_id", voice_session.id)

        # End all realtime sessions in the room
        for rt_session in realtime._get_room_sessions(room_id):
            await realtime.end_session(rt_session)

        await kit.close_room(room_id)

    def on_disconnect(voice_session: VoiceSession) -> None:
        asyncio.get_running_loop().create_task(handle_disconnect(voice_session))

    sip_backend.on_call_disconnected(on_disconnect)

    # -------------------------------------------------------------------
    # Start
    # -------------------------------------------------------------------

    await sip_backend.start()
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
        await sip_backend.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
