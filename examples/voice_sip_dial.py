#!/usr/bin/env python3
"""Outbound SIP call with Gemini Live speech-to-speech AI.

Demonstrates how to use SIPVoiceBackend.dial() to initiate an outbound
SIP call through a proxy, with optional digest authentication.  Once
the call is answered, Gemini Live handles the conversation in real time
— the remote party talks to the AI over the phone.

Audio is resampled transparently between telephony rates (8/16 kHz)
and Gemini's native rates (16 kHz input, 24 kHz output).

Requirements:
    pip install roomkit[sip,realtime-gemini]

Usage:
    GOOGLE_API_KEY=... python examples/voice_sip_dial.py

    # Environment variables (all optional unless noted):
    #   GOOGLE_API_KEY  — Google AI API key (REQUIRED)
    #   SIP_PROXY_HOST  — SIP proxy/PBX IP   (default: 127.0.0.1)
    #   SIP_PROXY_PORT  — SIP proxy port      (default: 5060)
    #   SIP_FROM_URI    — caller SIP URI       (default: sip:bot@example.com)
    #   SIP_TO_URI      — callee SIP URI       (default: sip:alice@example.com)
    #   SIP_CODEC       — audio codec: pcmu, pcma, g722 (default: pcmu)
    #   SIP_AUTH_USER   — digest auth username (optional)
    #   SIP_AUTH_PASS   — digest auth password (optional)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_sip_dial")

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.hook import HookResult
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice import parse_voice_session
from roomkit.voice.backends.sip import PT_G722, PT_PCMA, PT_PCMU, SIPVoiceBackend
from roomkit.voice.realtime.events import RealtimeTranscriptionEvent
from roomkit.voice.realtime.sip_transport import SIPRealtimeTransport

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

SIP_PROXY_HOST = os.environ.get("SIP_PROXY_HOST", "127.0.0.1")
SIP_PROXY_PORT = int(os.environ.get("SIP_PROXY_PORT", "5060"))
FROM_URI = os.environ.get("SIP_FROM_URI", "sip:bot@example.com")
TO_URI = os.environ.get("SIP_TO_URI", "sip:alice@example.com")
CODEC = {"pcmu": PT_PCMU, "pcma": PT_PCMA, "g722": PT_G722}.get(
    os.environ.get("SIP_CODEC", "pcmu").lower(), PT_PCMU
)
AUTH_USER = os.environ.get("SIP_AUTH_USER", "")
AUTH_PASS = os.environ.get("SIP_AUTH_PASS", "")

GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
SYSTEM_PROMPT = (
    "You are a friendly phone assistant making an outbound call. "
    "Greet the person who picks up and be concise and helpful. "
    "You have access to a tool that returns the current date and time."
)
VOICE = "Aoede"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time in ISO 8601 format.",
        "parameters": {"type": "object", "properties": {}},
    },
]


async def handle_tool_call(session, name: str, arguments: dict) -> str:
    if name == "get_current_datetime":
        now = datetime.now(UTC).astimezone()
        return json.dumps(
            {
                "datetime": now.isoformat(),
                "date": now.strftime("%A, %B %d, %Y"),
                "time": now.strftime("%I:%M %p"),
            }
        )
    return json.dumps({"error": f"Unknown tool: {name}"})


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")  # noqa: T201
        return

    kit = RoomKit()

    # -- SIP backend --
    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", 5070),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        rtp_port_start=10000,
        rtp_port_end=20000,
    )

    # -- Gemini Live provider --
    gemini = GeminiLiveProvider(api_key=api_key, model=GEMINI_MODEL)

    # -- Bridge: SIP audio <-> Gemini audio --
    transport = SIPRealtimeTransport(backend)

    realtime = RealtimeVoiceChannel(
        "realtime-voice",
        provider=gemini,
        transport=transport,
        system_prompt=SYSTEM_PROMPT,
        voice=VOICE,
        tools=TOOLS,
        tool_handler=handle_tool_call,
        input_sample_rate=16000,
        output_sample_rate=24000,
    )
    kit.register_channel(realtime)

    # -------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(
        event: RealtimeTranscriptionEvent,
        ctx: RoomContext,
    ) -> HookResult:
        tag = "USER" if event.role == "user" else "AI"
        final = "final" if event.is_final else "interim"
        logger.info("[%s] (%s): %s", tag, final, event.text)
        return HookResult.allow()

    # -------------------------------------------------------------------
    # Call handlers
    # -------------------------------------------------------------------

    @backend.on_call
    async def handle_call(session):
        callee = session.metadata.get("callee")
        logger.info("Call active — session=%s callee=%s", session.id, callee)
        await kit.process_inbound(
            parse_voice_session(session, channel_id="realtime-voice"),
            room_id=session.room_id,
        )

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("Call ended — session=%s", session.id)
        room_id = session.metadata.get("room_id", session.id)
        for rt_session in realtime._get_room_sessions(room_id):
            await realtime.end_session(rt_session)
        await kit.close_room(room_id)

    # -------------------------------------------------------------------
    # Start backend and dial
    # -------------------------------------------------------------------

    await backend.start()

    # Build optional digest auth
    auth = None
    if AUTH_USER:
        from aiosipua import SipDigestAuth

        auth = SipDigestAuth(username=AUTH_USER, password=AUTH_PASS)

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
            codec=CODEC,
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
