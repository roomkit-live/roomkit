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
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_dial")

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.hook import HookResult
from roomkit.models.trace import ProtocolTrace
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.backends.sip import PT_G722, PT_PCMA, PT_PCMU, SIPVoiceBackend
from roomkit.voice.realtime.events import RealtimeTranscriptionEvent
from roomkit.voice.realtime.sip_transport import SIPRealtimeTransport

# ---------------------------------------------------------------------------
# Debug (set SIP_DEBUG=1 to enable verbose SIP/RTP/SDP tracing)
# ---------------------------------------------------------------------------

if os.environ.get("SIP_DEBUG", "0") in ("1", "true", "yes"):
    logging.getLogger("roomkit.voice.sip").setLevel(logging.DEBUG)
    logging.getLogger("aiosipua").setLevel(logging.DEBUG)
    logging.getLogger("aiortp").setLevel(logging.DEBUG)

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

GEMINI_MODEL = "gemini-3.1-flash-live"
SYSTEM_PROMPT = (
    "You are a friendly phone assistant making an outbound call. "
    "When the person answers, greet them in one short sentence (no more than "
    "ten words) and then wait for them to reply. Do not monologue. "
    "Keep every reply to one or two short sentences. "
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


async def handle_tool_call(name: str, arguments: dict) -> str:
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
    env = require_env("GOOGLE_API_KEY")

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # -- SIP backend --
    # send_silence_on_answer=0.5 primes outbound RTP with 500 ms of silence
    # right after the call is answered. Needed for PSTN trunks that wait for
    # our RTP before sending their own (symmetric-RTP learning / NAT latching).
    # jitter_prefetch=3 absorbs ~60 ms of carrier jitter before playout —
    # reduces audible glitches on PSTN at the cost of +60 ms latency.
    # outbound_silence_fill=True keeps the RTP stream continuous at 50 pps
    # during gaps between Gemini TTS chunks. Without it the pacer stalls,
    # producing choppy audio for PSTN callees (no packet-loss concealment).
    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", 5070),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        rtp_port_start=10000,
        rtp_port_end=10010,
        send_silence_on_answer=0.5,
        jitter_prefetch=3,
        outbound_silence_fill=True,
    )

    # -- Gemini Live provider --
    gemini = GeminiLiveProvider(api_key=env["GOOGLE_API_KEY"], model=GEMINI_MODEL)

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
    # Debug: SIP/SDP protocol trace + first-RTP-packet tap + remote addr
    # -------------------------------------------------------------------

    def _trace(event: ProtocolTrace) -> None:
        logger.info(
            "[SIP TRACE] %s %s %s  meta=%s",
            event.direction.upper(),
            event.protocol,
            event.summary,
            {k: v for k, v in event.metadata.items() if k != "x_headers"},
        )
        if event.raw:
            raw = event.raw.decode(errors="replace") if isinstance(event.raw, bytes) else event.raw
            logger.info("[SIP TRACE RAW]\n%s", raw)

    realtime.on_trace(_trace, protocols=["sip"])

    _first_rtp_seen: dict[str, bool] = {}

    @backend.on_session_ready
    def _on_ready(session):  # type: ignore[no-untyped-def]
        state = backend._session_states.get(session.id)  # noqa: SLF001
        cs = state.call_session if state is not None else None
        remote = cs.remote_addr if cs is not None else None
        logger.info(
            "[SIP READY] session=%s codec_rate=%s clock_rate=%s local_rtp_port=%s remote_rtp=%s",
            session.id[:8],
            getattr(state, "codec_rate", None),
            getattr(state, "clock_rate", None),
            getattr(state, "rtp_port", None),
            remote,
        )

    def _first_packet_tap(session, frame):  # type: ignore[no-untyped-def]
        sid = session.id
        if not _first_rtp_seen.get(sid):
            _first_rtp_seen[sid] = True
            logger.info(
                "[RTP FIRST INBOUND] session=%s bytes=%d sample_rate=%s",
                sid[:8],
                len(frame.data),
                frame.sample_rate,
            )

    _prev_audio_cb = backend._audio_received_callback  # noqa: SLF001

    def _audio_chain(session, frame):  # type: ignore[no-untyped-def]
        _first_packet_tap(session, frame)
        if _prev_audio_cb is not None:
            _prev_audio_cb(session, frame)

    backend._audio_received_callback = _audio_chain  # noqa: SLF001

    # -------------------------------------------------------------------
    # Room setup — create once at startup
    # -------------------------------------------------------------------

    room_id = "sip-dial"
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "realtime-voice")

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
        await kit.join(
            room_id,
            "realtime-voice",
            participant_id=session.participant_id or session.id,
            connection=session,
        )

        # Outbound-dial greeting: trigger the AI to speak first.
        # start_audio_stream=True opens the realtime audio path before
        # the text goes in — the provider then responds as if someone
        # just answered, producing its greeting per the system prompt.
        rt_sessions = realtime.get_room_sessions(room_id)
        if rt_sessions:
            rt_session = rt_sessions[0]
            await realtime.inject_text(rt_session, "Allô ?", role="user", start_audio_stream=True)

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("Call ended — session=%s", session.id)
        await kit.leave(session)

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
    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
