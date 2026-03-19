#!/usr/bin/env python3
"""Bridge two SIP calls with live transcription.

Demonstrates VoiceChannel with ``bridge=True`` and STT: audio from
caller A is forwarded directly to caller B (and vice versa) while
Deepgram transcribes both sides in real time.

The example accepts two incoming SIP calls (INVITEs) and bridges them
in the same room.  Audio flows directly between the two participants
at native quality — no AI roundtrip.  Meanwhile, each caller's speech
is transcribed and logged via the ``ON_TRANSCRIPTION`` hook.

Architecture::

    Caller A ──mic──► Pipeline ──► STT (Deepgram streaming)
                          │                  │
                          └──bridge──► Caller B   └──► ON_TRANSCRIPTION hook
                                                       (live transcript log)

Requirements:
    pip install roomkit[sip]

Run with:
    DEEPGRAM_API_KEY=... uv run python examples/voice_sip_bridge.py

Environment variables:
    DEEPGRAM_API_KEY — Deepgram API key (required for STT)
    STT_LANGUAGE     — Language code for STT (default: en)
    SIP_LISTEN_ADDR  — SIP listen IP     (default: 0.0.0.0)
    SIP_LISTEN_PORT  — SIP listen port   (default: 5060)
    RTP_IP           — RTP bind IP       (default: 0.0.0.0)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_sip_bridge")

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_LISTEN_ADDR = os.environ.get("SIP_LISTEN_ADDR", "0.0.0.0")
SIP_LISTEN_PORT = int(os.environ.get("SIP_LISTEN_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")

ROOM_ID = "bridge-room"


async def main() -> None:
    if not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY is required. Set it and re-run.")
        return

    kit = RoomKit()

    # --- SIP backend ----------------------------------------------------------
    backend = SIPVoiceBackend(
        local_sip_addr=(SIP_LISTEN_ADDR, SIP_LISTEN_PORT),  # nosec B104
        local_rtp_ip=RTP_IP,  # nosec B104
    )

    # --- STT: Deepgram streaming for live transcription -----------------------
    # No local VAD — Deepgram handles endpointing server-side.  This gives
    # continuous streaming transcription instead of batch (collect-then-send).
    stt = DeepgramSTTProvider(
        config=DeepgramConfig(
            api_key=DEEPGRAM_API_KEY,
            model="nova-3",
            language=STT_LANGUAGE,
            punctuate=True,
            smart_format=True,
            endpointing=300,
        )
    )

    # --- Voice channel with bridge + STT --------------------------------------
    voice = VoiceChannel(
        "voice",
        backend=backend,
        stt=stt,
        bridge=True,
    )
    kit.register_channel(voice)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session(event, ctx):
        logger.info("Session started: %s", event.session.id)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        """Log each caller's transcribed speech in real time."""
        # Use SIP display name or user part, fall back to session ID
        meta = event.session.metadata
        name = meta.get("caller_display_name") or meta.get("caller_user") or event.session.id
        logger.info("[TRANSCRIPT %s] %s", name, event.text)
        return HookResult.allow()

    # --- Handle incoming calls ------------------------------------------------
    @backend.on_call
    async def handle_call(session):
        meta = session.metadata
        name = meta.get("caller_display_name") or meta.get("caller_user") or session.id
        logger.info("Incoming call — session=%s, caller=%s", session.id, name)
        await kit.join(ROOM_ID, "voice", session=session)
        count = voice._bridge.get_participant_count(ROOM_ID)
        logger.info(
            "Participants in room: %d — %s",
            count,
            "bridged!" if count >= 2 else "waiting for second caller",
        )

    # --- Start ----------------------------------------------------------------
    await backend.start()
    logger.info(
        "Listening for SIP calls on %s:%d — two callers will be bridged together",
        SIP_LISTEN_ADDR,
        SIP_LISTEN_PORT,
    )
    logger.info(
        "STT: Deepgram nova-3 (language=%s) — transcripts logged as [TRANSCRIPT]",
        STT_LANGUAGE,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    await backend.close()
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
