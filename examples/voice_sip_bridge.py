#!/usr/bin/env python3
"""Bridge two SIP calls — human-to-human voice via audio forwarding.

Demonstrates VoiceChannel with ``bridge=True``: audio from caller A is
forwarded directly to caller B and vice versa, bypassing the STT/TTS
pipeline.  Optionally adds STT for live transcription.

The example accepts two incoming SIP calls (INVITEs) and bridges them
in the same room.  Audio flows directly between the two participants
at native quality — no AI, no text roundtrip.

Requirements:
    pip install roomkit[sip]

Run with:
    uv run python examples/voice_sip_bridge.py

Environment variables (all optional):
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
    ChannelBinding,
    ChannelType,
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.backends.sip import SIPVoiceBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_LISTEN_ADDR = os.environ.get("SIP_LISTEN_ADDR", "0.0.0.0")
SIP_LISTEN_PORT = int(os.environ.get("SIP_LISTEN_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")

ROOM_ID = "bridge-room"


async def main() -> None:
    kit = RoomKit()

    # --- SIP backend ----------------------------------------------------------
    backend = SIPVoiceBackend(
        local_sip_addr=(SIP_LISTEN_ADDR, SIP_LISTEN_PORT),  # nosec B104
        local_rtp_ip=RTP_IP,  # nosec B104
    )

    # --- Voice channel with bridge enabled ------------------------------------
    voice = VoiceChannel(
        "voice",
        backend=backend,
        bridge=True,  # Enable audio forwarding between sessions
        # stt=deepgram,  # Uncomment to add live transcription
    )
    kit.register_channel(voice)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session(event, ctx):
        logger.info("Session started: %s", event.session.id)

    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech(event, ctx):
        logger.info("Speech detected on session %s", event.session.id)

    # --- Handle incoming calls ------------------------------------------------
    call_count = 0

    @backend.on_call
    async def handle_call(session):
        nonlocal call_count
        call_count += 1
        participant_id = f"caller-{call_count}"
        logger.info(
            "Incoming call #%d — session=%s, participant=%s",
            call_count,
            session.id,
            participant_id,
        )
        binding = ChannelBinding(
            room_id=ROOM_ID,
            channel_id="voice",
            channel_type=ChannelType.VOICE,
        )
        voice.bind_session(session, ROOM_ID, binding)
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
