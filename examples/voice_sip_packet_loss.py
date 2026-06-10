#!/usr/bin/env python3
"""SIP voice backend with packet loss concealment (PLC).

On lossy networks (Wi-Fi callers, congested trunks), RTP packets go
missing.  Without concealment the lost 20 ms frames are simply skipped:
the delivered audio timeline compresses, recordings shorten, and AEC
reference alignment drifts.  With ``plc=True`` (the default), packets
confirmed lost by the jitter buffer are replaced with concealment PCM —
native Opus PLC when the codec supports it, otherwise last-frame
repetition fading to silence over 60 ms — so the pipeline always
receives a temporally continuous stream.

Concealment is fully transparent: no pipeline stage needs configuration.
This example answers inbound calls and surfaces the diagnostics:

- the periodic stats line (DEBUG) shows ``concealed=N`` per session
- the final stats line (INFO) on hangup shows the session total

Sender pauses (DTMF, VAD suppression) are never concealed — detection
is sequence-number based, so only genuine network loss triggers it.

Requirements:
    pip install roomkit[sip]

Usage:
    python examples/voice_sip_packet_loss.py

    # Environment variables (all optional):
    #   SIP_LOCAL_PORT  — local SIP listen port (default: 5060)
    #   SIP_PLC         — set to 0 to disable concealment (A/B testing)

To observe concealment, call the agent over a lossy link (or simulate
loss, e.g. ``tc qdisc add dev eth0 root netem loss 5%`` on Linux) and
watch the ``concealed=`` counter climb while audio stays continuous.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_packet_loss")

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.backends.sip import SIPVoiceBackend

LOCAL_PORT = int(os.environ.get("SIP_LOCAL_PORT", "5060"))
PLC_ENABLED = os.environ.get("SIP_PLC", "1") != "0"

# The periodic per-session stats line (incl. concealed=N) logs at DEBUG
logging.getLogger("roomkit.voice.backends.sip").setLevel(logging.DEBUG)


async def main() -> None:
    kit = RoomKit()
    console_cleanup = setup_console(kit)

    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", LOCAL_PORT),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        skip_audio_gaps=True,  # default — required for loss confirmation
        plc=PLC_ENABLED,  # default True — replace confirmed losses
    )

    voice = VoiceChannel("voice", backend=backend)
    kit.register_channel(voice)

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "voice")

    @backend.on_call
    async def handle_call(session):
        logger.info(
            "Incoming call — session=%s plc=%s",
            session.id,
            "enabled" if PLC_ENABLED else "disabled",
        )
        await kit.join("support", "voice", session=session)

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        # Final stats (incl. concealed=N) are logged by the backend at INFO
        logger.info("Call ended — session=%s", session.id)
        await kit.leave(session)

    await backend.start()
    logger.info(
        "SIP agent ready on port %d — packet loss concealment %s",
        LOCAL_PORT,
        "enabled" if PLC_ENABLED else "disabled",
    )

    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
