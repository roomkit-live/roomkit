#!/usr/bin/env python3
"""SIP voice backend with RFC 3389 comfort noise on outbound silence.

When the agent stops talking (between TTS responses, while the LLM
thinks), a plain RTP stream goes silent — many carriers and handsets
read that as a dead call, and callers hear unsettling absolute silence.
With ``cn=True`` the backend emits RFC 3389 comfort-noise packets during
outbound silence (via aiortp): the line keeps a natural low-level hiss,
matched to the level of the preceding speech, at a fraction of the
bandwidth of real audio frames.

Comfort noise is fully transparent to the pipeline: no stage needs
configuration, and talkspurt resumption is marked on the RTP stream so
the remote jitter buffer resynchronises cleanly.

Requirements:
    pip install roomkit[sip]

Usage:
    python examples/voice_sip_comfort_noise.py

    # Environment variables (all optional):
    #   SIP_LOCAL_PORT      — local SIP listen port (default: 5060)
    #   SIP_RTP_PORT_START  — first RTP port (default: 10000)
    #   SIP_CN              — set to 0 to disable comfort noise (A/B testing)

To hear the difference, call the agent and stay quiet after its first
response: with SIP_CN=1 the line carries a soft hiss, with SIP_CN=0 it
goes fully dead between responses.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_comfort_noise")

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.backends.sip import SIPVoiceBackend

LOCAL_PORT = int(os.environ.get("SIP_LOCAL_PORT", "5060"))
RTP_PORT_START = int(os.environ.get("SIP_RTP_PORT_START", "10000"))
CN_ENABLED = os.environ.get("SIP_CN", "1") != "0"


async def main() -> None:
    kit = RoomKit()
    console_cleanup = setup_console(kit)

    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", LOCAL_PORT),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_START + 1000,
        cn=CN_ENABLED,  # RFC 3389 comfort noise on outbound silence
    )

    voice = VoiceChannel("voice", backend=backend)
    kit.register_channel(voice)

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "voice")

    @backend.on_call
    async def handle_call(session):
        logger.info(
            "Incoming call — session=%s comfort noise=%s",
            session.id,
            "enabled" if CN_ENABLED else "disabled",
        )
        await kit.join("support", "voice", session=session)

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("Call ended — session=%s", session.id)
        await kit.leave(session)

    await backend.start()
    logger.info(
        "SIP agent ready on port %d — comfort noise %s",
        LOCAL_PORT,
        "enabled" if CN_ENABLED else "disabled",
    )

    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(cleanup)


if __name__ == "__main__":
    asyncio.run(main())
