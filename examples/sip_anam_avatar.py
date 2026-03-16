"""RoomKit — SIP-to-Anam AI Avatar bridge.

Accept incoming SIP video calls and connect them to an Anam AI avatar.
Caller audio → Anam STT → LLM → TTS → avatar audio+video → SIP/RTP → caller.

Prerequisites:
    pip install roomkit[anam,sip,video]

Run with:
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    export ANAM_VOICE_ID="your-voice-id"
    export ANAM_LLM_ID="your-llm-id"
    uv run python examples/sip_anam_avatar.py

Environment variables:
    ANAM_API_KEY         Anam API key (required)
    ANAM_AVATAR_ID       Avatar ID from lab.anam.ai (required)
    ANAM_VOICE_ID        Voice ID from lab.anam.ai (required)
    ANAM_LLM_ID          LLM ID from lab.anam.ai (required)
    ANAM_PERSONA_ID      Pre-defined persona (alternative to above three)
    ANAM_LANGUAGE         Language code, e.g. "fr" (default: "en")
    ANAM_SYSTEM_PROMPT    System prompt for the LLM
    SIP_PORT              SIP listener port (default: 5060)
    RTP_IP                IP to advertise in SDP (default: 0.0.0.0)
    DEBUG                 Set to 1 for verbose logging

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sip_anam_avatar")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)

from roomkit import AnamConfig, AnamRealtimeProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.pipeline.encoder.pyav import PyAVVideoEncoder
from roomkit.voice.realtime.bridge import RealtimeAVBridge


async def main() -> None:
    api_key = os.environ.get("ANAM_API_KEY", "")
    if not api_key:
        logger.error("Set ANAM_API_KEY environment variable")
        return

    persona_id = os.environ.get("ANAM_PERSONA_ID")
    avatar_id = os.environ.get("ANAM_AVATAR_ID")
    voice_id = os.environ.get("ANAM_VOICE_ID")
    llm_id = os.environ.get("ANAM_LLM_ID")

    if not persona_id and not (avatar_id and voice_id and llm_id):
        logger.error("Set either ANAM_PERSONA_ID or ANAM_AVATAR_ID + ANAM_VOICE_ID + ANAM_LLM_ID")
        return

    # --- SIP backend ----------------------------------------------------------
    sip = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", int(os.environ.get("SIP_PORT", "5060"))),
        local_rtp_ip=os.environ.get("RTP_IP", "0.0.0.0"),
        rtp_port_start=int(os.environ.get("RTP_PORT_START", "10000")),
        supported_video_codecs=["H264"],
    )

    # --- Anam provider --------------------------------------------------------
    provider = AnamRealtimeProvider(
        AnamConfig(
            api_key=api_key,
            persona_id=persona_id,
            avatar_id=avatar_id,
            voice_id=voice_id,
            llm_id=llm_id,
            language_code=os.environ.get("ANAM_LANGUAGE", "en"),
            system_prompt=os.environ.get(
                "ANAM_SYSTEM_PROMPT",
                "You are a helpful AI avatar on a video call. "
                "Keep responses conversational and concise.",
            ),
        )
    )

    # --- Bridge: SIP ↔ Anam (with H.264 encoding) ----------------------------
    bridge = RealtimeAVBridge(
        provider,
        sip,
        encoder=PyAVVideoEncoder(fps=25),
        provider_sample_rate=48000,
        on_transcription=lambda role, text, _: logger.info("[%s] %s", role.upper(), text),
    )

    # --- Start ----------------------------------------------------------------
    await sip.start()
    logger.info("SIP + Anam Avatar bridge on 0.0.0.0:%s", os.environ.get("SIP_PORT", "5060"))
    logger.info("Call this SIP endpoint with a video phone.")
    logger.info("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    await bridge.close()
    await sip.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
