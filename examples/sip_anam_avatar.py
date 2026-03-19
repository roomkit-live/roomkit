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

from roomkit.providers.anam import AnamConfig, AnamRealtimeProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.pipeline import VideoPipelineConfig
from roomkit.video.pipeline.encoder.pyav import PyAVVideoEncoder
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.video.utils import make_text_frame
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

    # --- Video pipeline: overlay "RoomKit" + timestamp on avatar video ------
    video_pipeline = VideoPipelineConfig(
        filters=[
            WatermarkFilter(
                "RoomKit | {timestamp}",
                position="bottom-left",
                color=(255, 255, 255),
                bg_color=(0, 0, 0),
                font_scale=0.5,
            ),
        ],
    )

    # --- Bridge: SIP ↔ Anam (pipeline + H.264 encoding) --------------------
    bridge = RealtimeAVBridge(
        provider,
        sip,
        video_pipeline=video_pipeline,
        encoder=PyAVVideoEncoder(fps=25, bitrate=3_000_000, preset="medium"),
        connecting_frame=make_text_frame("Connecting to avatar...\nPlease wait"),
        provider_sample_rate=48000,
        on_transcription=lambda role, text, _: logger.info("[%s] %s", role.upper(), text),
    )

    # --- Start ----------------------------------------------------------------
    await sip.start()
    logger.info("SIP + Anam Avatar bridge on 0.0.0.0:%s", os.environ.get("SIP_PORT", "5060"))
    logger.info("Call this SIP endpoint with a video phone.")
    logger.info("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()
    force_exit = False

    def _signal_handler() -> None:
        nonlocal force_exit
        if stop.is_set():
            # Second Ctrl+C: force exit
            force_exit = True
            logger.warning("Force exit")
            raise SystemExit(1)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)
    await stop.wait()

    logger.info("Shutting down (Ctrl+C again to force)...")
    try:
        await asyncio.wait_for(bridge.close(), timeout=5.0)
    except TimeoutError:
        logger.warning("Bridge close timed out")
    await sip.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
