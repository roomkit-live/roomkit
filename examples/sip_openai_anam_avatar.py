"""RoomKit — SIP + OpenAI Realtime + Anam Avatar.

Accept SIP video calls. OpenAI Realtime handles the conversation
(speech-to-speech), Anam renders the avatar face (passthrough mode).

Architecture:
    SIP phone → audio → RealtimeAVBridge → OpenAI Realtime (STT → LLM → TTS)
                                                 ↓
                                          TTS audio out
                                           ↓          ↓
                                    SIP speaker    AnamAvatarProvider
                                    (caller)         ↓
                                                  Anam Cloud (lip-sync)
                                                     ↓
                                               video frames
                                                     ↓
                                          pipeline → H.264 → SIP video

Prerequisites:
    pip install roomkit[anam,sip,video,realtime-openai]

Run with:
    export OPENAI_API_KEY="sk-..."
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    export ANAM_VOICE_ID="your-voice-id"
    export ANAM_LLM_ID="your-llm-id"
    uv run python examples/sip_openai_anam_avatar.py

Environment variables:
    OPENAI_API_KEY       OpenAI API key (required)
    OPENAI_MODEL         Realtime model (default: gpt-4o-mini-realtime-preview)
    OPENAI_VOICE         Voice preset (default: alloy)
    ANAM_API_KEY         Anam API key (required)
    ANAM_AVATAR_ID       Avatar from lab.anam.ai (required)
    ANAM_VOICE_ID        Voice from lab.anam.ai (required)
    ANAM_LLM_ID          LLM from lab.anam.ai (required)
    SIP_PORT             SIP listener port (default: 5060)
    RTP_IP               IP for SDP (default: 0.0.0.0)
    DEBUG                Set to 1 for verbose logging

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
logger = logging.getLogger("sip_openai_anam")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)

from roomkit.providers.anam import AnamConfig
from roomkit.providers.anam.avatar import AnamAvatarProvider
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.pipeline import VideoPipelineConfig
from roomkit.video.pipeline.encoder.pyav import PyAVVideoEncoder
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.video.utils import make_text_frame
from roomkit.voice.realtime.bridge import RealtimeAVBridge


async def main() -> None:
    # --- Validate environment -------------------------------------------------
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anam_key = os.environ.get("ANAM_API_KEY", "")
    if not openai_key or not anam_key:
        logger.error("Set OPENAI_API_KEY and ANAM_API_KEY")
        return

    avatar_id = os.environ.get("ANAM_AVATAR_ID")
    voice_id = os.environ.get("ANAM_VOICE_ID")
    llm_id = os.environ.get("ANAM_LLM_ID")
    if not (avatar_id and voice_id and llm_id):
        logger.error("Set ANAM_AVATAR_ID + ANAM_VOICE_ID + ANAM_LLM_ID")
        return

    # --- SIP backend ----------------------------------------------------------
    sip = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", int(os.environ.get("SIP_PORT", "5060"))),
        local_rtp_ip=os.environ.get("RTP_IP", "0.0.0.0"),
        rtp_port_start=int(os.environ.get("RTP_PORT_START", "10000")),
        supported_video_codecs=["H264"],
    )

    # --- OpenAI Realtime (speech-to-speech, audio only) -----------------------
    openai_provider = OpenAIRealtimeProvider(
        api_key=openai_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini-realtime-preview"),
    )

    # --- Anam avatar (passthrough — lip-sync only, no STT/LLM) ---------------
    avatar = AnamAvatarProvider(
        AnamConfig(
            api_key=anam_key,
            avatar_id=avatar_id,
            voice_id=voice_id,
            llm_id=llm_id,
            enable_audio_passthrough=True,
        ),
        audio_sample_rate=24000,
    )

    # --- Bridge: SIP ↔ OpenAI + Anam avatar -----------------------------------
    bridge = RealtimeAVBridge(
        openai_provider,
        sip,
        avatar=avatar,
        video_pipeline=VideoPipelineConfig(
            filters=[
                WatermarkFilter(
                    "RoomKit | OpenAI + Anam | {timestamp}",
                    position="bottom-left",
                    font_scale=0.4,
                ),
            ],
        ),
        encoder=PyAVVideoEncoder(fps=25, bitrate=3_000_000, preset="medium"),
        connecting_frame=make_text_frame("Connecting...\nPlease wait"),
        provider_sample_rate=24000,
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a helpful AI assistant on a video call. "
            "Respond in the same language as the user. "
            "Keep responses conversational and concise.",
        ),
        voice=os.environ.get("OPENAI_VOICE", "alloy"),
        on_transcription=lambda role, text, _: logger.info("[%s] %s", role.upper(), text),
    )

    # --- Start ----------------------------------------------------------------
    await sip.start()
    logger.info(
        "SIP + OpenAI Realtime + Anam Avatar on 0.0.0.0:%s",
        os.environ.get("SIP_PORT", "5060"),
    )
    logger.info("Call this SIP endpoint with a video phone.")
    logger.info("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()

    def _signal() -> None:
        if stop.is_set():
            raise SystemExit(1)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal)
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
