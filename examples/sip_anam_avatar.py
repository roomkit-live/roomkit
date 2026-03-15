"""RoomKit — SIP-to-Anam AI Avatar bridge.

Accept incoming SIP calls and connect them to an Anam AI avatar.
The caller's audio is forwarded to Anam (via ``send_user_audio``),
and Anam's avatar audio+video is streamed back over SIP/RTP.

Architecture:
    SIP phone → RTP audio → RealtimeAudioVideoChannel
                                    ↕
                             AnamRealtimeProvider
                                    ↕  (WebRTC)
                               Anam Cloud
                          STT → LLM → TTS → Avatar
                                    ↕
                        audio → SIP/RTP → caller
                        video → H.264 encode → SIP/RTP → caller

Prerequisites:
    pip install roomkit[anam,sip,video]

Run with:
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    export ANAM_VOICE_ID="your-voice-id"
    export ANAM_LLM_ID="your-llm-id"
    uv run python examples/sip_anam_avatar.py

Environment variables:
    ANAM_API_KEY       Anam API key (required)
    ANAM_AVATAR_ID     Avatar ID from lab.anam.ai (required)
    ANAM_VOICE_ID      Voice ID from lab.anam.ai (required)
    ANAM_LLM_ID        LLM ID from lab.anam.ai (required)
    ANAM_PERSONA_ID    Pre-defined persona (alternative to above three)
    SIP_PORT           SIP listener port (default: 5060)
    RTP_IP             IP to advertise in SDP (default: 0.0.0.0)
    RTP_PORT_START     First RTP port to allocate (default: 10000)
    DEBUG              Set to 1 for verbose logging

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

from roomkit import (
    AnamConfig,
    AnamRealtimeProvider,
    RealtimeAudioVideoChannel,
    RoomKit,
)
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeTransport


async def main() -> None:
    # --- Validate environment -------------------------------------------------
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

    # --- SIP backend for caller audio/video -----------------------------------
    sip_port = int(os.environ.get("SIP_PORT", "5060"))
    rtp_ip = os.environ.get("RTP_IP", "0.0.0.0")
    rtp_port_start = int(os.environ.get("RTP_PORT_START", "10000"))

    sip_backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", sip_port),
        local_rtp_ip=rtp_ip,
        rtp_port_start=rtp_port_start,
        supported_video_codecs=["H264"],
    )

    # --- Anam AI provider -----------------------------------------------------
    config = AnamConfig(
        api_key=api_key,
        persona_id=persona_id,
        avatar_id=avatar_id,
        voice_id=voice_id,
        llm_id=llm_id,
        system_prompt=(
            "You are a helpful AI avatar on a video call. "
            "Keep responses conversational and concise."
        ),
    )
    provider = AnamRealtimeProvider(config)

    # --- Realtime A/V channel -------------------------------------------------
    # The transport carries the caller's audio to/from the SIP backend.
    # Anam provider handles the AI pipeline and avatar rendering.
    transport = MockRealtimeTransport()

    channel = RealtimeAudioVideoChannel(
        "anam-sip",
        provider=provider,
        transport=transport,
    )

    # --- Video tap: log frames ------------------------------------------------
    frame_count = 0

    def on_video(session: object, frame: object) -> None:
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 1:
            logger.info("Avatar video frame #%d", frame_count)

    channel.add_video_media_tap(on_video)  # type: ignore[arg-type]

    # --- RoomKit setup --------------------------------------------------------
    kit = RoomKit()
    kit.register_channel(channel)

    # --- Route incoming SIP calls to rooms ------------------------------------
    async def on_call(session: VoiceSession) -> None:
        room_id = session.metadata.get("room_id", session.id)
        caller = session.metadata.get("caller", "unknown")
        has_video = session.metadata.get("has_video", False)

        logger.info(
            "Incoming SIP call: room=%s, caller=%s, video=%s",
            room_id,
            caller,
            has_video,
        )

        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "anam-sip")

        # Start the Anam avatar session for this caller
        await channel.start_session(
            room_id,
            participant_id=caller,
            connection=session,
        )
        logger.info("Anam avatar session started for caller %s", caller)

    sip_backend.on_call(on_call)

    def on_call_ended(session: object) -> None:
        session_id = getattr(session, "id", "unknown")
        room_id = getattr(session, "room_id", None)
        logger.info(
            "Call ended: session=%s, avatar_frames=%d",
            session_id,
            frame_count,
        )
        if room_id:
            asyncio.create_task(kit.close_room(room_id))

    sip_backend.on_client_disconnected(on_call_ended)

    # --- Start ----------------------------------------------------------------
    await sip_backend.start()

    logger.info("SIP + Anam Avatar bridge listening on 0.0.0.0:%d", sip_port)
    logger.info("Send a SIP INVITE to test.")
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C --------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    await sip_backend.close()
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
