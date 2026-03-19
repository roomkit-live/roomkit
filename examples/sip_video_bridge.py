"""RoomKit — Bridge two SIP audio+video calls.

Accept two incoming SIP calls and bridge them: audio flows directly
between participants via AudioBridge, video flows via VideoBridge.
Both sides hear and see each other at native quality — no AI roundtrip.

Architecture::

    Caller A ──audio──► Pipeline ──► AudioBridge ──► Caller B speaker
                                         ▲
    Caller B ──audio──► Pipeline ────────┘

    Caller A ──video──► Pipeline ──► VideoBridge ──► Caller B display
                                         ▲
    Caller B ──video──► Pipeline ────────┘

Optional STT transcribes both sides in parallel (bridge + STT are
independent paths).

Prerequisites:
    pip install roomkit[sip]

Run with:
    uv run python examples/sip_video_bridge.py

Then send two SIP INVITEs (with m=audio + m=video) to port 5060.
The two callers will be bridged together.

Environment variables:
    SIP_PORT         SIP listener port (default: 5060)
    RTP_IP           IP to advertise in SDP (default: 0.0.0.0)
    RTP_PORT_START   First RTP port to allocate (default: 10000)
    DEBUG            Set to 1 for verbose logging

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from roomkit import (
    AudioVideoChannel,
    ChannelBinding,
    ChannelType,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VideoBridgeConfig,
)
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("sip_video_bridge")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_PORT = int(os.environ.get("SIP_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")
RTP_PORT_START = int(os.environ.get("RTP_PORT_START", "10000"))
ROOM_ID = "bridge-room"


async def main() -> None:
    kit = RoomKit()

    # --- SIP A/V backend ------------------------------------------------------
    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", SIP_PORT),
        local_rtp_ip=RTP_IP,
        rtp_port_start=RTP_PORT_START,
        supported_video_codecs=["H264", "VP8", "VP9"],
    )

    # --- A/V channel with audio + video bridge --------------------------------
    # bridge=True enables AudioBridge (from VoiceChannel parent).
    # video_bridge=True enables VideoBridge for video forwarding.
    av = AudioVideoChannel(
        "av",
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(),
        bridge=True,
        video_bridge=VideoBridgeConfig(max_participants=2),
    )
    kit.register_channel(av)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "av")

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session(event, ctx):
        logger.info("Session started: %s", event.session.id)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        """Log speech transcriptions from both participants."""
        meta = event.session.metadata
        name = meta.get("caller_display_name") or meta.get("caller_user") or event.session.id
        logger.info("[TRANSCRIPT %s] %s", name, event.text)
        return HookResult.allow()

    # --- Video frame counter for logging --------------------------------------
    frame_counts: dict[str, int] = {}

    def on_video(session, frame):
        sid = session.id
        frame_counts[sid] = frame_counts.get(sid, 0) + 1
        count = frame_counts[sid]
        if count % 30 == 1:  # log every ~1s at 30fps
            logger.info(
                "Video: session=%s frame=#%d codec=%s %dx%d %s",
                sid[:8],
                count,
                frame.codec,
                frame.width,
                frame.height,
                "KEY" if frame.keyframe else "",
            )

    av.add_video_media_tap(on_video)

    # --- Handle incoming SIP calls --------------------------------------------
    @backend.on_call
    async def handle_call(session: VoiceSession) -> None:
        meta = session.metadata
        has_video = meta.get("has_video", False)
        name = meta.get("caller_display_name") or meta.get("caller_user") or session.id

        logger.info(
            "Incoming call: session=%s caller=%s video=%s",
            session.id[:8],
            name,
            has_video,
        )

        binding = ChannelBinding(
            room_id=ROOM_ID,
            channel_id="av",
            channel_type=ChannelType.AUDIO_VIDEO,
        )
        av.bind_session(session, ROOM_ID, binding)

        audio_count = av._bridge.get_participant_count(ROOM_ID) if av._bridge else 0
        video_count = av._video_bridge.get_participant_count(ROOM_ID) if av._video_bridge else 0
        logger.info(
            "Room: %d audio, %d video — %s",
            audio_count,
            video_count,
            "bridged!" if audio_count >= 2 else "waiting for second caller",
        )

    def on_call_ended(session: object) -> None:
        sid = getattr(session, "id", "unknown")
        total = frame_counts.pop(sid, 0)
        audio_count = av._bridge.get_participant_count(ROOM_ID) if av._bridge else 0
        video_count = av._video_bridge.get_participant_count(ROOM_ID) if av._video_bridge else 0
        logger.info(
            "Call ended: session=%s, video_frames=%d — room: %d audio, %d video",
            sid[:8],
            total,
            audio_count,
            video_count,
        )

    backend.on_client_disconnected(on_call_ended)

    # --- Start ----------------------------------------------------------------
    await backend.start()
    logger.info("SIP A/V bridge listening on 0.0.0.0:%d", SIP_PORT)
    logger.info("Send two SIP INVITEs (with m=audio + m=video) to bridge.")
    logger.info("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    await backend.close()
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
