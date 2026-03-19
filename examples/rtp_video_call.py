"""RoomKit — RTP audio+video direct transport.

Connect to a pre-configured RTP endpoint with both audio and video.
Audio goes through the voice pipeline (mock STT/TTS), video frames
are delivered to a callback.

Audio flow:
    RTP audio → Pipeline → STT → print

Video flow:
    RTP video → H.264 NAL → on_video_received → print

Prerequisites:
    pip install roomkit[rtp]

Run with:
    uv run python examples/rtp_video_call.py

Then send RTP media to the configured ports:

    ffmpeg -re -i video.mp4 \
        -c:v libx264 -f rtp rtp://127.0.0.1:10002 \
        -ar 8000 -ac 1 -acodec pcm_mulaw -f rtp rtp://127.0.0.1:10000

Environment variables:
    REMOTE_IP         Remote RTP endpoint IP (default: 127.0.0.1)
    REMOTE_PORT       Remote audio RTP port (default: 20000)
    REMOTE_VIDEO_PORT Remote video RTP port (default: 20002)
    LOCAL_PORT        Local audio RTP port (default: 10000)
    LOCAL_VIDEO_PORT  Local video RTP port (default: 10002)
    DEBUG             Set to 1 for verbose logging

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from roomkit import (
    AudioVideoChannel,
    HookResult,
    HookTrigger,
    RoomKit,
)
from roomkit.video.backends.rtp import RTPVideoBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("rtp_video_call")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)


async def main() -> None:
    kit = RoomKit()

    # --- Configuration --------------------------------------------------------
    remote_ip = os.environ.get("REMOTE_IP", "127.0.0.1")
    remote_port = int(os.environ.get("REMOTE_PORT", "20000"))
    remote_video_port = int(os.environ.get("REMOTE_VIDEO_PORT", "20002"))
    local_port = int(os.environ.get("LOCAL_PORT", "10000"))
    local_video_port = int(os.environ.get("LOCAL_VIDEO_PORT", "10002"))

    # --- RTP A/V backend ------------------------------------------------------
    backend = RTPVideoBackend(
        local_addr=("0.0.0.0", local_port),
        remote_addr=(remote_ip, remote_port),
        video_local_addr=("0.0.0.0", local_video_port),
        video_remote_addr=(remote_ip, remote_video_port),
    )

    # --- Video callback -------------------------------------------------------
    frame_count = 0

    def on_video(_session, frame):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 1:
            logger.info(
                "Video frame #%d: codec=%s %s seq=%d ts=%.1fms",
                frame_count,
                frame.codec,
                "KEY" if frame.keyframe else "   ",
                frame.sequence,
                frame.timestamp_ms or 0,
            )

    # --- A/V channel -----------------------------------------------------------
    av = AudioVideoChannel(
        "voice",
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(),
    )
    av.add_video_media_tap(on_video)
    kit.register_channel(av)

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        print(f"\n>>> {event.text}\n")
        return HookResult.block("demo — no AI provider")

    # --- Connect to room ------------------------------------------------------
    room_id = "rtp-video-room"
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "voice")

    session = await backend.connect(room_id, "remote-user", "voice")
    await kit.join(room_id, "voice", session=session)

    _ = backend.get_video_session(session.id)
    logger.info(
        "RTP A/V session active: audio=%s:%d, video=%s:%d",
        "0.0.0.0",
        local_port,
        "0.0.0.0",
        local_video_port,
    )
    logger.info(
        "Sending RTP to %s (audio:%d, video:%d)",
        remote_ip,
        remote_port,
        remote_video_port,
    )
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C --------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    await backend.disconnect(session)
    await kit.close()
    logger.info("Done. Received %d video frames.", frame_count)


if __name__ == "__main__":
    asyncio.run(main())
