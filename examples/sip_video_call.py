"""RoomKit — SIP audio+video call handler.

Accept incoming SIP calls with audio and video, transcribe speech with
a mock STT provider, and deliver H.264 video frames to a vision callback.
Audio-only calls are handled transparently — video is added when the remote
party offers it.

Audio flow:
    SIP INVITE → SDP negotiation (A/V) → RTP audio → Pipeline → STT → print

Video flow:
    SIP INVITE → SDP negotiation → RTP video → H.264 NAL → on_video_received → print

Prerequisites:
    pip install roomkit[sip]

Run with:
    uv run python examples/sip_video_call.py

Then send a SIP INVITE (with m=audio + m=video) to port 5060.

Environment variables:
    SIP_PORT         SIP listener port (default: 5060)
    RTP_IP           IP to advertise in SDP (default: auto-detect)
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
    HookResult,
    HookTrigger,
    RoomKit,
)
from roomkit.recorder.base import (
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.pyav import PyAVMediaRecorder
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sip_video_call")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)


async def main() -> None:
    kit = RoomKit()

    # --- Configuration --------------------------------------------------------
    sip_port = int(os.environ.get("SIP_PORT", "5060"))
    rtp_ip = os.environ.get("RTP_IP", "0.0.0.0")
    rtp_port_start = int(os.environ.get("RTP_PORT_START", "10000"))

    # --- SIP A/V backend ------------------------------------------------------
    # SIPVideoBackend extends SIPVoiceBackend: audio-only calls work normally,
    # video is added when the remote party includes m=video in the INVITE.
    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", sip_port),
        local_rtp_ip=rtp_ip,
        rtp_port_start=rtp_port_start,
        supported_video_codecs=["H264", "VP8", "VP9"],
    )

    # --- Video callback — print frame info ------------------------------------
    frame_count = 0

    def on_video(session, frame):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 1:  # log every ~1s at 30fps
            logger.info(
                "Video frame #%d: codec=%s %s seq=%d ts=%.1fms",
                frame_count,
                frame.codec,
                "KEY" if frame.keyframe else "   ",
                frame.sequence,
                frame.timestamp_ms or 0,
            )

    # --- A/V channel (mock STT/TTS for demo) ------------------------------------
    av = AudioVideoChannel(
        "voice",
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(),
    )
    av.add_video_media_tap(on_video)
    kit.register_channel(av)

    # --- Hooks: print transcriptions ------------------------------------------
    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        print(f"\n>>> {event.text}\n")
        return HookResult.block("demo — no AI provider")

    # --- Room recorder --------------------------------------------------------
    recorder = PyAVMediaRecorder()
    recorder_binding = RoomRecorderBinding(
        recorder=recorder,
        config=MediaRecordingConfig(storage="recordings", video_codec="libx264"),
    )

    # --- Route incoming SIP calls to rooms ------------------------------------
    async def on_call(session: VoiceSession) -> None:
        room_id = session.metadata.get("room_id", session.id)
        has_video = session.metadata.get("has_video", False)
        caller = session.metadata.get("caller", "unknown")

        logger.info(
            "Incoming call: room=%s, caller=%s, video=%s",
            room_id,
            caller,
            has_video,
        )

        await kit.create_room(
            room_id=room_id,
            recorders=[recorder_binding],
        )
        await kit.attach_channel(room_id, "voice")
        await kit.join(room_id, "voice", session=session)

    backend.on_call(on_call)

    def on_call_ended(session: object) -> None:
        """Finalize recording when the SIP call ends (BYE received)."""
        session_id = getattr(session, "id", "unknown")
        room_id = getattr(session, "room_id", None)
        logger.info("Call ended: session=%s, video_frames=%d", session_id, frame_count)
        if room_id:
            asyncio.create_task(kit.close_room(room_id))

    backend.on_client_disconnected(on_call_ended)

    # --- Start ----------------------------------------------------------------
    await backend.start()

    logger.info("SIP A/V backend listening on 0.0.0.0:%d", sip_port)
    logger.info("Send a SIP INVITE with m=audio + m=video to test.")
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C --------------------------------------------
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
