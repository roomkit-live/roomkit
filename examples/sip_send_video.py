"""RoomKit -- SIP send video test: encode and send a test pattern.

Accepts SIP calls and sends a simple color test pattern as H.264
video to the caller.  Tests the full outbound video path:
  raw RGB → H.264 encode (PyAV) → RTP packetize → UDP send

Prerequisites:
    pip install roomkit[sip,video]

Run with:
    uv run python examples/sip_send_video.py

Then call the SIP endpoint — you should see colored video.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import signal

import av
import numpy as np

from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.voice.base import VoiceSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sip_send_video")


class SimpleH264Encoder:
    """Encode raw RGB frames to H.264 NAL units using PyAV."""

    def __init__(self, width: int = 320, height: int = 240, fps: int = 30) -> None:
        self._width = width
        self._height = height
        self._codec_ctx = av.CodecContext.create("libx264", "w")
        self._codec_ctx.width = width
        self._codec_ctx.height = height
        self._codec_ctx.pix_fmt = "yuv420p"
        from fractions import Fraction

        self._codec_ctx.time_base = Fraction(1, fps)
        self._codec_ctx.options = {"tune": "zerolatency", "preset": "ultrafast"}
        self._codec_ctx.open()
        self._pts = 0

    def encode(self, rgb_data: bytes) -> list[bytes]:
        """Encode one RGB frame, return H.264 NAL unit bytes."""
        arr = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
            self._height,
            self._width,
            3,
        )
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = self._pts
        self._pts += 1

        packets = self._codec_ctx.encode(frame)
        return [bytes(pkt) for pkt in packets]

    def flush(self) -> list[bytes]:
        packets = self._codec_ctx.encode(None)
        return [bytes(pkt) for pkt in packets]


def make_test_frame(
    width: int,
    height: int,
    frame_num: int,
) -> bytes:
    """Generate a colored test pattern frame (RGB24)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Cycle through colors every 30 frames
    colors = [
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 0, 255),  # blue
        (255, 255, 0),  # yellow
        (0, 255, 255),  # cyan
        (255, 0, 255),  # magenta
    ]
    color = colors[(frame_num // 30) % len(colors)]
    arr[:, :] = color

    # Add a moving white bar
    bar_y = (frame_num * 3) % height
    bar_h = min(10, height - bar_y)
    arr[bar_y : bar_y + bar_h, :] = (255, 255, 255)

    return arr.tobytes()


async def main() -> None:
    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", 5060),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        rtp_port_start=10000,
        supported_video_codecs=["H264", "VP9"],
    )

    encoder = SimpleH264Encoder(width=320, height=240, fps=15)
    active_tasks: dict[str, asyncio.Task[None]] = {}

    async def send_video_loop(session: VoiceSession) -> None:
        """Send test pattern video frames to the caller."""

        # Wait a bit for video RTP session to start
        await asyncio.sleep(1.0)

        video_session = backend.get_video_session(session.id)
        if video_session is None:
            logger.warning("No video session for %s", session.id[:8])
            return

        logger.info("Starting video send loop for %s", session.id[:8])
        frame_num = 0
        try:
            while True:
                rgb = make_test_frame(320, 240, frame_num)
                nals = encoder.encode(rgb)
                for nal in nals:
                    vcs = backend._video_call_sessions.get(session.id)
                    if vcs is None:
                        return
                    # Send H.264 NAL units via RTP
                    ts = frame_num * (90000 // 15)  # 90kHz clock, 15fps
                    vcs.send_frame([nal], ts, keyframe=(frame_num % 30 == 0))

                frame_num += 1
                if frame_num % 30 == 0:
                    logger.info("Sent %d video frames to %s", frame_num, session.id[:8])
                await asyncio.sleep(1.0 / 15)  # 15 fps
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Video send error")

    async def on_call(session: VoiceSession) -> None:
        has_video = session.metadata.get("has_video", False)
        caller = session.metadata.get("caller", "unknown")
        logger.info("Call from %s (video=%s)", caller, has_video)

        if has_video:
            task = asyncio.create_task(send_video_loop(session))
            active_tasks[session.id] = task

    def on_disconnect(session: object) -> None:
        sid = getattr(session, "id", "")
        task = active_tasks.pop(sid, None)
        if task:
            task.cancel()
        logger.info("Call ended: %s", sid[:8])

    backend.on_call(on_call)
    backend.on_client_disconnected(on_disconnect)

    await backend.start()

    print("SIP Send Video Test")
    print("=" * 60)
    print("SIP     : 0.0.0.0:5060")
    print("Video   : H.264 test pattern (320x240 @ 15fps)")
    print("Colors cycle: red → green → blue → yellow → cyan → magenta")
    print("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    for task in active_tasks.values():
        task.cancel()
    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
