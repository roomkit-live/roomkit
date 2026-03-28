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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio

import av
import numpy as np
from shared import run_until_stopped, setup_logging

from roomkit import RoomKit
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.voice.base import VoiceSession

logger = setup_logging("sip_send_video")


class SimpleH264Encoder:
    """Encode raw RGB frames to H.264 NAL units using PyAV.

    Outputs individual NAL units (split from Annex B stream) suitable
    for RTP packetization.  Uses Constrained Baseline profile for
    WebRTC compatibility.
    """

    def __init__(self, width: int = 320, height: int = 240, fps: int = 30) -> None:
        self._width = width
        self._height = height
        self._codec_ctx = av.CodecContext.create("libx264", "w")
        self._codec_ctx.width = width
        self._codec_ctx.height = height
        self._codec_ctx.pix_fmt = "yuv420p"
        from fractions import Fraction

        self._codec_ctx.time_base = Fraction(1, fps)
        self._codec_ctx.options = {
            "tune": "zerolatency",
            "preset": "ultrafast",
            "profile": "baseline",
            "level": "3.1",
        }
        self._codec_ctx.open()
        self._pts = 0

    def encode(self, rgb_data: bytes) -> list[bytes]:
        """Encode one RGB frame, return individual H.264 NAL units."""
        arr = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
            self._height,
            self._width,
            3,
        )
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = self._pts
        self._pts += 1

        packets = self._codec_ctx.encode(frame)
        nals: list[bytes] = []
        for pkt in packets:
            nals.extend(self._split_annex_b(bytes(pkt)))
        return nals

    def flush(self) -> list[bytes]:
        packets = self._codec_ctx.encode(None)
        nals: list[bytes] = []
        for pkt in packets:
            nals.extend(self._split_annex_b(bytes(pkt)))
        return nals

    @staticmethod
    def _split_annex_b(data: bytes) -> list[bytes]:
        """Split Annex B byte stream into individual NAL units."""
        nals: list[bytes] = []
        i = 0
        start = -1
        while i < len(data):
            # Look for 3-byte or 4-byte start codes
            if i + 3 <= len(data) and data[i : i + 3] == b"\x00\x00\x01":
                if start >= 0:
                    nals.append(data[start:i])
                start = i + 3
                i += 3
            elif i + 4 <= len(data) and data[i : i + 4] == b"\x00\x00\x00\x01":
                if start >= 0:
                    nals.append(data[start:i])
                start = i + 4
                i += 4
            else:
                i += 1
        if start >= 0 and start < len(data):
            nals.append(data[start:])
        return nals


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
    kit = RoomKit()

    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", 5060),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        rtp_port_start=10000,
        supported_video_codecs=["H264"],  # must match our encoder
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
                if nals:
                    vcs = backend._video_call_sessions.get(session.id)
                    if vcs is None:
                        return
                    # Send all NAL units for this frame at once
                    # (send_frame sets marker=1 on last RTP packet)
                    ts = frame_num * (90000 // 15)  # 90kHz clock, 15fps
                    is_key = any((nal[0] & 0x1F) == 5 for nal in nals if nal)
                    vcs.send_frame(nals, ts, is_key)

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

    logger.info("SIP Send Video Test")
    logger.info("SIP     : 0.0.0.0:5060")
    logger.info("Video   : H.264 test pattern (320x240 @ 15fps)")
    logger.info("Colors cycle: red -> green -> blue -> yellow -> cyan -> magenta")
    logger.info("Press Ctrl+C to stop.")

    async def cleanup() -> None:
        for task in active_tasks.values():
            task.cancel()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
