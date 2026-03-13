"""OpenCV video recorder — writes frames to MP4/AVI files.

Requires ``opencv-python-headless``::

    pip install roomkit[local-video]
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.video.recorder.base import (
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
    build_recording_path,
)

if TYPE_CHECKING:
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.recorder.opencv")


def _import_cv2() -> Any:
    """Import OpenCV, raising a clear error if missing."""
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for OpenCVVideoRecorder. "
            "Install with: pip install roomkit[local-video]"
        ) from exc


class _ActiveRecording:
    """Internal state for an active recording."""

    __slots__ = ("writer", "frame_count", "start_time", "path", "cv2", "fps", "codec")

    def __init__(self, writer: Any, path: str, cv2_mod: Any, *, fps: float, codec: str) -> None:
        self.writer = writer
        self.path = path
        self.cv2 = cv2_mod
        self.fps = fps
        self.codec = codec
        self.frame_count = 0
        self.start_time = time.monotonic()


class OpenCVVideoRecorder(VideoRecorder):
    """Video recorder using OpenCV's VideoWriter.

    Writes raw RGB frames to MP4 (or AVI) files. Frames are
    converted from RGB to BGR (OpenCV native) before writing.
    """

    def __init__(self) -> None:
        self._cv2 = _import_cv2()
        self._active: dict[str, _ActiveRecording] = {}

    @property
    def name(self) -> str:
        return "OpenCVVideoRecorder"

    def start(self, session: VideoSession, config: VideoRecordingConfig) -> VideoRecordingHandle:
        rec_id = uuid4().hex[:12]
        path = build_recording_path(session, config)

        codec = config.codec if config.codec != "auto" else "mp4v"

        handle = VideoRecordingHandle(
            id=rec_id,
            session_id=session.id,
            path=path,
        )
        # Writer is created lazily on first frame (need dimensions)
        self._active[rec_id] = _ActiveRecording(
            writer=None, path=path, cv2_mod=self._cv2, fps=config.fps, codec=codec
        )
        logger.info("Recording started: %s → %s", rec_id, path)
        return handle

    def stop(self, handle: VideoRecordingHandle) -> VideoRecordingResult:
        handle.state = "stopped"
        active = self._active.pop(handle.id, None)
        if active is None:
            return VideoRecordingResult(id=handle.id)

        if active.writer is not None:
            active.writer.release()

        elapsed = time.monotonic() - active.start_time
        size = os.path.getsize(active.path) if os.path.exists(active.path) else 0

        logger.info(
            "Recording stopped: %s (%d frames, %.1fs, %d bytes)",
            handle.id,
            active.frame_count,
            elapsed,
            size,
        )
        return VideoRecordingResult(
            id=handle.id,
            url=active.path,
            frame_count=active.frame_count,
            duration_seconds=round(elapsed, 2),
            format=os.path.splitext(active.path)[1].lstrip("."),
            size_bytes=size,
        )

    def tap_frame(self, handle: VideoRecordingHandle, frame: VideoFrame) -> None:
        active = self._active.get(handle.id)
        if active is None:
            return

        cv2 = active.cv2

        # Lazy-create writer on first frame (now we know dimensions)
        if active.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*active.codec)
            active.writer = cv2.VideoWriter(
                active.path, fourcc, active.fps, (frame.width, frame.height)
            )
            if not active.writer.isOpened():
                logger.error("Failed to open VideoWriter for %s", active.path)
                active.writer = None
                return

        # Convert raw RGB to BGR for OpenCV
        if frame.codec == "raw_rgb24":
            import numpy as np

            pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 3
            )
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            active.writer.write(bgr)
            active.frame_count += 1
        elif frame.codec == "raw_bgr24":
            import numpy as np

            bgr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            active.writer.write(bgr)
            active.frame_count += 1
        else:
            # Encoded frames can't be written directly by VideoWriter
            logger.debug("Skipping encoded frame (codec=%s)", frame.codec)

    def close(self) -> None:
        for rec_id, active in list(self._active.items()):
            if active.writer is not None:
                active.writer.release()
            logger.info("Recording closed: %s", rec_id)
        self._active.clear()
