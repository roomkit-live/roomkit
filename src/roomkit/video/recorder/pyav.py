"""PyAV video recorder — H.264/H.265 encoding to MP4.

Produces properly compressed MP4 files using FFmpeg via PyAV.
Supports software encoding (libx264) and hardware encoding
(h264_nvenc on NVIDIA GPUs).

Requires the ``av`` package::

    pip install roomkit[video]
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.video.recorder.base import (
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
    safe_filename,
    validate_storage_path,
)

if TYPE_CHECKING:
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.recorder.pyav")


def _import_av() -> Any:
    """Import PyAV, raising a clear error if missing."""
    try:
        import av

        return av
    except ImportError as exc:
        raise ImportError(
            "av (PyAV) is required for PyAVVideoRecorder. Install with: pip install roomkit[video]"
        ) from exc


def _pick_codec(config: VideoRecordingConfig, av_mod: Any) -> str:
    """Pick the best available codec."""
    codec = config.codec
    if codec == "auto":
        try:
            av_mod.codec.Codec("h264_nvenc", "w")
            return "h264_nvenc"
        except Exception:
            return "libx264"
    return codec


class _ActiveRecording:
    __slots__ = ("container", "stream", "frame_count", "start_time", "path", "av")

    def __init__(
        self,
        container: Any,
        stream: Any,
        path: str,
        av_mod: Any,
    ) -> None:
        self.container = container
        self.stream = stream
        self.path = path
        self.av = av_mod
        self.frame_count = 0
        self.start_time = time.monotonic()


class PyAVVideoRecorder(VideoRecorder):
    """Video recorder using PyAV (FFmpeg) for H.264/H.265 encoding.

    Produces compressed MP4 files — 10-50x smaller than raw OpenCV output.
    Supports NVIDIA hardware encoding (h264_nvenc) when available.
    """

    def __init__(self) -> None:
        self._av = _import_av()
        self._active: dict[str, _ActiveRecording] = {}

    @property
    def name(self) -> str:
        return "PyAVVideoRecorder"

    def start(self, session: VideoSession, config: VideoRecordingConfig) -> VideoRecordingHandle:
        rec_id = uuid4().hex[:12]
        safe_id = safe_filename(session.id[:16])
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        fmt = config.format if config.format != "mp4v" else "mp4"
        filename = f"{safe_id}_{ts}.{fmt}"

        storage = config.storage or os.path.join(os.getcwd(), "recordings")
        resolved = validate_storage_path(storage)
        path = os.path.join(resolved, filename)

        codec = _pick_codec(config, self._av)
        container = self._av.open(path, mode="w")
        try:
            stream = container.add_stream(codec, rate=int(config.fps))
            stream.pix_fmt = "yuv420p"
            # Dimensions set on first frame (lazy)
            stream.width = 0
            stream.height = 0
        except Exception:
            container.close()
            raise

        handle = VideoRecordingHandle(
            id=rec_id,
            session_id=session.id,
            started_at=datetime.now(UTC),
            path=path,
        )
        self._active[rec_id] = _ActiveRecording(
            container=container, stream=stream, path=path, av_mod=self._av
        )
        logger.info("Recording started: %s → %s (codec=%s)", rec_id, path, codec)
        return handle

    def stop(self, handle: VideoRecordingHandle) -> VideoRecordingResult:
        handle.state = "stopped"
        active = self._active.pop(handle.id, None)
        if active is None:
            return VideoRecordingResult(id=handle.id)

        # Flush encoder only if frames were written (codec is open)
        if active.frame_count > 0:
            for packet in active.stream.encode():
                active.container.mux(packet)
        active.container.close()

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

        av = active.av

        # Set dimensions on first frame
        if active.stream.width == 0:
            active.stream.width = frame.width
            active.stream.height = frame.height

        # Build av.VideoFrame from raw pixels
        if frame.codec == "raw_rgb24":
            av_frame = av.VideoFrame.from_ndarray(_to_ndarray(frame), format="rgb24")
        elif frame.codec == "raw_bgr24":
            av_frame = av.VideoFrame.from_ndarray(_to_ndarray(frame), format="bgr24")
        else:
            logger.debug("Skipping encoded frame (codec=%s)", frame.codec)
            return

        av_frame.pts = active.frame_count
        for packet in active.stream.encode(av_frame):
            active.container.mux(packet)
        active.frame_count += 1

    def close(self) -> None:
        for rec_id, active in list(self._active.items()):
            if active.frame_count > 0:
                for packet in active.stream.encode():
                    active.container.mux(packet)
            active.container.close()
            logger.info("Recording closed: %s", rec_id)
        self._active.clear()


def _to_ndarray(frame: VideoFrame) -> Any:
    """Convert VideoFrame raw bytes to numpy ndarray."""
    import numpy as np

    return np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
