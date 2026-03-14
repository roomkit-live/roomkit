"""PyAV muxing helpers — codec resolution, dimension probing, stream creation.

Extracted from pyav.py to keep each module focused.
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.recorder.base import MediaRecordingConfig, RecordingTrack

logger = logging.getLogger("roomkit.recorder.pyav")

ENCODED_VIDEO_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})

_H264_START_CODE_4 = b"\x00\x00\x00\x01"
_H264_START_CODE_3 = b"\x00\x00\x01"


def h264_annex_b(data: bytes) -> bytes:
    """Prepend Annex B start code if not already present (single NAL only)."""
    if data[:4] == _H264_START_CODE_4 or data[:3] == _H264_START_CODE_3:
        return data
    return _H264_START_CODE_4 + data


def import_av() -> Any:
    """Import PyAV, raising a clear error if missing."""
    try:
        import av

        return av
    except ImportError as exc:
        raise ImportError(
            "av (PyAV) is required for PyAVMediaRecorder. Install with: pip install roomkit[video]"
        ) from exc


def resolve_video_codec(codec: str) -> str:
    """Resolve codec name for the video encoder.

    The default is ``libx264`` which works everywhere and supports
    ``tune=zerolatency`` for immediate output.  Users can pass a
    specific encoder name (e.g. ``h264_nvenc``) if they prefer
    GPU encoding and have verified compatibility.
    """
    return codec or "libx264"


def compute_pts(
    timestamp_ms: float | None,
    t0_ms: float,
    rate: int,
    last_pts: int,
    fallback_pts: int,
) -> int:
    """Compute monotonically increasing PTS from timestamp or fallback."""
    if timestamp_ms is not None:
        elapsed_s = (timestamp_ms - t0_ms) / 1000.0
        pts = max(round(elapsed_s * rate), 0)
        return max(pts, last_pts + 1)
    return fallback_pts


def safe_mux(
    stream: Any,
    container: Any,
    frame: Any,
    mux_error_logged: list[bool],
    path: str,
    *,
    label: str = "",
) -> None:
    """Encode frame and mux packets; log first error then suppress."""
    try:
        for packet in stream.encode(frame):
            container.mux(packet)
    except Exception:
        if not mux_error_logged[0]:
            mux_error_logged[0] = True
            frame_pts = getattr(frame, "pts", "?")
            frame_sr = getattr(frame, "sample_rate", None)
            logger.error(
                "Mux failed [%s] pts=%s rate=%s for %s",
                label or "unknown",
                frame_pts,
                frame_sr,
                path,
                exc_info=True,
            )


def probe_encoded_dimensions(
    av_mod: Any,
    pending: list[tuple[bytes, float | None]],
    codec_name: str,
) -> tuple[int, int] | None:
    """Decode pending frames to learn video dimensions.

    H.264 may require multiple NAL units (SPS+PPS+IDR) before a
    frame is decoded, so we feed all pending data then flush.

    Returns (width, height) or None if probing fails.
    """
    if not pending or not codec_name:
        return None
    decoder = av_mod.CodecContext.create(codec_name, "r")
    try:
        for data, _ in pending:
            raw = h264_annex_b(data) if codec_name == "h264" else data
            try:
                for frame in decoder.decode(av_mod.Packet(raw)):
                    logger.debug(
                        "Probed %s dimensions: %dx%d",
                        codec_name,
                        frame.width,
                        frame.height,
                    )
                    return (frame.width, frame.height)
            except Exception:  # nosec B112 — probe is best-effort
                continue
        # Flush decoder — H.264 may buffer frames
        try:
            for frame in decoder.decode(None):
                logger.debug(
                    "Probed %s dimensions (flush): %dx%d",
                    codec_name,
                    frame.width,
                    frame.height,
                )
                return (frame.width, frame.height)
        except Exception:  # nosec B110 — probe is best-effort
            pass
        logger.debug(
            "Could not probe dimensions for %s after %d frames",
            codec_name,
            len(pending),
        )
        return None
    finally:
        del decoder


def create_stream(
    container: Any,
    track: RecordingTrack,
    config: MediaRecordingConfig,
) -> Any:
    """Add a stream to the container with known parameters."""
    if track.kind == "video":
        w = track.width or 640
        h = track.height or 480
        codec = resolve_video_codec(config.video_codec)
        try:
            stream = container.add_stream(codec, rate=config.video_fps)
            stream.pix_fmt = "yuv420p"
            stream.width = w
            stream.height = h
            # libx264 buffers frames by default — zerolatency forces
            # immediate output so the MP4 muxer sees video data before
            # audio advances (prevents EINVAL from interleave check).
            # NVENC/other HW encoders are already low-latency.
            if codec == "libx264":
                stream.options = {"tune": "zerolatency", "preset": "ultrafast"}
        except Exception:
            if codec != "libx264":
                logger.info(
                    "Codec %s failed, falling back to libx264",
                    codec,
                )
                stream = container.add_stream(
                    "libx264",
                    rate=config.video_fps,
                )
                stream.pix_fmt = "yuv420p"
                stream.width = w
                stream.height = h
                stream.options = {"tune": "zerolatency", "preset": "ultrafast"}
            else:
                raise
        return stream
    # audio
    rate = track.sample_rate or config.audio_sample_rate
    stream = container.add_stream(config.audio_codec, rate=rate)
    stream.layout = "mono"
    return stream
