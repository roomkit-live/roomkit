"""PyAV muxing helpers — codec resolution, dimension probing, stream creation.

Extracted from pyav.py to keep each module focused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from roomkit.recorder.base import MediaRecordingConfig, RecordingTrack

logger = logging.getLogger("roomkit.recorder.pyav")

ENCODED_VIDEO_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})

PCM_SAMPLE_FORMATS = {
    "pcm_s8": ("int8", "u8"),
    "pcm_u8": ("uint8", "u8"),
    "pcm_s16le": ("int16", "s16"),
    "pcm_s32le": ("int32", "s32"),
    "pcm_f32le": ("float32", "flt"),
}
"""How a track's declared PCM codec is read, and handed to PyAV.

The first element is the NumPy dtype the incoming bytes are in, the second the
sample format PyAV is given. They differ for signed 8-bit because FFmpeg has no
signed 8-bit sample format at all: ``pcm_s8`` is a codec, ``u8`` is the only
8-bit format a frame can be built in, and the two are one offset apart — so
those samples are shifted rather than mislabelled. Everything else is read as
what it says it is.
"""

_DEFAULT_AUDIO_CODEC = "pcm_s16le"
"""What an audio track that declares no codec is read as.

Every recording the framework opens declares one. A caller that does not gets
the format the rest of the framework carries, which is what this recorder
assumed of everyone before tracks began saying.
"""


def audio_layout(channels: int | None) -> str:
    """The channel layout of an audio track. Unstated means mono."""
    return "stereo" if channels == 2 else "mono"


@dataclass(frozen=True)
class AudioTrackFormat:
    """How one track's audio is read from its bytes and handed to PyAV.

    Resolved once for a track and held for its recording: a frame carries no
    format, so this is the same answer every time, and asking it per frame put
    a dictionary lookup and a log line on the delivery path.
    """

    dtype: str
    """NumPy dtype the incoming bytes are in."""

    sample_format: str
    """Sample format the frame handed to PyAV is built in."""

    layout: str
    channels: int

    @property
    def shifts_to_unsigned(self) -> bool:
        """Whether the samples move to reach their format, rather than the label.

        True for signed 8-bit alone: FFmpeg has no signed 8-bit sample format,
        so those samples are offset into unsigned rather than relabelled — which
        would read every one of them as its own complement.
        """
        return self.dtype == "int8"

    @property
    def buffer_dtype(self) -> str:
        """Dtype of the array PyAV is given, after any shift."""
        return "uint8" if self.shifts_to_unsigned else self.dtype

    @property
    def silence(self) -> int:
        """The sample value that is silence in this format — mid-scale if unsigned."""
        return 128 if self.sample_format == "u8" else 0


def audio_track_format(track: RecordingTrack) -> AudioTrackFormat:
    """Resolve how a track's audio is to be read, once for that track.

    An unknown codec falls back to 16-bit rather than refusing the track: every
    PCM name in circulation is 16-bit little-endian under a spelling this table
    does not have, and refusing would lose audio that is very probably fine. It
    is said in the log, once, because the fallback is a guess.
    """
    codec = track.codec or _DEFAULT_AUDIO_CODEC
    resolved = PCM_SAMPLE_FORMATS.get(codec)
    if resolved is None:
        logger.warning(
            "Track %s declares audio codec %r, which this recorder does not know; "
            "reading its samples as %s",
            track.id,
            track.codec,
            _DEFAULT_AUDIO_CODEC,
        )
        resolved = PCM_SAMPLE_FORMATS[_DEFAULT_AUDIO_CODEC]
    layout = audio_layout(track.channels)
    return AudioTrackFormat(
        dtype=resolved[0],
        sample_format=resolved[1],
        layout=layout,
        channels=2 if layout == "stereo" else 1,
    )


_H264_START_CODE_4 = b"\x00\x00\x00\x01"
_H264_START_CODE_3 = b"\x00\x00\x01"


def h264_annex_b(data: bytes) -> bytes:
    """Prepend Annex B start code if not already present.

    Expects a single NAL unit (as produced by RTP depacketization).
    Multi-NAL aggregation packets must be split before calling this.
    """
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
    """Compute monotonically increasing PTS from timestamp or fallback.

    Both the timestamp and fallback paths enforce monotonicity so that
    PTS never goes backward even when timestamps are missing or jittery.
    """
    if timestamp_ms is not None:
        elapsed_s = (timestamp_ms - t0_ms) / 1000.0
        pts = max(round(elapsed_s * rate), 0)
    else:
        pts = fallback_pts
    return max(pts, last_pts + 1)


def safe_mux(
    stream: Any,
    container: Any,
    frame: Any,
    track_state: Any,
    path: str,
    *,
    label: str = "",
) -> bool:
    """Encode frame and mux packets; log first error per track then suppress.

    Args:
        track_state: Object with a ``mux_error_logged`` bool attribute
            (typically ``_TrackState``).  Set to ``True`` after the
            first error to suppress subsequent log spam.

    Returns:
        ``True`` if all packets were muxed successfully, ``False`` on error.
    """
    try:
        for packet in stream.encode(frame):
            # Capture PTS/DTS before mux — FFmpeg takes ownership of the
            # packet buffer so reading fields after a failed mux is UB.
            pre_pts = packet.pts
            pre_dts = packet.dts
            pre_tb = packet.time_base
            try:
                container.mux(packet)
            except Exception:
                if not track_state.mux_error_logged:
                    track_state.mux_error_logged = True
                    logger.error(
                        "Mux failed [%s] frame_pts=%s rate=%s "
                        "pkt(pts=%s dts=%s tb=%s) frame_count=%s "
                        "stream_tb=%s for %s",
                        label or "unknown",
                        getattr(frame, "pts", "?"),
                        getattr(frame, "sample_rate", None),
                        pre_pts,
                        pre_dts,
                        pre_tb,
                        getattr(track_state, "frame_count", "?"),
                        stream.time_base,
                        path,
                        exc_info=True,
                    )
                return False
        return True
    except Exception:
        # Encode itself failed (not mux)
        if not track_state.mux_error_logged:
            track_state.mux_error_logged = True
            logger.error(
                "Encode failed [%s] frame_pts=%s rate=%s frame_count=%s for %s",
                label or "unknown",
                getattr(frame, "pts", "?"),
                getattr(frame, "sample_rate", None),
                getattr(track_state, "frame_count", "?"),
                path,
                exc_info=True,
            )
        return False


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
            # Each data blob is a single NAL from RTP depacketization
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


def _even(n: int) -> int:
    """Round down to nearest even number (libx264 requires even dimensions)."""
    return n & ~1


def create_stream(
    container: Any,
    track: RecordingTrack,
    config: MediaRecordingConfig,
) -> Any:
    """Add a stream to the container with known parameters."""
    if track.kind == "video":
        w = _even(track.width or 640)
        h = _even(track.height or 480)
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
    # From the track rather than fixed: a stereo track written into a mono
    # stream is not a downmix, it is interleaved samples read as twice the
    # audio at twice the speed, and nothing in the file says so.
    stream.layout = audio_layout(track.channels)
    return stream
