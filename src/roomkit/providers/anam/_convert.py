"""PyAV frame conversion utilities for Anam provider."""

from __future__ import annotations

from typing import Any


def av_audio_to_pcm(av_frame: Any, np: Any) -> bytes:
    """Convert a PyAV AudioFrame to PCM int16 mono bytes.

    Handles both float32 (range -1..1) and int16 formats, and
    downmixes stereo (interleaved or planar) to mono.

    Args:
        av_frame: PyAV AudioFrame with ``to_ndarray()`` method.
        np: The numpy module (lazy-loaded).

    Returns:
        PCM int16 bytes (mono).
    """
    arr = av_frame.to_ndarray()
    is_int16 = arr.dtype == np.int16

    # Stereo downmix: interleaved (1, 2*N) → average pairs
    channels = getattr(getattr(av_frame, "layout", None), "channels", None)
    n_channels = len(channels) if channels else 1
    if n_channels >= 2:
        flat = arr.flatten()
        n = len(flat) // n_channels * n_channels
        # Use left channel only (faster than mean, avoids float promotion)
        mono = flat[:n:n_channels]
    else:
        mono = arr.flatten()

    if is_int16:
        return bytes(mono.astype(np.int16).tobytes())
    # float32/float64: range is -1.0 .. 1.0
    pcm_int16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
    return bytes(pcm_int16.tobytes())


def av_video_to_frame(av_frame: Any, sequence: int = 0) -> Any:
    """Convert a PyAV VideoFrame to a RoomKit VideoFrame.

    Args:
        av_frame: PyAV VideoFrame with ``to_ndarray(format=...)`` method.
        sequence: Frame sequence number.

    Returns:
        A :class:`~roomkit.video.video_frame.VideoFrame` instance.
    """
    from roomkit.video.video_frame import VideoFrame

    rgb = av_frame.to_ndarray(format="rgb24")
    return VideoFrame(
        data=rgb.tobytes(),
        codec="raw_rgb24",
        width=av_frame.width,
        height=av_frame.height,
        sequence=sequence,
    )
