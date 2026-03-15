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

    # Stereo downmix: interleaved (1, 2*N) → take every other sample
    channels = getattr(getattr(av_frame, "layout", None), "channels", None)
    n_channels = len(channels) if channels else 1
    if n_channels >= 2:
        arr = arr.flatten()
        # Interleaved: L,R,L,R... → average pairs
        n = len(arr) // n_channels * n_channels
        arr = arr[:n].reshape(-1, n_channels).mean(axis=1)

    arr = arr.flatten()

    # Convert to int16 depending on source dtype
    if arr.dtype == np.int16:
        return bytes(arr.tobytes())
    # float32/float64: range is -1.0 .. 1.0
    pcm_int16 = np.clip(arr * 32768.0, -32768, 32767).astype(np.int16)
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
