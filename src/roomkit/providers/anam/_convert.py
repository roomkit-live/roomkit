"""PyAV frame conversion utilities for Anam provider."""

from __future__ import annotations

from typing import Any


def av_audio_to_pcm(av_frame: Any, np: Any) -> bytes:
    """Convert a PyAV AudioFrame to PCM int16 bytes.

    Args:
        av_frame: PyAV AudioFrame with ``to_ndarray()`` method.
        np: The numpy module (lazy-loaded).

    Returns:
        PCM int16 bytes (mono).
    """
    pcm_float = av_frame.to_ndarray()
    if pcm_float.ndim > 1:
        pcm_float = pcm_float.mean(axis=0)
    pcm_int16 = (np.clip(pcm_float * 32768.0, -32768, 32767)).astype(np.int16)
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
