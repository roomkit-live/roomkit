"""Shared video frame encoding utilities for vision providers."""

from __future__ import annotations

import base64
import io

from roomkit.video.video_frame import ENCODED_CODECS, VideoFrame


def frame_to_jpeg(frame: VideoFrame) -> bytes:
    """Convert a VideoFrame to JPEG bytes.

    For raw codecs (rgb24, bgr24), uses OpenCV or Pillow to encode.
    For encoded codecs (h264, vp8, etc.), returns the raw bytes
    directly (the model may or may not support them).

    Raises:
        ValueError: If the raw codec is not supported for JPEG encoding.
        ImportError: If neither opencv-python-headless nor Pillow is installed.
    """
    if frame.codec in ENCODED_CODECS:
        return frame.data

    # Try OpenCV first (fast, already installed for LocalVideoBackend)
    try:
        import cv2
        import numpy as np

        if frame.codec == "raw_rgb24":
            pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 3
            )
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        elif frame.codec == "raw_bgr24":
            bgr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        else:
            raise ValueError(f"JPEG encoding for codec {frame.codec!r} not implemented")

        _, jpeg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg_buf.tobytes()
    except ImportError:
        pass

    # Fallback: Pillow
    try:
        from PIL import Image

        if frame.codec == "raw_rgb24":
            img = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
        elif frame.codec == "raw_bgr24":
            img = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
        else:
            raise ValueError(f"JPEG encoding for codec {frame.codec!r} not implemented")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        pass

    raise ImportError(
        "Either opencv-python-headless or Pillow is required to encode "
        "video frames as JPEG. Install with: pip install roomkit[local-video] "
        "or pip install Pillow"
    )


def frame_to_jpeg_base64(frame: VideoFrame) -> str:
    """Convert a VideoFrame to a base64-encoded JPEG string."""
    return base64.b64encode(frame_to_jpeg(frame)).decode("ascii")
