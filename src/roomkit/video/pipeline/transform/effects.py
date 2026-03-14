"""Built-in video effect transforms using OpenCV."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.transform.base import VideoTransformProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.transform")


def _lazy_imports() -> tuple[Any, Any]:
    """Lazy-import cv2 and numpy, raising clear errors."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for VideoEffectTransform. "
            "Install with: pip install opencv-python-headless"
        ) from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for VideoEffectTransform. Install with: pip install numpy"
        ) from exc
    return cv2, np


class VideoEffectTransform(VideoTransformProvider):
    """Apply visual effects to raw RGB24 video frames.

    Uses OpenCV (lazy-imported) for image processing. All effects
    operate on raw_rgb24 frames; other codecs pass through unchanged.

    Supported effects:
        - ``"grayscale"`` -- convert to grayscale (still RGB, 3 channels)
        - ``"sepia"`` -- warm brownish tone
        - ``"invert"`` -- negative image (255 - pixel)
        - ``"blur"`` -- Gaussian blur
        - ``"cartoon"`` -- edge detection + bilateral filter
        - ``"edges"`` -- Canny edge detection
        - ``"sketch"`` -- pencil sketch effect
        - ``"pixelate"`` -- pixelation (downscale + upscale)
    """

    def __init__(self, effect: str = "grayscale") -> None:
        self._dispatch: dict[str, Any] = {
            "grayscale": self._grayscale,
            "sepia": self._sepia,
            "invert": self._invert,
            "blur": self._blur,
            "cartoon": self._cartoon,
            "edges": self._edges,
            "sketch": self._sketch,
            "pixelate": self._pixelate,
        }
        if effect not in self._dispatch:
            raise ValueError(f"Unknown effect {effect!r}. Supported: {sorted(self._dispatch)}")
        self._effect = effect

    @property
    def name(self) -> str:
        return f"effect:{self._effect}"

    def transform(self, frame: VideoFrame) -> VideoFrame:
        """Apply the configured effect to a raw_rgb24 frame."""
        if frame.codec != "raw_rgb24":
            return frame

        cv2, np = _lazy_imports()

        w, h = frame.width, frame.height
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 3).copy()

        handler = self._dispatch[self._effect]
        result = handler(arr, cv2, np)

        from roomkit.video.video_frame import VideoFrame as VideoFrameModel

        return VideoFrameModel(
            data=result.tobytes(),
            codec="raw_rgb24",
            width=w,
            height=h,
            timestamp_ms=frame.timestamp_ms,
            keyframe=frame.keyframe,
            sequence=frame.sequence,
            metadata={**frame.metadata, "transform": self._effect},
        )

    # -- Effect implementations ------------------------------------------------
    # Each takes (arr: ndarray[H,W,3] RGB, cv2, np) -> ndarray[H,W,3] RGB.

    def _grayscale(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Convert to grayscale, keeping 3 channels for VideoFrame."""
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _sepia(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Apply warm brownish sepia tone."""
        # Sepia kernel expects BGR input
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        kernel = np.array(
            [
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189],
            ],
            dtype=np.float64,
        )
        sepia = cv2.transform(bgr, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sepia, cv2.COLOR_BGR2RGB)

    def _invert(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Negative image."""
        return (255 - arr).astype(np.uint8)

    def _blur(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Gaussian blur with (15, 15) kernel."""
        return cv2.GaussianBlur(arr, (15, 15), 0)

    def _cartoon(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Cartoon effect: bilateral filter + adaptive threshold edges."""
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=9,
        )
        color = cv2.bilateralFilter(bgr, d=9, sigmaColor=300, sigmaSpace=300)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_bgr)
        return cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

    def _edges(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Canny edge detection, output as 3-channel RGB."""
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def _sketch(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Pencil sketch effect via dodge blending."""
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blurred = cv2.GaussianBlur(inv, (21, 21), sigmaX=0)
        # Dodge blend: divide gray by inverted-blurred, scale to 256
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    def _pixelate(self, arr: Any, cv2: Any, np: Any) -> Any:
        """Pixelation: downscale to 1/10 then upscale back."""
        h, w = arr.shape[:2]
        small_w = max(1, w // 10)
        small_h = max(1, h // 10)
        small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
