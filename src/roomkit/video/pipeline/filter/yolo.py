"""YOLO object detection filter — detects objects and updates FilterContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.filter")


def _load_ultralytics() -> Any:
    """Lazy-load ultralytics, raising a clear error if missing."""
    try:
        import ultralytics  # noqa: F811
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLODetectorFilter. Install with: pip install ultralytics"
        ) from exc
    return ultralytics


class YOLODetectorFilter(VideoFilterProvider):
    """Detect objects in video frames using a YOLO model.

    Runs YOLO inference on each frame (or every *N* frames for
    performance) and updates :class:`FilterContext` with detected
    labels and bounding boxes.  Does **not** modify the frame by
    default — downstream filters (e.g. :class:`CensorVideoFilter`)
    act on the detections.

    Args:
        model: YOLO model filename or path (default ``"yolo26n.pt"``).
        confidence: Minimum confidence threshold (0.0--1.0).
        classes: Optional set of class names to detect.  ``None``
            means detect all classes the model supports.
        every_n_frames: Run detection every N frames.  Intermediate
            frames reuse the most recent result.  Higher values
            reduce CPU/GPU load.
        draw_boxes: When ``True``, draw bounding boxes on the frame.
    """

    def __init__(
        self,
        *,
        model: str = "yolo26n.pt",
        confidence: float = 0.5,
        classes: set[str] | None = None,
        every_n_frames: int = 1,
        draw_boxes: bool = False,
    ) -> None:
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")
        if every_n_frames < 1:
            raise ValueError(f"every_n_frames must be >= 1, got {every_n_frames}")

        self._model_name = model
        self._confidence = confidence
        self._target_classes = {c.lower() for c in classes} if classes else None
        self._every_n = every_n_frames
        self._draw_boxes = draw_boxes

        self._model: Any = None
        self._frame_count = 0
        self._last_detections: list[dict[str, Any]] = []
        self._last_labels: set[str] = set()
        self._logged_first = False

    @property
    def name(self) -> str:
        return "yolo"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        self._frame_count += 1

        # Throttle: only run detection every N frames
        if self._frame_count % self._every_n != 1 and self._every_n > 1:
            self._apply_cached(context)
            return frame

        return self._detect(frame, context)

    def _detect(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        """Run YOLO inference on the frame and update context."""
        self._ensure_model()

        import numpy as np

        # Convert raw_rgb24 bytes to numpy array for YOLO
        img = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)

        results = self._model(img, conf=self._confidence, verbose=False)
        detections = self._parse_results(results)

        self._last_detections = detections
        self._last_labels = {d["label"] for d in detections}

        self._apply_cached(context)

        if not self._logged_first and detections:
            logger.info(
                "YOLO first detection: model=%s, found %d objects: %s",
                self._model_name,
                len(detections),
                self._last_labels,
            )
            self._logged_first = True

        if self._draw_boxes and detections:
            return self._draw(frame, img, detections)

        return frame

    def _parse_results(self, results: Any) -> list[dict[str, Any]]:
        """Extract detection dicts from YOLO results."""
        detections: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = result.names[cls_id].lower()

                # Filter by target classes if specified
                if self._target_classes and label not in self._target_classes:
                    continue

                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(
                    {
                        "label": label,
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": round(conf, 4),
                    }
                )

        return detections

    def _apply_cached(self, context: FilterContext) -> None:
        """Apply cached detection results to the filter context."""
        context.labels_detected = self._last_labels.copy()
        context.metadata["detections"] = self._last_detections

    def _ensure_model(self) -> None:
        """Load the YOLO model on first use."""
        if self._model is not None:
            return
        ultralytics = _load_ultralytics()
        logger.info("Loading YOLO model: %s", self._model_name)
        self._model = ultralytics.YOLO(self._model_name)

    def _draw(
        self,
        frame: VideoFrame,
        img: Any,
        detections: list[dict[str, Any]],
    ) -> VideoFrame:
        """Draw bounding boxes on the frame and return a new VideoFrame."""
        import numpy as np

        from roomkit.video.video_frame import VideoFrame as VideoFrameType

        # Work on a copy to avoid mutating the original
        canvas = img.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]

            # Draw rectangle (green, 2px)
            canvas[y1:y2, x1 : min(x1 + 2, x2)] = [0, 255, 0]
            canvas[y1:y2, max(x2 - 2, x1) : x2] = [0, 255, 0]
            canvas[y1 : min(y1 + 2, y2), x1:x2] = [0, 255, 0]
            canvas[max(y2 - 2, y1) : y2, x1:x2] = [0, 255, 0]

            # Draw label background (top-left corner)
            text = f"{label} {conf:.2f}"
            text_h, text_w = 12, len(text) * 7
            lx2 = min(x1 + text_w, frame.width)
            ly2 = min(y1 + text_h, frame.height)
            canvas[y1:ly2, x1:lx2] = [0, 255, 0]

        return VideoFrameType(
            data=canvas.astype(np.uint8).tobytes(),
            codec=frame.codec,
            width=frame.width,
            height=frame.height,
            timestamp_ms=frame.timestamp_ms,
            keyframe=frame.keyframe,
            sequence=frame.sequence,
        )

    def reset(self) -> None:
        self._frame_count = 0
        self._last_detections = []
        self._last_labels = set()
        self._logged_first = False

    def close(self) -> None:
        self._model = None
