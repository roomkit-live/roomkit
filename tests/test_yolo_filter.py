"""Tests for YOLODetectorFilter — YOLO object detection video filter."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from roomkit.video.pipeline.filter.base import FilterContext
from roomkit.video.pipeline.filter.yolo import YOLODetectorFilter, _load_ultralytics
from roomkit.video.video_frame import VideoFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_frame(
    width: int = 64,
    height: int = 48,
    seq: int = 0,
) -> VideoFrame:
    """Create a small raw RGB24 frame for testing."""
    data = b"\x80" * (width * height * 3)
    return VideoFrame(
        data=data,
        codec="raw_rgb24",
        width=width,
        height=height,
        sequence=seq,
    )


def _mock_yolo_result(
    detections: list[dict[str, Any]],
    names: dict[int, str] | None = None,
) -> MagicMock:
    """Build a mock YOLO result object matching ultralytics API."""
    if names is None:
        names = {0: "person", 1: "car", 2: "dog"}

    result = MagicMock()
    result.names = names

    boxes = MagicMock()
    n = len(detections)

    cls_list = []
    conf_list = []
    xyxy_list = []
    for det in detections:
        cls_list.append(det["cls_id"])
        conf_list.append(det["conf"])
        xyxy_list.append(det["box"])

    # Make boxes iterable by length
    boxes.__len__ = lambda self: n
    boxes.cls = [MagicMock() for _ in range(n)]
    boxes.conf = [MagicMock() for _ in range(n)]
    boxes.xyxy = [MagicMock() for _ in range(n)]

    for i in range(n):
        boxes.cls[i].__int__ = lambda self, _i=i: cls_list[_i]
        boxes.conf[i].__float__ = lambda self, _i=i: conf_list[_i]
        boxes.xyxy[i].tolist = lambda _i=i: list(xyxy_list[_i])

    result.boxes = boxes
    return result


def _make_mock_model(results: list[list[dict[str, Any]]]) -> MagicMock:
    """Create a mock YOLO model that returns canned results.

    Args:
        results: List of detection lists, one per call to model().
    """
    call_idx = {"n": 0}

    def model_call(img: Any, conf: float = 0.5, verbose: bool = False) -> list[MagicMock]:
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx < len(results):
            return [_mock_yolo_result(results[idx])]
        return [_mock_yolo_result([])]

    model = MagicMock(side_effect=model_call)
    return model


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestYOLODetectorFilterInit:
    def test_default_params(self) -> None:
        filt = YOLODetectorFilter()
        assert filt.name == "yolo"
        assert filt._model_name == "yolo26n.pt"
        assert filt._confidence == 0.5
        assert filt._target_classes is None
        assert filt._every_n == 1
        assert filt._draw_boxes is False

    def test_custom_params(self) -> None:
        filt = YOLODetectorFilter(
            model="yolov8s.pt",
            confidence=0.7,
            classes={"person", "Car"},
            every_n_frames=3,
            draw_boxes=True,
        )
        assert filt._model_name == "yolov8s.pt"
        assert filt._confidence == 0.7
        assert filt._target_classes == {"person", "car"}
        assert filt._every_n == 3
        assert filt._draw_boxes is True

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be between"):
            YOLODetectorFilter(confidence=1.5)

    def test_invalid_every_n_frames(self) -> None:
        with pytest.raises(ValueError, match="every_n_frames must be >= 1"):
            YOLODetectorFilter(every_n_frames=0)


# ---------------------------------------------------------------------------
# Detection with mocked ultralytics
# ---------------------------------------------------------------------------


class TestYOLODetection:
    def _make_filter(self, **kwargs: Any) -> YOLODetectorFilter:
        """Create a filter with a pre-injected mock model."""
        filt = YOLODetectorFilter(**kwargs)
        return filt

    def _inject_model(
        self,
        filt: YOLODetectorFilter,
        results: list[list[dict[str, Any]]],
    ) -> MagicMock:
        """Inject a mock model into the filter."""
        model = _make_mock_model(results)
        filt._model = model
        return model

    def test_detects_objects(self) -> None:
        filt = self._make_filter()
        self._inject_model(
            filt,
            [
                [{"cls_id": 0, "conf": 0.9, "box": (10, 20, 50, 60)}],
            ],
        )

        ctx = FilterContext()
        frame = _raw_frame()
        result = filt.filter(frame, ctx)

        assert result is frame
        assert "person" in ctx.labels_detected
        assert len(ctx.metadata["detections"]) == 1
        det = ctx.metadata["detections"][0]
        assert det["label"] == "person"
        assert det["box"] == (10, 20, 50, 60)
        assert det["confidence"] == 0.9

    def test_multiple_detections(self) -> None:
        filt = self._make_filter()
        self._inject_model(
            filt,
            [
                [
                    {"cls_id": 0, "conf": 0.85, "box": (10, 20, 50, 60)},
                    {"cls_id": 1, "conf": 0.7, "box": (100, 100, 200, 200)},
                ],
            ],
        )

        ctx = FilterContext()
        filt.filter(_raw_frame(), ctx)

        assert ctx.labels_detected == {"person", "car"}
        assert len(ctx.metadata["detections"]) == 2

    def test_no_detections(self) -> None:
        filt = self._make_filter()
        self._inject_model(filt, [[]])

        ctx = FilterContext()
        filt.filter(_raw_frame(), ctx)

        assert ctx.labels_detected == set()
        assert ctx.metadata["detections"] == []


# ---------------------------------------------------------------------------
# Confidence threshold filtering
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_below_threshold_filtered_by_model(self) -> None:
        """Confidence filtering is done by YOLO (conf param), not post-hoc."""
        filt = YOLODetectorFilter(confidence=0.8)
        model = _make_mock_model(
            [
                # Model already filters by conf, so only high-conf results appear
                [{"cls_id": 0, "conf": 0.9, "box": (10, 20, 50, 60)}],
            ]
        )
        filt._model = model

        ctx = FilterContext()
        filt.filter(_raw_frame(), ctx)

        # Verify conf param was passed to model
        model.assert_called_once()
        call_kwargs = model.call_args
        assert call_kwargs[1]["conf"] == 0.8


# ---------------------------------------------------------------------------
# Class filtering
# ---------------------------------------------------------------------------


class TestClassFiltering:
    def test_filters_by_class(self) -> None:
        filt = YOLODetectorFilter(classes={"person"})
        model = _make_mock_model(
            [
                [
                    {"cls_id": 0, "conf": 0.9, "box": (10, 20, 50, 60)},
                    {"cls_id": 1, "conf": 0.8, "box": (100, 100, 200, 200)},
                ],
            ]
        )
        filt._model = model

        ctx = FilterContext()
        filt.filter(_raw_frame(), ctx)

        # Only "person" should be in results, "car" filtered out
        assert ctx.labels_detected == {"person"}
        assert len(ctx.metadata["detections"]) == 1
        assert ctx.metadata["detections"][0]["label"] == "person"

    def test_no_matching_classes(self) -> None:
        filt = YOLODetectorFilter(classes={"cat"})
        model = _make_mock_model(
            [
                [{"cls_id": 0, "conf": 0.9, "box": (10, 20, 50, 60)}],
            ]
        )
        filt._model = model

        ctx = FilterContext()
        filt.filter(_raw_frame(), ctx)

        assert ctx.labels_detected == set()
        assert ctx.metadata["detections"] == []


# ---------------------------------------------------------------------------
# Frame throttling (every_n_frames)
# ---------------------------------------------------------------------------


class TestEveryNFrames:
    def test_throttle_reuses_last_result(self) -> None:
        filt = YOLODetectorFilter(every_n_frames=3)
        model = _make_mock_model(
            [
                [{"cls_id": 0, "conf": 0.9, "box": (10, 20, 50, 60)}],
                [{"cls_id": 1, "conf": 0.8, "box": (100, 100, 200, 200)}],
            ]
        )
        filt._model = model

        ctx = FilterContext()

        # Frame 1: runs detection (frame_count=1, 1%3=1 -> runs)
        filt.filter(_raw_frame(seq=0), ctx)
        assert ctx.labels_detected == {"person"}
        assert model.call_count == 1

        # Frame 2: skipped (frame_count=2, 2%3=2 -> skip)
        filt.filter(_raw_frame(seq=1), ctx)
        assert ctx.labels_detected == {"person"}  # cached
        assert model.call_count == 1

        # Frame 3: skipped (frame_count=3, 3%3=0 -> skip)
        filt.filter(_raw_frame(seq=2), ctx)
        assert ctx.labels_detected == {"person"}  # cached
        assert model.call_count == 1

        # Frame 4: runs detection (frame_count=4, 4%3=1 -> runs)
        filt.filter(_raw_frame(seq=3), ctx)
        assert ctx.labels_detected == {"car"}
        assert model.call_count == 2

    def test_every_1_runs_every_frame(self) -> None:
        filt = YOLODetectorFilter(every_n_frames=1)
        model = _make_mock_model(
            [
                [{"cls_id": 0, "conf": 0.9, "box": (10, 20, 50, 60)}],
                [],
            ]
        )
        filt._model = model

        ctx = FilterContext()
        filt.filter(_raw_frame(seq=0), ctx)
        assert model.call_count == 1

        filt.filter(_raw_frame(seq=1), ctx)
        assert model.call_count == 2


# ---------------------------------------------------------------------------
# Draw boxes
# ---------------------------------------------------------------------------


class TestDrawBoxes:
    def test_draw_boxes_modifies_frame(self) -> None:
        pytest.importorskip("numpy")

        filt = YOLODetectorFilter(draw_boxes=True)
        model = _make_mock_model(
            [
                [{"cls_id": 0, "conf": 0.9, "box": (5, 5, 30, 30)}],
            ]
        )
        filt._model = model

        frame = _raw_frame(width=64, height=48)
        ctx = FilterContext()
        result = filt.filter(frame, ctx)

        # Should return a new frame (not the original)
        assert result is not frame
        assert result.width == frame.width
        assert result.height == frame.height
        assert result.codec == frame.codec
        assert result.sequence == frame.sequence

    def test_draw_boxes_false_returns_original(self) -> None:
        filt = YOLODetectorFilter(draw_boxes=False)
        model = _make_mock_model(
            [
                [{"cls_id": 0, "conf": 0.9, "box": (5, 5, 30, 30)}],
            ]
        )
        filt._model = model

        frame = _raw_frame()
        ctx = FilterContext()
        result = filt.filter(frame, ctx)

        assert result is frame


# ---------------------------------------------------------------------------
# Integration with CensorVideoFilter
# ---------------------------------------------------------------------------


class TestYOLOCensorIntegration:
    def test_yolo_feeds_censor(self) -> None:
        """YOLO updates labels_detected, CensorFilter reads it."""
        from roomkit.video.pipeline.filter.censor import CensorVideoFilter

        yolo = YOLODetectorFilter()
        model = _make_mock_model(
            [
                [{"cls_id": 0, "conf": 0.9, "box": (10, 20, 50, 60)}],
            ]
        )
        yolo._model = model

        censor = CensorVideoFilter(blocked_labels={"person"})

        ctx = FilterContext()
        frame = _raw_frame()

        # YOLO runs first, updates context
        frame = yolo.filter(frame, ctx)
        assert "person" in ctx.labels_detected

        # Censor reads context, censors the frame
        result = censor.filter(frame, ctx)
        assert result is not frame  # censored (replaced with black)
        assert result.data == b"\x00" * (frame.width * frame.height * 3)

    def test_yolo_no_person_no_censor(self) -> None:
        """When YOLO detects no person, censor passes through."""
        from roomkit.video.pipeline.filter.censor import CensorVideoFilter

        yolo = YOLODetectorFilter()
        model = _make_mock_model(
            [
                [{"cls_id": 1, "conf": 0.9, "box": (10, 20, 50, 60)}],
            ]
        )
        yolo._model = model

        censor = CensorVideoFilter(blocked_labels={"person"})

        ctx = FilterContext()
        frame = _raw_frame()

        frame = yolo.filter(frame, ctx)
        result = censor.filter(frame, ctx)
        assert result is frame  # not censored


# ---------------------------------------------------------------------------
# Reset and close
# ---------------------------------------------------------------------------


class TestYOLOLifecycle:
    def test_reset_clears_state(self) -> None:
        filt = YOLODetectorFilter()
        filt._frame_count = 10
        filt._last_detections = [{"label": "person"}]
        filt._last_labels = {"person"}
        filt._logged_first = True

        filt.reset()

        assert filt._frame_count == 0
        assert filt._last_detections == []
        assert filt._last_labels == set()
        assert filt._logged_first is False

    def test_close_clears_model(self) -> None:
        filt = YOLODetectorFilter()
        filt._model = MagicMock()

        filt.close()
        assert filt._model is None


# ---------------------------------------------------------------------------
# Lazy import error
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_import_error_message(self) -> None:
        with (
            patch.dict(sys.modules, {"ultralytics": None}),
            pytest.raises(ImportError, match="ultralytics is required"),
        ):
            _load_ultralytics()

    def test_ensure_model_raises_without_ultralytics(self) -> None:
        filt = YOLODetectorFilter()
        with (
            patch(
                "roomkit.video.pipeline.filter.yolo._load_ultralytics",
                side_effect=ImportError("ultralytics is required"),
            ),
            pytest.raises(ImportError, match="ultralytics is required"),
        ):
            filt._ensure_model()


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------


class TestYOLOName:
    def test_name(self) -> None:
        assert YOLODetectorFilter().name == "yolo"
