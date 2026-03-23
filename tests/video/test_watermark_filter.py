"""Tests for WatermarkFilter (video/pipeline/filter/watermark.py)."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _build_mock_cv2() -> MagicMock:
    """Build a mock cv2 module with required constants."""
    cv2 = MagicMock()
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.LINE_AA = 16
    cv2.getTextSize.return_value = ((50, 10), 3)
    return cv2


def _build_mock_numpy() -> MagicMock:
    """Build a mock numpy module."""
    np = MagicMock()
    np.uint8 = "uint8"
    arr = MagicMock()
    arr.tobytes.return_value = b"\x00" * (4 * 4 * 3)
    arr.copy.return_value = arr
    np.frombuffer.return_value = MagicMock(reshape=MagicMock(return_value=arr))
    return np


class TestWatermarkFilter:
    def test_constructor_and_name(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.video.pipeline.filter.watermark")
            importlib.reload(mod)
            wm = mod.WatermarkFilter(text="TEST", position="top-left")
            assert wm.name == "watermark"

    def test_invalid_position_raises(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.video.pipeline.filter.watermark")
            importlib.reload(mod)
            with pytest.raises(ValueError, match="position must be one of"):
                mod.WatermarkFilter(position="invalid")

    def test_resolve_text_timestamp(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.video.pipeline.filter.watermark")
            importlib.reload(mod)
            wm = mod.WatermarkFilter(text="Time: {timestamp}")
            # Use a mock frame with .sequence
            frame = SimpleNamespace(sequence=42, codec="raw_rgb24", is_raw=True)
            result = wm._resolve_text(frame)
            assert "{timestamp}" not in result
            assert "Time:" in result

    def test_resolve_text_frame_number(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.video.pipeline.filter.watermark")
            importlib.reload(mod)
            wm = mod.WatermarkFilter(text="Frame {frame}")
            frame = SimpleNamespace(sequence=99, codec="raw_rgb24", is_raw=True)
            result = wm._resolve_text(frame)
            assert result == "Frame 99"

    def test_passthrough_for_non_raw_rgb24(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.video.pipeline.filter.watermark")
            importlib.reload(mod)
            wm = mod.WatermarkFilter()
            frame = SimpleNamespace(
                data=b"\x00" * 100,
                codec="h264",
                is_raw=False,
                width=10,
                height=10,
            )
            ctx = mod.FilterContext()
            result = wm.filter(frame, ctx)
            assert result is frame  # Should pass through unchanged
