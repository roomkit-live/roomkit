"""Tests for TextOverlayRenderer."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

from roomkit.video.pipeline.overlay.base import Overlay, OverlayPosition


def _build_mock_cv2() -> MagicMock:
    cv2 = MagicMock()
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.LINE_AA = 16
    cv2.getTextSize = MagicMock(return_value=((100, 20), 5))
    cv2.putText = MagicMock()
    return cv2


def _build_mock_numpy() -> MagicMock:
    np_mod = MagicMock()
    # zeros returns an array-like mock with shape
    arr = MagicMock()
    arr.shape = (480, 640, 4)
    arr.__getitem__ = MagicMock(return_value=arr)
    arr.__setitem__ = MagicMock()
    arr.astype = MagicMock(return_value=arr)
    arr.tobytes = MagicMock(return_value=b"\x00" * (480 * 640 * 3))
    np_mod.zeros = MagicMock(return_value=arr)
    np_mod.uint8 = "uint8"
    np_mod.float32 = "float32"
    np_mod.array = MagicMock(return_value=arr)
    return np_mod


def _make_renderer():
    mock_cv2 = _build_mock_cv2()
    mock_np = _build_mock_numpy()

    with patch.dict(
        sys.modules,
        {"cv2": mock_cv2, "numpy": mock_np},
    ):
        importlib.invalidate_caches()
        mod = importlib.import_module("roomkit.video.pipeline.overlay.text")
        importlib.reload(mod)
        renderer = mod.TextOverlayRenderer()

    return renderer, mock_cv2, mock_np


class TestTextOverlayRenderer:
    def test_overlay_type(self) -> None:
        renderer, _, _ = _make_renderer()
        assert renderer.overlay_type == "text"

    def test_renders_text(self) -> None:
        renderer, cv2, np_mod = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="t1", content="Hello World")
        renderer.render(canvas, overlay, 640, 480)

        # putText should have been called during patch rendering
        cv2.putText.assert_called()

    def test_skips_empty_content(self) -> None:
        renderer, cv2, _ = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="t1", content="")
        result = renderer.render(canvas, overlay, 640, 480)
        assert result is canvas  # unchanged
        cv2.putText.assert_not_called()

    def test_skips_non_string_content(self) -> None:
        renderer, cv2, _ = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="t1", content=b"bytes")
        result = renderer.render(canvas, overlay, 640, 480)
        assert result is canvas

    def test_cache_invalidation(self) -> None:
        renderer, _, _ = _make_renderer()
        renderer._cache["t1"] = (0, MagicMock(), 100, 30)
        renderer.invalidate_cache("t1")
        assert "t1" not in renderer._cache

    def test_cache_clear_all(self) -> None:
        renderer, _, _ = _make_renderer()
        renderer._cache["a"] = (0, MagicMock(), 100, 30)
        renderer._cache["b"] = (0, MagicMock(), 100, 30)
        renderer.invalidate_cache("")
        assert len(renderer._cache) == 0

    def test_custom_style(self) -> None:
        renderer, cv2, np_mod = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(
            id="t1",
            content="Styled",
            style={"font_scale": 1.5, "color": (255, 0, 0)},
        )
        renderer.render(canvas, overlay, 640, 480)
        cv2.putText.assert_called()


class TestOverlayPosition:
    def test_all_positions_valid(self) -> None:
        from roomkit.video.pipeline.overlay.base import compute_position

        for pos in OverlayPosition:
            if pos == OverlayPosition.CUSTOM:
                x, y = compute_position(pos, 640, 480, 100, 30, custom_x=50, custom_y=50)
                assert x == 50
                assert y == 50
            else:
                x, y = compute_position(pos, 640, 480, 100, 30)
                assert 0 <= x <= 640
                assert 0 <= y <= 480

    def test_bottom_center(self) -> None:
        from roomkit.video.pipeline.overlay.base import compute_position

        x, y = compute_position(OverlayPosition.BOTTOM_CENTER, 640, 480, 100, 30, padding=10)
        assert x == (640 - 100) // 2
        assert y == 480 - 30 - 10
