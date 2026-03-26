"""Tests for ImageOverlayRenderer."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

from roomkit.video.pipeline.overlay.base import Overlay


def _build_mock_cv2() -> MagicMock:
    cv2 = MagicMock()
    cv2.IMREAD_UNCHANGED = -1
    cv2.INTER_AREA = 3

    # imdecode returns a 3-channel BGR image
    img = MagicMock()
    img.ndim = 3
    img.shape = (100, 200, 3)
    cv2.imdecode = MagicMock(return_value=img)

    # cvtColor returns an RGBA image
    rgba = MagicMock()
    rgba.shape = (100, 200, 4)
    rgba.__getitem__ = MagicMock(return_value=rgba)
    cv2.cvtColor = MagicMock(return_value=rgba)
    cv2.COLOR_BGR2RGBA = 1
    cv2.COLOR_BGRA2RGBA = 2
    cv2.COLOR_GRAY2RGBA = 3

    resized = MagicMock()
    resized.shape = (50, 100, 4)
    cv2.resize = MagicMock(return_value=resized)

    return cv2


def _build_mock_numpy() -> MagicMock:
    np_mod = MagicMock()
    np_mod.uint8 = "uint8"
    np_mod.float32 = "float32"
    buf = MagicMock()
    np_mod.frombuffer = MagicMock(return_value=buf)
    return np_mod


def _make_renderer():
    mock_cv2 = _build_mock_cv2()
    mock_np = _build_mock_numpy()

    with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
        importlib.invalidate_caches()
        mod = importlib.import_module("roomkit.video.pipeline.overlay.image")
        importlib.reload(mod)
        renderer = mod.ImageOverlayRenderer()

    return renderer, mock_cv2, mock_np


class TestImageOverlayRenderer:
    def test_overlay_type(self) -> None:
        renderer, _, _ = _make_renderer()
        assert renderer.overlay_type == "image"

    def test_skips_non_bytes_content(self) -> None:
        renderer, cv2, _ = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="i1", content="not bytes", overlay_type="image")
        result = renderer.render(canvas, overlay, 640, 480)
        assert result is canvas
        cv2.imdecode.assert_not_called()

    def test_skips_empty_bytes(self) -> None:
        renderer, cv2, _ = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="i1", content=b"", overlay_type="image")
        result = renderer.render(canvas, overlay, 640, 480)
        assert result is canvas

    def test_decodes_image(self) -> None:
        renderer, cv2, np_mod = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="i1", content=b"\x89PNG...", overlay_type="image")
        renderer.render(canvas, overlay, 640, 480)

        np_mod.frombuffer.assert_called_once()
        cv2.imdecode.assert_called_once()
        cv2.cvtColor.assert_called_once()

    def test_cache_invalidation(self) -> None:
        renderer, _, _ = _make_renderer()
        renderer._cache["i1"] = (0, MagicMock(), 100, 50)
        renderer.invalidate_cache("i1")
        assert "i1" not in renderer._cache

    def test_clear_cache(self) -> None:
        renderer, _, _ = _make_renderer()
        renderer._cache["a"] = (0, MagicMock(), 100, 50)
        renderer._cache["b"] = (0, MagicMock(), 100, 50)
        renderer.clear_cache()
        assert len(renderer._cache) == 0

    def test_decode_returns_none_on_failure(self) -> None:
        renderer, cv2, _ = _make_renderer()
        cv2.imdecode = MagicMock(return_value=None)
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="i1", content=b"bad data", overlay_type="image")
        result = renderer.render(canvas, overlay, 640, 480)
        assert result is canvas
