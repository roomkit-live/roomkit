"""Tests for RichOverlayRenderer."""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import MagicMock, patch

from roomkit.video.pipeline.overlay.base import Overlay


def _build_mock_pillow() -> tuple[MagicMock, MagicMock, MagicMock]:
    image_mod = MagicMock()
    draw_mod = MagicMock()
    font_mod = MagicMock()

    img = MagicMock()
    image_mod.new = MagicMock(return_value=img)
    draw_mod.Draw = MagicMock(return_value=MagicMock())
    font_mod.truetype = MagicMock(return_value=MagicMock())
    font_mod.load_default = MagicMock(return_value=MagicMock())

    return image_mod, draw_mod, font_mod


def _build_mock_numpy() -> MagicMock:
    np_mod = MagicMock()
    np_mod.uint8 = "uint8"
    np_mod.float32 = "float32"

    arr = MagicMock()
    arr.shape = (100, 400, 4)
    arr.__getitem__ = MagicMock(return_value=arr)
    arr.__setitem__ = MagicMock()
    arr.astype = MagicMock(return_value=arr)
    np_mod.array = MagicMock(return_value=arr)
    return np_mod


def _make_renderer():
    image_mod, draw_mod, font_mod = _build_mock_pillow()
    mock_np = _build_mock_numpy()

    pil_mock = MagicMock()
    pil_mock.Image = image_mod
    pil_mock.ImageDraw = draw_mod
    pil_mock.ImageFont = font_mod

    with patch.dict(
        sys.modules,
        {
            "PIL": pil_mock,
            "PIL.Image": image_mod,
            "PIL.ImageDraw": draw_mod,
            "PIL.ImageFont": font_mod,
            "numpy": mock_np,
        },
    ):
        importlib.invalidate_caches()
        mod = importlib.import_module("roomkit.video.pipeline.overlay.rich")
        importlib.reload(mod)
        renderer = mod.RichOverlayRenderer()

    return renderer, image_mod, draw_mod, mock_np


class TestRichOverlayRenderer:
    def test_overlay_type(self) -> None:
        renderer, _, _, _ = _make_renderer()
        assert renderer.overlay_type == "rich"

    def test_skips_empty_content(self) -> None:
        renderer, image_mod, _, _ = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="r1", content="", overlay_type="rich")
        result = renderer.render(canvas, overlay, 640, 480)
        assert result is canvas
        image_mod.new.assert_not_called()

    def test_skips_non_string_content(self) -> None:
        renderer, image_mod, _, _ = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="r1", content=b"bytes", overlay_type="rich")
        result = renderer.render(canvas, overlay, 640, 480)
        assert result is canvas

    def test_renders_plain_text(self) -> None:
        renderer, image_mod, draw_mod, np_mod = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        overlay = Overlay(id="r1", content="Hello\nWorld", overlay_type="rich")
        renderer.render(canvas, overlay, 640, 480)

        image_mod.new.assert_called_once()
        draw_mod.Draw.assert_called_once()

    def test_renders_table(self) -> None:
        renderer, image_mod, draw_mod, np_mod = _make_renderer()
        canvas = MagicMock()
        canvas.shape = (480, 640, 3)

        table = json.dumps({"headers": ["Name", "Value"], "rows": [["A", "1"], ["B", "2"]]})
        overlay = Overlay(id="r1", content=table, overlay_type="rich")
        renderer.render(canvas, overlay, 640, 480)

        image_mod.new.assert_called_once()

    def test_cache_invalidation(self) -> None:
        renderer, _, _, _ = _make_renderer()
        renderer._cache["r1"] = (0, MagicMock(), 400, 100)
        renderer.invalidate_cache("r1")
        assert "r1" not in renderer._cache

    def test_clear_cache(self) -> None:
        renderer, _, _, _ = _make_renderer()
        renderer._cache["a"] = (0, MagicMock(), 400, 100)
        renderer._cache["b"] = (0, MagicMock(), 400, 100)
        renderer.clear_cache()
        assert len(renderer._cache) == 0

    def test_font_fallback(self) -> None:
        renderer, _, _, _ = _make_renderer()
        renderer._font_cls.truetype = MagicMock(side_effect=OSError)
        renderer._get_font(16)
        renderer._font_cls.load_default.assert_called_once()


class TestImportError:
    def test_helpful_message(self) -> None:
        import pytest

        with (
            patch.dict(
                sys.modules,
                {
                    "PIL": None,
                    "PIL.Image": None,
                    "PIL.ImageDraw": None,
                    "PIL.ImageFont": None,
                },
            ),
            pytest.raises(ImportError, match="pip install roomkit"),
        ):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.video.pipeline.overlay.rich")
            importlib.reload(mod)
            mod.RichOverlayRenderer()
