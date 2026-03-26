"""Tests for OverlayFilter."""

from __future__ import annotations

from types import SimpleNamespace

from roomkit.video.pipeline.overlay.base import Overlay
from roomkit.video.pipeline.overlay.filter import OverlayFilter
from roomkit.video.pipeline.overlay.mock import MockOverlayRenderer


def _make_frame(codec: str = "raw_rgb24", width: int = 640, height: int = 480) -> SimpleNamespace:
    return SimpleNamespace(
        data=b"\x00" * (width * height * 3),
        codec=codec,
        is_raw=codec.startswith("raw_"),
        width=width,
        height=height,
        timestamp_ms=0.0,
        keyframe=True,
        sequence=0,
    )


def _make_context() -> SimpleNamespace:
    return SimpleNamespace(metadata={})


class TestOverlayFilterBasics:
    def test_name(self) -> None:
        f = OverlayFilter()
        assert f.name == "overlay"

    def test_passthrough_non_raw(self) -> None:
        f = OverlayFilter()
        frame = _make_frame(codec="h264")
        frame.is_raw = False
        result = f.filter(frame, _make_context())
        assert result is frame

    def test_passthrough_empty_overlays(self) -> None:
        f = OverlayFilter()
        frame = _make_frame()
        result = f.filter(frame, _make_context())
        assert result is frame

    def test_renders_overlay(self) -> None:
        renderer = MockOverlayRenderer("text")
        f = OverlayFilter(renderers=[renderer])
        f.add_overlay(Overlay(id="t1", content="hello"))

        frame = _make_frame()
        f.filter(frame, _make_context())

        assert renderer.render_count == 1
        assert renderer.last_overlay is not None
        assert renderer.last_overlay.content == "hello"

    def test_z_order_sorting(self) -> None:
        renderer = MockOverlayRenderer("text")
        f = OverlayFilter(renderers=[renderer])

        f.add_overlay(Overlay(id="high", content="top", z_order=10))
        f.add_overlay(Overlay(id="low", content="bottom", z_order=1))

        frame = _make_frame()
        f.filter(frame, _make_context())

        # Renderer called twice, last overlay should be the high z_order
        assert renderer.render_count == 2
        assert renderer.last_overlay is not None
        assert renderer.last_overlay.id == "high"

    def test_skips_unknown_renderer(self) -> None:
        f = OverlayFilter()  # no renderers registered
        f.add_overlay(Overlay(id="t1", content="hello", overlay_type="text"))

        frame = _make_frame()
        # Should not raise
        f.filter(frame, _make_context())


class TestOverlayManagement:
    def test_add_and_get(self) -> None:
        f = OverlayFilter()
        ov = Overlay(id="t1", content="hello")
        f.add_overlay(ov)
        assert f.get_overlay("t1") is ov

    def test_update_bumps_version(self) -> None:
        f = OverlayFilter()
        f.add_overlay(Overlay(id="t1", content="v0"))
        f.update_overlay("t1", content="v1")
        ov = f.get_overlay("t1")
        assert ov is not None
        assert ov.content == "v1"
        assert ov.version == 1

    def test_update_unknown_is_noop(self) -> None:
        f = OverlayFilter()
        f.update_overlay("nonexistent", content="x")  # should not raise

    def test_remove(self) -> None:
        f = OverlayFilter()
        f.add_overlay(Overlay(id="t1", content="x"))
        f.remove_overlay("t1")
        assert f.get_overlay("t1") is None

    def test_remove_unknown_is_noop(self) -> None:
        f = OverlayFilter()
        f.remove_overlay("nonexistent")  # should not raise

    def test_remove_invalidates_cache(self) -> None:
        renderer = MockOverlayRenderer("text")
        f = OverlayFilter(renderers=[renderer])
        f.add_overlay(Overlay(id="t1", content="x"))
        f.remove_overlay("t1")
        assert "t1" in renderer.invalidated

    def test_update_invalidates_cache(self) -> None:
        renderer = MockOverlayRenderer("text")
        f = OverlayFilter(renderers=[renderer])
        f.add_overlay(Overlay(id="t1", content="x"))
        f.update_overlay("t1", content="y")
        assert "t1" in renderer.invalidated

    def test_reset_clears_all(self) -> None:
        f = OverlayFilter()
        f.add_overlay(Overlay(id="t1", content="x"))
        f.add_overlay(Overlay(id="t2", content="y"))
        f.reset()
        assert f.get_overlay("t1") is None
        assert f.get_overlay("t2") is None

    def test_register_renderer(self) -> None:
        f = OverlayFilter()
        renderer = MockOverlayRenderer("custom")
        f.register_renderer(renderer)
        f.add_overlay(Overlay(id="c1", content="x", overlay_type="custom"))
        f.filter(_make_frame(), _make_context())
        assert renderer.render_count == 1


class TestComputePosition:
    def test_all_nine_positions(self) -> None:
        from roomkit.video.pipeline.overlay.base import OverlayPosition, compute_position

        expected = {
            OverlayPosition.TOP_LEFT: (10, 10),
            OverlayPosition.TOP_CENTER: (270, 10),
            OverlayPosition.TOP_RIGHT: (530, 10),
            OverlayPosition.CENTER_LEFT: (10, 225),
            OverlayPosition.CENTER: (270, 225),
            OverlayPosition.CENTER_RIGHT: (530, 225),
            OverlayPosition.BOTTOM_LEFT: (10, 440),
            OverlayPosition.BOTTOM_CENTER: (270, 440),
            OverlayPosition.BOTTOM_RIGHT: (530, 440),
        }
        for pos, (ex, ey) in expected.items():
            x, y = compute_position(pos, 640, 480, 100, 30, padding=10)
            assert (x, y) == (ex, ey), f"{pos}: got ({x},{y}), expected ({ex},{ey})"

    def test_custom_position(self) -> None:
        from roomkit.video.pipeline.overlay.base import OverlayPosition, compute_position

        x, y = compute_position(
            OverlayPosition.CUSTOM, 640, 480, 100, 30, custom_x=42, custom_y=99
        )
        assert (x, y) == (42, 99)

    def test_clamps_to_zero(self) -> None:
        from roomkit.video.pipeline.overlay.base import OverlayPosition, compute_position

        x, y = compute_position(OverlayPosition.BOTTOM_RIGHT, 50, 50, 200, 200, padding=10)
        assert x >= 0
        assert y >= 0


class TestFilterContextMetadata:
    def test_publishes_overlay_ids(self) -> None:
        renderer = MockOverlayRenderer("text")
        f = OverlayFilter(renderers=[renderer])
        f.add_overlay(Overlay(id="a", content="1", z_order=2))
        f.add_overlay(Overlay(id="b", content="2", z_order=1))

        ctx = _make_context()
        f.filter(_make_frame(), ctx)
        assert ctx.metadata["overlay_ids"] == ["b", "a"]  # sorted by z_order
