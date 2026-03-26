"""Tests for SubtitleManager and subtitle_overlay."""

from __future__ import annotations

from unittest.mock import MagicMock

from roomkit.video.pipeline.overlay.base import OverlayPosition
from roomkit.video.pipeline.overlay.filter import OverlayFilter
from roomkit.video.pipeline.overlay.mock import MockOverlayRenderer
from roomkit.video.pipeline.overlay.subtitle import (
    SUBTITLE_OVERLAY_ID,
    SubtitleManager,
    subtitle_overlay,
)


def _mock_kit() -> MagicMock:
    """Build a mock RoomKit with hook registration."""
    kit = MagicMock()
    kit._registered_hooks: list = []

    def fake_hook(trigger, *, execution=None, name=None):
        def decorator(fn):
            kit._registered_hooks.append((trigger, fn))
            return fn

        return decorator

    kit.hook = fake_hook
    return kit


def _mock_filter() -> OverlayFilter:
    """Create an OverlayFilter with a MockOverlayRenderer (no cv2 needed)."""
    return OverlayFilter(renderers=[MockOverlayRenderer("text")])


class TestSubtitleManager:
    def test_creates_overlay_filter(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter())
        assert mgr.overlay_filter is not None
        assert isinstance(mgr.overlay_filter, OverlayFilter)

    def test_uses_provided_filter(self) -> None:
        kit = _mock_kit()
        custom_filter = OverlayFilter(renderers=[MockOverlayRenderer()])
        mgr = SubtitleManager(kit, overlay_filter=custom_filter)
        assert mgr.overlay_filter is custom_filter

    def test_adds_subtitle_overlay(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter())
        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.overlay_type == "text"
        assert ov.z_order == 100

    def test_registers_on_transcription_hook(self) -> None:
        kit = _mock_kit()
        SubtitleManager(kit, overlay_filter=_mock_filter())
        assert len(kit._registered_hooks) == 1
        trigger, _ = kit._registered_hooks[0]
        assert trigger.value == "on_transcription"

    async def test_hook_updates_overlay(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter())
        _, hook_fn = kit._registered_hooks[0]

        event = MagicMock()
        event.text = "Bonjour le monde"
        ctx = MagicMock()
        await hook_fn(event, ctx)

        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.content == "Bonjour le monde"

    async def test_hook_with_translation(self) -> None:
        kit = _mock_kit()

        async def translate(text: str) -> str:
            return f"[EN] {text}"

        mgr = SubtitleManager(kit, overlay_filter=_mock_filter(), translate_fn=translate)
        _, hook_fn = kit._registered_hooks[0]

        event = MagicMock()
        event.text = "Bonjour"
        await hook_fn(event, MagicMock())

        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.content == "[EN] Bonjour"

    async def test_max_lines_rolling_window(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter(), max_lines=2)
        _, hook_fn = kit._registered_hooks[0]

        for text in ["Line 1", "Line 2", "Line 3"]:
            event = MagicMock()
            event.text = text
            await hook_fn(event, MagicMock())

        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.content == "Line 2\nLine 3"

    async def test_hook_reads_content_body_fallback(self) -> None:
        """When event.text is empty, falls back to event.content.body."""
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter())
        _, hook_fn = kit._registered_hooks[0]

        event = MagicMock()
        event.text = ""
        event.content = MagicMock()
        event.content.body = "Fallback text"
        await hook_fn(event, MagicMock())

        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.content == "Fallback text"

    async def test_hook_ignores_empty_text(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter())
        _, hook_fn = kit._registered_hooks[0]

        event = MagicMock()
        event.text = ""
        event.content = None
        await hook_fn(event, MagicMock())

        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.content == ""

    def test_clear(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter())
        mgr._lines = ["hello", "world"]
        mgr.clear()
        assert mgr._lines == []
        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.content == ""

    def test_set_text(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(kit, overlay_filter=_mock_filter())
        mgr.set_text("Manual subtitle")
        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.content == "Manual subtitle"

    def test_custom_position(self) -> None:
        kit = _mock_kit()
        mgr = SubtitleManager(
            kit, overlay_filter=_mock_filter(), position=OverlayPosition.TOP_CENTER
        )
        ov = mgr.overlay_filter.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.position == OverlayPosition.TOP_CENTER


class TestSubtitleOverlayFactory:
    def test_returns_overlay_filter(self) -> None:
        kit = _mock_kit()
        result = subtitle_overlay(kit, overlay_filter=_mock_filter())
        assert isinstance(result, OverlayFilter)
        assert result.get_overlay(SUBTITLE_OVERLAY_ID) is not None

    def test_passes_style(self) -> None:
        kit = _mock_kit()
        result = subtitle_overlay(
            kit, overlay_filter=_mock_filter(), font_scale=1.2, color=(255, 0, 0)
        )
        ov = result.get_overlay(SUBTITLE_OVERLAY_ID)
        assert ov is not None
        assert ov.style["font_scale"] == 1.2
        assert ov.style["color"] == (255, 0, 0)

    def test_passes_translate_fn(self) -> None:
        kit = _mock_kit()

        async def t(s: str) -> str:
            return s

        result = subtitle_overlay(kit, overlay_filter=_mock_filter(), translate_fn=t)
        assert result.get_overlay(SUBTITLE_OVERLAY_ID) is not None
