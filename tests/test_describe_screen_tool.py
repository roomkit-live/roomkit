"""Tests for DescribeScreenTool, locate_element, and capture_screen_frame."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.screen_tool import (
    DESCRIBE_SCREEN_TOOL,
    DescribeScreenTool,
    _parse_coordinates,
    capture_screen_frame,
)

# ---------------------------------------------------------------------------
# capture_screen_frame
# ---------------------------------------------------------------------------


def _make_mock_mss(monitors: list[dict[str, int]], shot_data: bytes, w: int, h: int) -> object:
    """Build a mock mss module with configurable monitors and screenshot."""
    mock_shot = MagicMock()
    mock_shot.rgb = shot_data
    mock_shot.width = w
    mock_shot.height = h

    mock_sct = MagicMock()
    mock_sct.monitors = monitors
    mock_sct.grab.return_value = mock_shot

    mock_mss_ctx = MagicMock()
    mock_mss_ctx.__enter__ = MagicMock(return_value=mock_sct)
    mock_mss_ctx.__exit__ = MagicMock(return_value=False)

    mock_mod = types.ModuleType("mss")
    mock_mod.mss = MagicMock(return_value=mock_mss_ctx)  # type: ignore[attr-defined]
    return mock_mod


class TestCaptureScreenFrame:
    def test_returns_none_when_mss_missing(self) -> None:
        with patch.dict(sys.modules, {"mss": None}):
            result = capture_screen_frame(monitor=1)
        assert result is None

    def test_returns_frame_with_mocked_mss(self) -> None:
        empty_mon: dict[str, int] = {}
        primary = {"left": 0, "top": 0, "width": 100, "height": 100}
        mock_mod = _make_mock_mss(
            monitors=[empty_mon, primary],
            shot_data=b"\x00" * (100 * 100 * 3),
            w=100,
            h=100,
        )
        with patch.dict(sys.modules, {"mss": mock_mod}):
            frame = capture_screen_frame(monitor=1)

        assert frame is not None
        assert frame.codec == "raw_rgb24"
        assert frame.width == 100
        assert frame.height == 100

    def test_returns_none_for_invalid_monitor(self) -> None:
        mock_mod = _make_mock_mss(
            monitors=[{}],
            shot_data=b"",
            w=0,
            h=0,
        )
        with patch.dict(sys.modules, {"mss": mock_mod}):
            frame = capture_screen_frame(monitor=5)
        assert frame is None


# ---------------------------------------------------------------------------
# _parse_coordinates
# ---------------------------------------------------------------------------


class TestParseCoordinates:
    def test_json_object(self) -> None:
        assert _parse_coordinates('{"x": 100, "y": 200}') == (100, 200)

    def test_json_with_whitespace(self) -> None:
        assert _parse_coordinates('  {"x": 50, "y": 75}  ') == (50, 75)

    def test_json_embedded_in_text(self) -> None:
        text = 'The element is at {"x": 300, "y": 450} on the screen.'
        assert _parse_coordinates(text) == (300, 450)

    def test_comma_separated(self) -> None:
        assert _parse_coordinates("The coordinates are 500, 600") == (500, 600)

    def test_negative_coords_returns_none(self) -> None:
        assert _parse_coordinates('{"x": -1, "y": -1}') is None

    def test_garbage_returns_none(self) -> None:
        assert _parse_coordinates("I don't know") is None

    def test_json_with_error_returns_none(self) -> None:
        text = '{"x": -1, "y": -1, "error": "not found"}'
        assert _parse_coordinates(text) is None


# ---------------------------------------------------------------------------
# DescribeScreenTool
# ---------------------------------------------------------------------------


class TestDescribeScreenTool:
    def test_definition_compat(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["test"]))
        assert tool.definition == DESCRIBE_SCREEN_TOOL

    def test_definitions_includes_both(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["test"]))
        names = [d["name"] for d in tool.definitions]
        assert "describe_screen" in names
        assert "locate_element" in names

    async def test_analyze_returns_description(self) -> None:
        vision = MockVisionProvider(descriptions=["A desktop with Chrome open"])
        tool = DescribeScreenTool(vision, monitor=1)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.screen_tool.capture_screen_frame",
            return_value=fake_frame,
        ):
            result = await tool.analyze("What browser is open?")

        assert result == "A desktop with Chrome open"

    async def test_analyze_passes_query_as_prompt(self) -> None:
        vision = MockVisionProvider(descriptions=["result"])
        vision.analyze_frame = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(description="result"),
        )
        tool = DescribeScreenTool(vision)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.screen_tool.capture_screen_frame",
            return_value=fake_frame,
        ):
            await tool.analyze("Where is the Chrome icon?")

        vision.analyze_frame.assert_called_once_with(  # type: ignore[union-attr]
            fake_frame,
            prompt="Where is the Chrome icon?",
        )

    async def test_analyze_returns_error_when_no_frame(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["x"]))
        with patch(
            "roomkit.video.vision.screen_tool.capture_screen_frame",
            return_value=None,
        ):
            result = await tool.analyze("What is on screen?")
        assert "No screen frame available" in result

    async def test_locate_returns_coordinates(self) -> None:
        vision = MockVisionProvider(descriptions=['{"x": 150, "y": 300}'])
        tool = DescribeScreenTool(vision)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.screen_tool.capture_screen_frame",
            return_value=fake_frame,
        ):
            result = await tool.locate("Chrome icon")

        data = json.loads(result)
        assert data["x"] == 150
        assert data["y"] == 300

    async def test_locate_returns_error_on_not_found(self) -> None:
        vision = MockVisionProvider(descriptions=["I cannot find it"])
        tool = DescribeScreenTool(vision)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.screen_tool.capture_screen_frame",
            return_value=fake_frame,
        ):
            result = await tool.locate("nonexistent element")

        data = json.loads(result)
        assert "error" in data

    async def test_handler_routes_locate_element(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["x"]))
        tool.locate = AsyncMock(return_value='{"x": 10, "y": 20}')  # type: ignore[method-assign]

        session = object()
        result = await tool.handler(session, "locate_element", {"element": "button"})  # type: ignore[arg-type]
        assert '"x"' in result
        tool.locate.assert_called_once_with("button")  # type: ignore[union-attr]

    async def test_handler_routes_describe_screen(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["x"]))
        tool.analyze = AsyncMock(return_value="desc")  # type: ignore[method-assign]

        session = object()
        result = await tool.handler(session, "describe_screen", {"query": "q"})  # type: ignore[arg-type]
        assert result == "desc"

    async def test_handler_returns_unknown(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["x"]))
        session = object()
        result = await tool.handler(session, "other", {})  # type: ignore[arg-type]
        assert "Unknown tool" in result
