"""Tests for DescribeScreenTool and capture_screen_frame."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.screen_tool import (
    TOOL_DEFINITION,
    TOOL_NAME,
    DescribeScreenTool,
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
# DescribeScreenTool
# ---------------------------------------------------------------------------


class TestDescribeScreenTool:
    def test_definition_equals_constant(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["test"]))
        assert tool.definition == TOOL_DEFINITION
        assert tool.definition["name"] == TOOL_NAME

    def test_definition_has_required_fields(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["test"]))
        defn = tool.definition
        assert "name" in defn
        assert "description" in defn
        assert "parameters" in defn
        assert defn["parameters"]["required"] == ["query"]

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

    async def test_handler_uses_default_query(self) -> None:
        tool = DescribeScreenTool(MockVisionProvider(descriptions=["x"]))
        tool.analyze = AsyncMock(return_value="desc")  # type: ignore[method-assign]

        session = object()
        await tool.handler(session, "describe_screen", {})  # type: ignore[arg-type]
        tool.analyze.assert_called_once_with("Describe what is on this screen.")  # type: ignore[union-attr]
