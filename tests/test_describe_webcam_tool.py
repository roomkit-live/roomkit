"""Tests for DescribeWebcamTool, ListWebcamsTool, and capture_webcam_frame."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.webcam_tool import (
    LIST_TOOL_DEFINITION,
    LIST_TOOL_NAME,
    TOOL_DEFINITION,
    TOOL_NAME,
    DescribeWebcamTool,
    ListWebcamsTool,
    capture_webcam_frame,
    list_webcams,
    save_frame,
)

# ---------------------------------------------------------------------------
# capture_webcam_frame
# ---------------------------------------------------------------------------


def _make_mock_cv2(
    *,
    open_ok: bool = True,
    read_ok: bool = True,
    w: int = 100,
    h: int = 100,
) -> object:
    """Build a mock cv2 module with configurable camera behaviour."""
    bgr_frame = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = open_ok
    mock_cap.read.return_value = (read_ok, bgr_frame if read_ok else None)

    mock_mod = types.ModuleType("cv2")
    mock_mod.VideoCapture = MagicMock(return_value=mock_cap)  # type: ignore[attr-defined]
    mock_mod.COLOR_BGR2RGB = 4  # type: ignore[attr-defined]
    mock_mod.cvtColor = MagicMock(return_value=rgb_frame)  # type: ignore[attr-defined]
    return mock_mod


class TestCaptureWebcamFrame:
    def test_returns_none_when_cv2_missing(self) -> None:
        with patch.dict(sys.modules, {"cv2": None}):
            result = capture_webcam_frame(device=0)
        assert result is None

    def test_returns_frame_with_mocked_cv2(self) -> None:
        mock_mod = _make_mock_cv2(w=640, h=480)
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            frame = capture_webcam_frame(device=0)

        assert frame is not None
        assert frame.codec == "raw_rgb24"
        assert frame.width == 640
        assert frame.height == 480

    def test_returns_none_when_camera_not_opened(self) -> None:
        mock_mod = _make_mock_cv2(open_ok=False)
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            frame = capture_webcam_frame(device=0)
        assert frame is None

    def test_returns_none_when_read_fails(self) -> None:
        mock_mod = _make_mock_cv2(read_ok=False)
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            frame = capture_webcam_frame(device=0)
        assert frame is None

    def test_releases_camera_after_capture(self) -> None:
        mock_mod = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            capture_webcam_frame(device=0)
        mock_cap = mock_mod.VideoCapture.return_value  # type: ignore[union-attr]
        mock_cap.release.assert_called_once()

    def test_releases_camera_on_read_failure(self) -> None:
        mock_mod = _make_mock_cv2(read_ok=False)
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            capture_webcam_frame(device=0)
        mock_cap = mock_mod.VideoCapture.return_value  # type: ignore[union-attr]
        mock_cap.release.assert_called_once()

    def test_passes_device_to_video_capture(self) -> None:
        mock_mod = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            capture_webcam_frame(device=2)
        mock_mod.VideoCapture.assert_called_once_with(2)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# DescribeWebcamTool
# ---------------------------------------------------------------------------


class TestDescribeWebcamTool:
    def test_definition_equals_constant(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["test"]))
        assert tool.definition == TOOL_DEFINITION
        assert tool.definition["name"] == TOOL_NAME

    def test_definition_has_required_fields(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["test"]))
        defn = tool.definition
        assert "name" in defn
        assert "description" in defn
        assert "parameters" in defn
        assert defn["parameters"]["required"] == ["query"]
        assert "device" in defn["parameters"]["properties"]
        assert "save_path" in defn["parameters"]["properties"]

    async def test_analyze_returns_description(self) -> None:
        vision = MockVisionProvider(descriptions=["A document with text"])
        tool = DescribeWebcamTool(vision, device=0)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.webcam_tool.capture_webcam_frame",
            return_value=fake_frame,
        ):
            result = await tool.analyze("Read the text on this document")

        assert result == "A document with text"

    async def test_analyze_passes_query_as_prompt(self) -> None:
        vision = MockVisionProvider(descriptions=["result"])
        vision.analyze_frame = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(description="result"),
        )
        tool = DescribeWebcamTool(vision)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.webcam_tool.capture_webcam_frame",
            return_value=fake_frame,
        ):
            await tool.analyze("What object is the user holding?")

        vision.analyze_frame.assert_called_once_with(  # type: ignore[union-attr]
            fake_frame,
            prompt="What object is the user holding?",
        )

    async def test_analyze_returns_error_when_no_frame(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["x"]))
        with patch(
            "roomkit.video.vision.webcam_tool.capture_webcam_frame",
            return_value=None,
        ):
            result = await tool.analyze("What is visible?")
        assert "No webcam frame available" in result

    async def test_analyze_uses_device_override(self) -> None:
        vision = MockVisionProvider(descriptions=["from camera 2"])
        tool = DescribeWebcamTool(vision, device=0)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.webcam_tool.capture_webcam_frame",
            return_value=fake_frame,
        ) as mock_capture:
            await tool.analyze("What is this?", device=2)

        mock_capture.assert_called_once_with(2)

    async def test_analyze_uses_default_device_when_none(self) -> None:
        vision = MockVisionProvider(descriptions=["ok"])
        tool = DescribeWebcamTool(vision, device=3)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.webcam_tool.capture_webcam_frame",
            return_value=fake_frame,
        ) as mock_capture:
            await tool.analyze("Describe", device=None)

        mock_capture.assert_called_once_with(3)

    async def test_handler_routes_describe_webcam(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["x"]))
        tool.analyze = AsyncMock(return_value="desc")  # type: ignore[method-assign]

        result = await tool.handler("describe_webcam", {"query": "q"})
        assert result == "desc"

    async def test_handler_returns_unknown(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["x"]))
        result = await tool.handler("other", {})
        assert "Unknown tool" in result

    async def test_analyze_saves_frame_when_save_path_given(self, tmp_path: Path) -> None:
        vision = MockVisionProvider(descriptions=["saved"])
        tool = DescribeWebcamTool(vision, device=0)
        dest = tmp_path / "capture.jpg"

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with (
            patch(
                "roomkit.video.vision.webcam_tool.capture_webcam_frame",
                return_value=fake_frame,
            ),
            patch(
                "roomkit.video.vision.webcam_tool.save_frame",
            ) as mock_save,
        ):
            result = await tool.analyze("Read", save_path=str(dest))

        mock_save.assert_called_once_with(fake_frame, str(dest))
        assert "Image saved" in result

    async def test_analyze_does_not_save_when_no_path(self) -> None:
        vision = MockVisionProvider(descriptions=["no save"])
        tool = DescribeWebcamTool(vision, device=0)

        fake_frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with (
            patch(
                "roomkit.video.vision.webcam_tool.capture_webcam_frame",
                return_value=fake_frame,
            ),
            patch(
                "roomkit.video.vision.webcam_tool.save_frame",
            ) as mock_save,
        ):
            result = await tool.analyze("Describe")

        mock_save.assert_not_called()
        assert "Image saved" not in result

    async def test_handler_uses_default_query(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["x"]))
        tool.analyze = AsyncMock(return_value="desc")  # type: ignore[method-assign]

        await tool.handler("describe_webcam", {})
        tool.analyze.assert_called_once_with(  # type: ignore[union-attr]
            "Describe what you see through the webcam.",
            device=None,
            save_path=None,
        )

    async def test_handler_passes_device_to_analyze(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["x"]))
        tool.analyze = AsyncMock(return_value="desc")  # type: ignore[method-assign]

        await tool.handler(
            "describe_webcam",
            {"query": "look", "device": 2},
        )
        tool.analyze.assert_called_once_with(  # type: ignore[union-attr]
            "look",
            device=2,
            save_path=None,
        )

    async def test_handler_passes_save_path_to_analyze(self) -> None:
        tool = DescribeWebcamTool(MockVisionProvider(descriptions=["x"]))
        tool.analyze = AsyncMock(return_value="desc")  # type: ignore[method-assign]

        await tool.handler(
            "describe_webcam",
            {"query": "read", "save_path": "/tmp/shot.jpg"},
        )
        tool.analyze.assert_called_once_with(  # type: ignore[union-attr]
            "read",
            device=None,
            save_path="/tmp/shot.jpg",
        )


# ---------------------------------------------------------------------------
# list_webcams
# ---------------------------------------------------------------------------


def _make_mock_cv2_for_list(available_devices: list[int], w: int = 640, h: int = 480) -> object:
    """Build a mock cv2 module where only specific device indices are available."""

    def _make_cap(device: int) -> MagicMock:
        cap = MagicMock()
        cap.isOpened.return_value = device in available_devices
        cap.get.side_effect = lambda prop: {3: float(w), 4: float(h)}.get(prop, 0.0)
        cap.getBackendName.return_value = "MOCK"
        return cap

    mock_mod = types.ModuleType("cv2")
    mock_mod.VideoCapture = MagicMock(side_effect=_make_cap)  # type: ignore[attr-defined]
    mock_mod.CAP_PROP_FRAME_WIDTH = 3  # type: ignore[attr-defined]
    mock_mod.CAP_PROP_FRAME_HEIGHT = 4  # type: ignore[attr-defined]
    return mock_mod


class TestListWebcams:
    def test_returns_empty_when_cv2_missing(self) -> None:
        with patch.dict(sys.modules, {"cv2": None}):
            result = list_webcams()
        assert result == []

    def test_returns_empty_when_no_cameras(self) -> None:
        mock_mod = _make_mock_cv2_for_list([])
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            result = list_webcams(max_devices=3)
        assert result == []

    def test_finds_single_camera(self) -> None:
        mock_mod = _make_mock_cv2_for_list([0])
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            result = list_webcams(max_devices=3)
        assert len(result) == 1
        assert result[0].device == 0
        assert result[0].width == 640
        assert result[0].height == 480

    def test_finds_multiple_cameras(self) -> None:
        mock_mod = _make_mock_cv2_for_list([0, 2])
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            result = list_webcams(max_devices=5)
        assert len(result) == 2
        assert result[0].device == 0
        assert result[1].device == 2

    def test_respects_max_devices(self) -> None:
        mock_mod = _make_mock_cv2_for_list([0, 5])
        with patch.dict(sys.modules, {"cv2": mock_mod}):
            result = list_webcams(max_devices=3)
        # Device 5 is beyond max_devices=3, so only device 0 found
        assert len(result) == 1


# ---------------------------------------------------------------------------
# ListWebcamsTool
# ---------------------------------------------------------------------------


class TestListWebcamsTool:
    def test_definition_equals_constant(self) -> None:
        tool = ListWebcamsTool()
        assert tool.definition == LIST_TOOL_DEFINITION
        assert tool.definition["name"] == LIST_TOOL_NAME

    def test_list_returns_no_devices_message(self) -> None:
        tool = ListWebcamsTool()
        with patch("roomkit.video.vision.webcam_tool.list_webcams", return_value=[]):
            result = tool.list()
        assert "No webcam devices found" in result

    def test_list_returns_camera_info(self) -> None:
        from roomkit.video.vision.webcam_tool import WebcamInfo

        cameras = [WebcamInfo(device=0, name="Camera 0 (MOCK)", width=640, height=480)]
        tool = ListWebcamsTool()
        with patch("roomkit.video.vision.webcam_tool.list_webcams", return_value=cameras):
            result = tool.list()
        assert "1 camera(s)" in result
        assert "Device 0" in result

    async def test_handler_routes_list_webcams(self) -> None:
        tool = ListWebcamsTool()
        with patch.object(tool, "list", return_value="Found 1 camera(s):"):
            result = await tool.handler("list_webcams", {})
        assert "Found 1" in result

    async def test_handler_returns_unknown(self) -> None:
        tool = ListWebcamsTool()
        result = await tool.handler("other_tool", {})
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# save_frame
# ---------------------------------------------------------------------------


class TestSaveFrame:
    def test_saves_jpeg_to_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.jpg"
        frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.webcam_tool.frame_to_jpeg",
            return_value=b"\xff\xd8fake-jpeg",
        ):
            result = save_frame(frame, dest)

        assert result == dest.resolve()
        assert dest.read_bytes() == b"\xff\xd8fake-jpeg"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "capture.jpg"
        frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.webcam_tool.frame_to_jpeg",
            return_value=b"\xff\xd8jpeg",
        ):
            save_frame(frame, dest)

        assert dest.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        dest = str(tmp_path / "string_path.jpg")
        frame = VideoFrame(
            data=b"\x00" * (10 * 10 * 3),
            codec="raw_rgb24",
            width=10,
            height=10,
        )
        with patch(
            "roomkit.video.vision.webcam_tool.frame_to_jpeg",
            return_value=b"\xff\xd8jpeg",
        ):
            result = save_frame(frame, dest)

        assert result.exists()
