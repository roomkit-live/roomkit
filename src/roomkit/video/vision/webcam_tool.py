"""Reusable ``describe_webcam`` tool for AI voice/chat agents.

Captures a single frame from the local webcam and analyzes it with a
:class:`VisionProvider`, returning a natural-language description.

Useful for letting an agent see physical documents, objects, or anything
the user points their camera at.

Example with RealtimeVoiceChannel::

    from roomkit import DescribeWebcamTool, GeminiVisionConfig, GeminiVisionProvider

    vision = GeminiVisionProvider(GeminiVisionConfig(api_key="..."))
    webcam_tool = DescribeWebcamTool(vision)

    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=backend,
        tools=[webcam_tool.definition],
        tool_handler=webcam_tool.handler,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import VisionProvider
from roomkit.video.vision.encode import frame_to_jpeg

logger = logging.getLogger("roomkit.video.vision.webcam_tool")

TOOL_NAME = "describe_webcam"

TOOL_DEFINITION: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Look through the user's webcam right now and answer a specific visual "
        "question. Use this tool when the user shows you a document, object, or "
        "anything physical via their camera. NEVER guess — always look first. "
        "Ask a precise, detailed question for accurate results. "
        "Use list_webcams first to discover available devices."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A precise visual question about what the webcam sees. "
                    "Examples: 'Read the text on this document', "
                    "'What object is the user holding?', "
                    "'Describe the form fields visible'"
                ),
            },
            "device": {
                "type": "integer",
                "description": (
                    "Camera device index to capture from. "
                    "Use list_webcams to discover available devices. "
                    "Defaults to the device configured at init time."
                ),
            },
            "save_path": {
                "type": "string",
                "description": (
                    "File path to save the captured image as JPEG. "
                    "When provided, the frame is written to disk before "
                    "analysis. Parent directories are created automatically."
                ),
            },
        },
        "required": ["query"],
    },
}


def capture_webcam_frame(
    device: int = 0,
    *,
    warmup_frames: int = 5,
) -> VideoFrame | None:
    """Capture a single frame from the local webcam as a raw RGB :class:`VideoFrame`.

    Uses ``opencv-python-headless`` for cross-platform webcam capture.
    The camera is opened, a few warmup frames are discarded to let
    auto-exposure settle, then one frame is grabbed and the device
    is immediately released.

    Args:
        device: Camera device index (0 = default webcam).
        warmup_frames: Number of frames to discard before the real
            capture, giving the camera time to adjust exposure and
            white balance. Set to 0 to skip warmup.

    Returns:
        A ``VideoFrame`` with ``codec="raw_rgb24"``, or ``None`` on failure.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python-headless is required — pip install roomkit[local-video]")
        return None

    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            cap.release()
            logger.error("Cannot open camera device %d", device)
            return None

        # Discard initial frames so auto-exposure/white-balance settle.
        for _ in range(warmup_frames):
            cap.read()

        ret, bgr_frame = cap.read()
        cap.release()

        if not ret or bgr_frame is None:
            logger.error("Failed to read frame from camera device %d", device)
            return None

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        return VideoFrame(
            data=rgb_frame.tobytes(),
            codec="raw_rgb24",
            width=w,
            height=h,
        )
    except Exception:
        logger.exception("Failed to capture webcam frame")
        return None


def save_frame(frame: VideoFrame, path: str | Path) -> Path:
    """Save a :class:`VideoFrame` as a JPEG file.

    Args:
        frame: The video frame to save (raw RGB/BGR or encoded).
        path: Destination file path. Parent directories are created
            automatically. If the extension is not ``.jpg``/``.jpeg``,
            the file is still written as JPEG.

    Returns:
        The resolved :class:`Path` that was written.
    """
    dest = Path(path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(frame_to_jpeg(frame))
    logger.info("Saved webcam frame to %s", dest)
    return dest


# ---------------------------------------------------------------------------
# List available webcams
# ---------------------------------------------------------------------------


@dataclass
class WebcamInfo:
    """Information about a detected webcam device."""

    device: int
    """Device index."""

    name: str
    """Human-readable name (backend-reported or generic)."""

    width: int
    """Default capture width."""

    height: int
    """Default capture height."""


def list_webcams(*, max_devices: int = 10) -> list[WebcamInfo]:
    """Probe local camera devices and return available webcams.

    Iterates device indices 0..*max_devices*-1, attempts to open each
    with OpenCV, and returns info for those that succeed.

    Args:
        max_devices: Maximum number of device indices to probe.

    Returns:
        List of :class:`WebcamInfo` for each reachable camera.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python-headless is required — pip install roomkit[local-video]")
        return []

    cameras: list[WebcamInfo] = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
        cap.release()
        cameras.append(
            WebcamInfo(
                device=idx,
                name=f"Camera {idx} ({backend_name})",
                width=w,
                height=h,
            ),
        )
    return cameras


LIST_TOOL_NAME = "list_webcams"

LIST_TOOL_DEFINITION: dict[str, Any] = {
    "name": LIST_TOOL_NAME,
    "description": (
        "List available webcam/camera devices on the user's machine. "
        "Returns device index, name, and resolution for each camera found. "
        "Call this before describe_webcam if you need to choose a specific camera."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


class ListWebcamsTool:
    """Tool for AI agents to discover available webcam devices.

    Args:
        max_devices: Maximum device indices to probe (default 10).

    Example::

        list_tool = ListWebcamsTool()

        channel = RealtimeVoiceChannel(
            "voice",
            tools=[list_tool.definition],
            tool_handler=list_tool.handler,
        )
    """

    def __init__(self, *, max_devices: int = 10) -> None:
        self._max_devices = max_devices

    @property
    def definition(self) -> dict[str, Any]:
        """Tool definition dict for ``RealtimeVoiceChannel(tools=[...])``."""
        return LIST_TOOL_DEFINITION

    def list(self) -> str:
        """Probe cameras and return a human-readable summary."""
        cameras = list_webcams(max_devices=self._max_devices)
        if not cameras:
            return "No webcam devices found."
        lines = [f"Found {len(cameras)} camera(s):"]
        for cam in cameras:
            lines.append(f"  Device {cam.device}: {cam.name} — {cam.width}x{cam.height}")
        return "\n".join(lines)

    async def handler(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Tool handler for list_webcams."""
        if name != LIST_TOOL_NAME:
            return f"Unknown tool: {name}"

        logger.info("list_webcams()")
        result = self.list()
        logger.info("list_webcams result: %s", result[:200])
        return result


class DescribeWebcamTool:
    """Webcam vision tool for AI agents.

    Captures a fresh frame from the local webcam and analyzes it with
    a :class:`VisionProvider`, passing the query as a per-call prompt.

    Args:
        vision: Vision provider for frame analysis.
        device: Camera device index (0 = default webcam).

    Example::

        vision = GeminiVisionProvider(GeminiVisionConfig(api_key="..."))
        tool = DescribeWebcamTool(vision, device=0)

        channel = RealtimeVoiceChannel(
            "voice",
            tools=[tool.definition],
            tool_handler=tool.handler,
        )
    """

    def __init__(self, vision: VisionProvider, *, device: int = 0) -> None:
        self._vision = vision
        self._device = device

    @property
    def definition(self) -> dict[str, Any]:
        """Tool definition dict for ``RealtimeVoiceChannel(tools=[...])``."""
        return TOOL_DEFINITION

    async def analyze(
        self,
        query: str,
        *,
        device: int | None = None,
        save_path: str | None = None,
    ) -> str:
        """Capture a fresh webcam frame and analyze it with *query*.

        Args:
            query: Visual question to answer about the frame.
            device: Camera device index override. ``None`` uses the
                default configured at init time.
            save_path: Optional file path to save the captured frame
                as JPEG before analysis.
        """
        effective_device = device if device is not None else self._device
        frame = capture_webcam_frame(effective_device)
        if frame is None:
            return "No webcam frame available. Check that a camera is connected."

        save_msg = ""
        if save_path:
            try:
                saved = save_frame(frame, save_path)
                save_msg = f"\n\n[Image saved to {saved}]"
                logger.info("Frame saved to %s", saved)
            except Exception as exc:
                save_msg = f"\n\n[ERROR: Failed to save image to {save_path}: {exc}]"
                logger.warning("Failed to save frame to %s: %s", save_path, exc)

        result = await self._vision.analyze_frame(frame, prompt=query)
        description = result.description or "Could not analyze the webcam image."
        return description + save_msg

    async def handler(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Tool handler for describe_webcam."""
        if name != TOOL_NAME:
            return f"Unknown tool: {name}"

        query = str(arguments.get("query", "Describe what you see through the webcam."))
        raw_device = arguments.get("device")
        device = int(raw_device) if raw_device is not None else None
        raw_path = arguments.get("save_path")
        path = str(raw_path) if raw_path is not None else None
        logger.info(
            "describe_webcam(query='%s', device=%s, save_path=%s)",
            query[:100],
            device,
            path,
        )
        result = await self.analyze(query, device=device, save_path=path)
        logger.info("describe_webcam result: %s", result[:200])
        return result
