"""Video vision (frame analysis) providers."""

from __future__ import annotations

from roomkit.video.vision.base import FaceDetection, VisionProvider, VisionResult
from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider
from roomkit.video.vision.screen_input import ScreenInputTools
from roomkit.video.vision.screen_tool import DescribeScreenTool, capture_screen_frame
from roomkit.video.vision.webcam_tool import (
    DescribeWebcamTool,
    ListWebcamsTool,
    WebcamInfo,
    capture_webcam_frame,
    list_webcams,
    save_frame,
)

__all__ = [
    "DescribeScreenTool",
    "DescribeWebcamTool",
    "ListWebcamsTool",
    "ScreenInputTools",
    "WebcamInfo",
    "FaceDetection",
    "GeminiVisionConfig",
    "GeminiVisionProvider",
    "MockVisionProvider",
    "OpenAIVisionConfig",
    "OpenAIVisionProvider",
    "VisionProvider",
    "VisionResult",
    "capture_screen_frame",
    "capture_webcam_frame",
    "list_webcams",
    "save_frame",
]
