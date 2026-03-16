"""Video vision (frame analysis) providers."""

from __future__ import annotations

from roomkit.video.vision.base import FaceDetection, VisionProvider, VisionResult
from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider
from roomkit.video.vision.screen_tool import DescribeScreenTool, capture_screen_frame

__all__ = [
    "DescribeScreenTool",
    "FaceDetection",
    "GeminiVisionConfig",
    "GeminiVisionProvider",
    "MockVisionProvider",
    "OpenAIVisionConfig",
    "OpenAIVisionProvider",
    "VisionProvider",
    "VisionResult",
    "capture_screen_frame",
]
