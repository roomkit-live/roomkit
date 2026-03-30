"""Video support for RoomKit (video transport, vision AI)."""

from __future__ import annotations

from roomkit.video.ai_integration import setup_realtime_vision, setup_video_vision
from roomkit.video.avatar import AvatarProvider, MockAvatarProvider, WebSocketAvatarProvider
from roomkit.video.backends import (
    get_fastrtc_video_backend,
    get_rtp_video_backend,
    get_screen_capture_backend,
    get_sip_video_backend,
    get_websocket_video_backend,
)
from roomkit.video.backends.base import VideoBackend
from roomkit.video.backends.mock import MockVideoBackend, MockVideoCall
from roomkit.video.base import (
    VideoCapability,
    VideoChunk,
    VideoDisconnectCallback,
    VideoReceivedCallback,
    VideoSession,
    VideoSessionReadyCallback,
    VideoSessionState,
)
from roomkit.video.bridge import (
    BridgeVideoFrameFilter,
    BridgeVideoFrameProcessor,
    VideoBridge,
    VideoBridgeConfig,
)
from roomkit.video.events import BridgeVideoEvent, VideoDetectionEvent
from roomkit.video.pipeline import (
    MockVideoDecoderProvider,
    MockVideoResizerProvider,
    VideoDecoderProvider,
    VideoPipeline,
    VideoPipelineConfig,
    VideoResizerProvider,
)
from roomkit.video.recorder import (
    MockVideoRecorder,
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
)
from roomkit.video.video_frame import VideoFrame
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
    # Base types
    "VideoBackend",
    "VideoCapability",
    "VideoChunk",
    "VideoFrame",
    "VideoSession",
    "VideoSessionState",
    # Callback types
    "VideoDisconnectCallback",
    "VideoReceivedCallback",
    "VideoSessionReadyCallback",
    # Vision
    "DescribeScreenTool",
    "DescribeWebcamTool",
    "ListWebcamsTool",
    "ScreenInputTools",
    "WebcamInfo",
    "FaceDetection",
    "GeminiVisionConfig",
    "GeminiVisionProvider",
    "OpenAIVisionConfig",
    "OpenAIVisionProvider",
    "VisionProvider",
    "VisionResult",
    "capture_screen_frame",
    "capture_webcam_frame",
    "list_webcams",
    "save_frame",
    # Recording
    "MockVideoRecorder",
    "VideoRecorder",
    "VideoRecordingConfig",
    "VideoRecordingHandle",
    "VideoRecordingResult",
    # Bridge / Detection
    "BridgeVideoEvent",
    "VideoDetectionEvent",
    "BridgeVideoFrameFilter",
    "BridgeVideoFrameProcessor",
    "VideoBridge",
    "VideoBridgeConfig",
    # Mocks
    "MockVideoBackend",
    "MockVideoCall",
    "MockVisionProvider",
    # Pipeline
    "MockVideoDecoderProvider",
    "MockVideoResizerProvider",
    "VideoDecoderProvider",
    "VideoPipeline",
    "VideoPipelineConfig",
    "VideoResizerProvider",
    # Avatar
    "AvatarProvider",
    "MockAvatarProvider",
    "WebSocketAvatarProvider",
    # AI integration
    "setup_realtime_vision",
    "setup_video_vision",
    # Utilities
    "make_text_frame",
    # Lazy loaders
    "get_fastrtc_video_backend",
    "get_local_video_backend",
    "get_rtp_video_backend",
    "get_screen_capture_backend",
    "get_sip_video_backend",
    "get_websocket_video_backend",
]


def make_text_frame(
    text: str,
    **kwargs: object,
) -> VideoFrame:
    """Generate a VideoFrame with centered text on a solid background.

    Requires ``opencv-python-headless`` (``pip install roomkit[local-video]``).
    """
    from roomkit.video.utils import make_text_frame as _make

    return _make(text, **kwargs)  # ty: ignore[invalid-argument-type]


def get_local_video_backend() -> type:
    """Get LocalVideoBackend class (requires opencv-python-headless).

    Install with: ``pip install roomkit[local-video]``
    """
    from roomkit.video.backends.local import LocalVideoBackend

    return LocalVideoBackend
