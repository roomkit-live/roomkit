"""VisionProvider abstract base class for video frame analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


@dataclass
class FaceDetection:
    """A detected face within a video frame."""

    x: int
    """Left coordinate in pixels."""

    y: int
    """Top coordinate in pixels."""

    width: int
    """Bounding box width in pixels."""

    height: int
    """Bounding box height in pixels."""

    confidence: float
    """Detection confidence (0.0-1.0)."""

    label: str | None = None
    """Optional identity label if face is recognized."""


@dataclass
class VisionResult:
    """Result from analyzing a video frame.

    Produced by :class:`VisionProvider` implementations.
    """

    description: str
    """Natural-language description of the frame content."""

    labels: list[str] = field(default_factory=list)
    """Detected object/scene labels."""

    confidence: float = 1.0
    """Overall analysis confidence (0.0-1.0)."""

    faces: list[FaceDetection] = field(default_factory=list)
    """Detected faces (empty if not applicable)."""

    text: str | None = None
    """OCR text extracted from the frame (useful for screen shares)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific extra data."""


DEFAULT_VISION_PROMPT = (
    "Describe what you see in this image briefly and precisely. "
    "Include key objects, people, actions, and any visible text."
)


class VisionProvider(ABC):
    """Abstract base class for video frame analysis.

    VisionProvider turns video frames into structured descriptions
    that can feed into AI conversation context.

    Implementations may call cloud vision APIs (OpenAI GPT-4o,
    Gemini Vision) or run local models (CLIP, YOLO, Tesseract).

    Example::

        provider = OpenAIVisionProvider(api_key="...")

        # Analyze a single frame
        result = await provider.analyze_frame(frame)
        print(result.description)  # "A person sitting at a desk..."

        # Stream analysis at intervals
        async for result in provider.analyze_stream(frames, interval_ms=2000):
            print(result.labels)
    """

    @property
    def name(self) -> str:
        """Provider name (e.g. 'openai-vision', 'gemini-vision', 'clip')."""
        return self.__class__.__name__

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming frame analysis."""
        return False

    @abstractmethod
    async def analyze_frame(self, frame: VideoFrame) -> VisionResult:
        """Analyze a single video frame.

        Args:
            frame: The video frame to analyze.  May be encoded (h264,
                vp8) or raw (rgb24, yuv420p) depending on the pipeline.
                Providers that require a specific format should document
                their requirements.

        Returns:
            VisionResult with description, labels, and optional faces/OCR.
        """
        ...

    async def analyze_stream(
        self,
        frames: AsyncIterator[VideoFrame],
        *,
        interval_ms: int = 2000,
        assumed_fps: float = 30.0,
    ) -> AsyncIterator[VisionResult]:
        """Analyze a stream of video frames at a given interval.

        The default implementation samples frames based on
        ``interval_ms`` and calls :meth:`analyze_frame` for each.
        Override for providers that support native streaming or
        batched analysis.

        Args:
            frames: Async iterator of video frames.
            interval_ms: Minimum interval between analyses in
                milliseconds.  Frames arriving faster are skipped.
            assumed_fps: Frame rate assumed for synthetic timestamps
                when frames lack ``timestamp_ms``.  Defaults to 30.

        Yields:
            VisionResult for each analyzed frame.
        """
        frame_interval_ms = 1000.0 / assumed_fps
        last_analyzed_ts: float = -float("inf")
        synthetic_ts: float = 0.0
        async for frame in frames:
            if frame.timestamp_ms is not None:
                ts = frame.timestamp_ms
            else:
                ts = synthetic_ts
                synthetic_ts += frame_interval_ms
            if ts - last_analyzed_ts >= interval_ms:
                last_analyzed_ts = ts
                yield await self.analyze_frame(frame)

    async def warmup(self) -> None:  # noqa: B027
        """Pre-load models so the first call is fast. Override in subclasses."""

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses if needed."""
