"""VideoFilterProvider ABC — inspect/replace video frames in the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionResult


@dataclass
class FilterEvent:
    """An event emitted by a filter during frame processing.

    Filters append events to :attr:`FilterContext.events` when they
    detect something noteworthy (e.g. face touch, object detection).
    The pipeline engine exposes :meth:`~VideoPipeline.drain_events`
    so the channel layer can drain and dispatch them to hooks.
    """

    kind: str
    """Detection type identifier (e.g. ``"face_touch"``, ``"object"``)."""

    data: Any = None
    """Typed event payload (e.g. :class:`~roomkit.video.events.VideoDetectionEvent`)."""


@dataclass
class FilterContext:
    """Shared state passed to the filter on every frame.

    Updated by the pipeline engine when new vision results arrive.
    The filter uses this to make per-frame decisions without running
    its own detection — vision does the heavy lifting periodically,
    and the filter acts on every frame based on the latest result.
    """

    session_id: str = ""
    """Active session identifier.  Set by the pipeline engine before
    calling filters so that stateful filters can key per-session data.
    """

    last_vision_result: VisionResult | None = None
    """Most recent vision analysis result (updated every vision_interval_ms)."""

    labels_detected: set[str] = field(default_factory=set)
    """Flattened set of labels from the last vision result."""

    censoring: bool = False
    """Whether the filter is currently censoring frames."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary state for custom filter implementations."""

    events: list[FilterEvent] = field(default_factory=list)
    """Events emitted by filters during the current frame processing.

    Drained by the channel layer after each :meth:`process_inbound` call.
    Filters append :class:`FilterEvent` instances here; the channel
    dispatches them to the appropriate hook triggers.
    """


class VideoFilterProvider(ABC):
    """Inspect or replace video frames in the pipeline.

    The filter runs on **every** frame (must be fast — no ML, no I/O).
    It uses :class:`FilterContext` populated by periodic vision analysis
    to decide whether to pass through or replace a frame.

    Typical use: censor frames when certain labels are detected
    (person, weapon, NSFW content).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""

    @abstractmethod
    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        """Process a frame, returning the original or a replacement.

        Args:
            frame: The current video frame (raw pixels after decode/resize).
            context: Shared state with latest vision results.

        Returns:
            The original frame to pass through unchanged, or a new
            ``VideoFrame`` to replace it (e.g., black/blurred image).
        """

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
