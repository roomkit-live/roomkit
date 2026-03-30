"""Video event types for hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass
class BridgeVideoEvent:
    """Video frame about to be forwarded via the bridge.

    Passed to ``BEFORE_BRIDGE_VIDEO`` hooks.  Return
    ``HookResult.block()`` to drop the frame, or
    ``HookResult.allow()`` to let it through.

    For frame *modification*, use
    :meth:`~roomkit.channels.video.VideoChannel.set_bridge_filter`
    which runs synchronously in the video callback thread.
    """

    session: VideoSession
    """The source video session producing this frame."""

    frame: VideoFrame
    """The video frame about to be forwarded."""

    room_id: str = ""
    """The room where the bridge is active."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the event was created."""


@dataclass
class VideoDetectionEvent:
    """Detection event emitted by video pipeline filters.

    Passed to ``ON_VIDEO_DETECTION`` hooks.  A single generic event
    type for all filter-originated detections — face touch, object
    detection, face recognition, motion, etc.  The :attr:`kind` field
    identifies the detection type and :attr:`metadata` carries
    kind-specific details.

    Example::

        @kit.hook(HookTrigger.ON_VIDEO_DETECTION)
        async def handle(event, ctx):
            if event.kind == "face_touch":
                zone = event.metadata.get("zone")
                print(f"Touch on {zone}, labels={event.labels}")
    """

    kind: str
    """Detection type identifier (e.g. ``"face_touch"``, ``"object"``)."""

    session: VideoSession | None = None
    """The video session that produced this detection.

    Set to ``None`` by pipeline filters (which don't have session access).
    The channel layer populates this when draining filter events.
    """

    labels: list[str] = field(default_factory=list)
    """Detected labels (e.g. ``["left_cheek", "chin"]`` or ``["person", "car"]``)."""

    confidence: float = 0.0
    """Overall detection confidence (0.0--1.0)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Kind-specific details (zone, hand, bbox, distance, etc.)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the detection occurred."""

    frame_sequence: int = 0
    """Frame sequence number for correlation."""
