"""Video event types for bridge hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
