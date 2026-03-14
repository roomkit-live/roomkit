"""Censor video filter — replaces frames when blocked labels are detected."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.filter")


class CensorVideoFilter(VideoFilterProvider):
    """Replace video frames when vision detects blocked content.

    Uses the latest vision result (via :class:`FilterContext`) to
    decide whether to censor.  When any label in ``blocked_labels``
    is detected, all frames are replaced with a black or solid-color
    image until vision reports the content is gone.

    Args:
        blocked_labels: Labels that trigger censoring (e.g., ``{"person", "weapon"}``).
        replacement: Replacement mode — ``"black"`` for solid black.
        grace_frames: Continue censoring for N frames after labels
            disappear, to avoid flicker between vision updates.
    """

    def __init__(
        self,
        blocked_labels: set[str],
        *,
        replacement: str = "black",
        grace_frames: int = 0,
    ) -> None:
        self._blocked_labels = {label.lower() for label in blocked_labels}
        self._replacement = replacement
        self._grace_frames = grace_frames
        self._grace_remaining = 0
        self._censoring = False
        self._logged_censor_start = False

    @property
    def name(self) -> str:
        return "censor"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        # Check if any blocked label is in the latest vision result
        detected = {label.lower() for label in context.labels_detected}
        has_blocked = bool(detected & self._blocked_labels)

        if has_blocked:
            self._censoring = True
            self._grace_remaining = self._grace_frames
            context.censoring = True
            if not self._logged_censor_start:
                matched = detected & self._blocked_labels
                logger.info("Censoring started: detected %s", matched)
                self._logged_censor_start = True
        elif self._censoring:
            if self._grace_remaining > 0:
                self._grace_remaining -= 1
            else:
                self._censoring = False
                context.censoring = False
                self._logged_censor_start = False
                logger.info("Censoring stopped: blocked labels cleared")

        if not self._censoring:
            return frame

        return self._make_replacement(frame)

    def _make_replacement(self, frame: VideoFrame) -> VideoFrame:
        """Create a replacement frame matching the original dimensions."""
        from roomkit.video.video_frame import VideoFrame

        w, h = frame.width, frame.height
        black_data = b"\x00" * (w * h * 3)
        return VideoFrame(
            data=black_data,
            codec="raw_rgb24",
            width=w,
            height=h,
            timestamp_ms=frame.timestamp_ms,
            keyframe=frame.keyframe,
            sequence=frame.sequence,
        )

    def reset(self) -> None:
        self._censoring = False
        self._grace_remaining = 0
        self._logged_censor_start = False
