"""Censor video filter — replaces frames when blocked labels are detected."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.filter")


@dataclass
class _CensorState:
    censoring: bool = False
    grace_remaining: int = 0
    logged_start: bool = False


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
        self._sessions: dict[str, _CensorState] = {}

    @property
    def name(self) -> str:
        return "censor"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        state = self._sessions.setdefault(context.session_id, _CensorState())
        # Check if any blocked label is in the latest vision result
        detected = {label.lower() for label in context.labels_detected}
        has_blocked = bool(detected & self._blocked_labels)

        if has_blocked:
            state.censoring = True
            state.grace_remaining = self._grace_frames
            context.censoring = True
            if not state.logged_start:
                matched = detected & self._blocked_labels
                logger.info("Censoring started: detected %s", matched)
                state.logged_start = True
        elif state.censoring:
            if state.grace_remaining > 0:
                state.grace_remaining -= 1
            else:
                state.censoring = False
                context.censoring = False
                state.logged_start = False
                logger.info("Censoring stopped: blocked labels cleared")

        context.censoring = state.censoring
        if not state.censoring:
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

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def reset(self) -> None:
        self._sessions.clear()

    def close(self) -> None:
        self.reset()
