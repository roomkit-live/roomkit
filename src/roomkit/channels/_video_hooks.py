"""VideoChannel mixin — hook firing, framework events, and vision analysis."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import ChannelType, HookTrigger

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionProvider, VisionResult

logger = logging.getLogger("roomkit.video")


class VideoHooksMixin:
    """Hook-firing, framework events, and vision analysis for VideoChannel."""

    channel_id: str
    _framework: RoomKit | None
    _vision: VisionProvider | None
    _last_vision_results: dict[str, Any]
    _session_bindings: dict[str, tuple[str, ChannelBinding]]

    async def _fire_session_hook(
        self, trigger: HookTrigger, session: VideoSession, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.models.session_event import SessionStartedEvent

            context = await self._framework._build_context(room_id)
            event = SessionStartedEvent(
                room_id=room_id,
                channel_id=self.channel_id,
                channel_type=ChannelType.VIDEO,
                participant_id=session.participant_id,
                session=session,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                trigger,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing %s hook", trigger.value)

    async def _emit_session_event(
        self, event_type: str, session: VideoSession, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            await self._framework._emit_framework_event(
                event_type,
                room_id=room_id,
                data={
                    "session_id": session.id,
                    "channel_id": self.channel_id,
                },
            )
        except Exception:
            logger.exception("Error emitting %s", event_type)

    async def _inject_vision_event(
        self,
        session: VideoSession,
        result: VisionResult,
        room_id: str,
        elapsed_ms: float = 0.0,
    ) -> None:
        """Emit a framework event with the vision analysis result."""
        if not self._framework:
            return
        await self._framework._emit_framework_event(
            "video_vision_result",
            room_id=room_id,
            data={
                "session_id": session.id,
                "channel_id": self.channel_id,
                "description": result.description,
                "labels": result.labels,
                "confidence": result.confidence,
                "text": result.text,
                "faces": len(result.faces),
                "elapsed_ms": round(elapsed_ms),
            },
        )

    async def _analyze_frame(self, session: VideoSession, frame: VideoFrame, room_id: str) -> None:
        """Run vision analysis on a frame. Interval check done in caller."""
        if self._vision is None:
            return
        t0 = time.perf_counter()
        try:
            result = await self._vision.analyze_frame(frame)
        except Exception:
            logger.exception("Vision analysis failed for session %s", session.id)
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Vision analysis: %.0fms (%s, session %s)",
            elapsed_ms,
            self._vision.name,
            session.id[:8],
        )

        self._last_vision_results[session.id] = result

        if self._framework and result.description:
            await self._inject_vision_event(session, result, room_id, elapsed_ms)
