"""VideoChannel mixin — hook firing, framework events, and vision analysis."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from roomkit.models.enums import ChannelType, HookTrigger

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionProvider, VisionResult
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.video")


@runtime_checkable
class VideoHookHost(Protocol):
    """Contract: capabilities a host class must provide for VideoHooksMixin.

    Attributes provided by the host's ``__init__``:
        channel_id: Unique identifier for this channel instance.
        _framework: Reference to the RoomKit orchestrator (``None`` until
            the channel is registered).  The mixin accesses
            ``_framework._build_context``, ``_framework.hook_engine``,
            ``_framework._emit_framework_event``, ``_framework._store``,
            and ``_framework._channels``.
        _vision: Direct vision provider set on the channel (fallback when
            no video pipeline config provides one).
        _last_vision_results: Per-session cache of the most recent
            :class:`~roomkit.video.vision.base.VisionResult`.
        _last_vision_ts: Per-session timestamp (monotonic ms) of the last
            completed vision analysis — used for interval gating.

    Optional attributes accessed via ``getattr`` with fallbacks:
        _video_pipeline_config: Video pipeline config (may have ``.vision``).
        _pipeline: Audio pipeline config (may have ``.vision``).
        _video_pipeline: Active video pipeline (``update_filter_context``).
        channel_type: Channel type enum (defaults to ``ChannelType.VIDEO``).
    """

    channel_id: str
    _framework: RoomKit | None
    _vision: VisionProvider | None
    _last_vision_results: dict[str, VisionResult]
    _last_vision_ts: dict[str, float]


class VideoHooksMixin:
    """Hook-firing, framework events, and vision analysis for VideoChannel.

    Host contract: :class:`VideoHookHost`.
    """

    channel_id: str
    _framework: RoomKit | None
    _vision: VisionProvider | None
    _last_vision_results: dict[str, VisionResult]
    _last_vision_ts: dict[str, float]

    async def _fire_session_hook(
        self, trigger: HookTrigger, session: VideoSession | VoiceSession, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.models.session_event import SessionStartedEvent

            context = await self._framework._build_context(room_id)
            event = SessionStartedEvent(
                room_id=room_id,
                channel_id=self.channel_id,
                channel_type=getattr(self, "channel_type", ChannelType.VIDEO),
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
        self, event_type: str, session: VideoSession | VoiceSession, room_id: str
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

    @property
    def _vision_provider(self) -> VisionProvider | None:
        """Resolve the active vision provider.

        Checks video pipeline config first, then falls back to
        the direct ``_vision`` attribute set on the channel.
        Subclasses can override for custom resolution logic.
        """
        pipeline_cfg = getattr(self, "_video_pipeline_config", None) or getattr(
            self, "_pipeline", None
        )
        if pipeline_cfg is not None:
            vision = getattr(pipeline_cfg, "vision", None)
            if vision is not None:
                return vision
        return self._vision

    async def _analyze_frame(self, session: VideoSession, frame: VideoFrame, room_id: str) -> None:
        """Run vision analysis on a frame. Interval check done in caller."""
        vision = self._vision_provider
        if vision is None:
            return
        t0 = time.perf_counter()
        try:
            result = await vision.analyze_frame(frame)
        except Exception:
            logger.exception("Vision analysis failed for session %s", session.id)
            return
        finally:
            # Reset interval timer AFTER completion so the next interval
            # starts from when the API call finished, not when it started.
            # Use frame timestamp if available (matches the interval check
            # in _on_video_received), fall back to wall clock.
            ts = (
                frame.timestamp_ms if frame.timestamp_ms is not None else time.monotonic() * 1000.0
            )
            self._last_vision_ts[session.id] = ts
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Vision analysis: %.0fms (%s, session %s)",
            elapsed_ms,
            vision.name,
            session.id[:8],
        )

        self._last_vision_results[session.id] = result

        # Update pipeline filter context with latest vision result
        pipeline = getattr(self, "_video_pipeline", None)
        if pipeline is not None:
            pipeline.update_filter_context(session.id, result)

        if not self._framework or not result.description:
            return

        # Fire ON_VISION_RESULT sync hook — can block or modify the result
        context = await self._framework._build_context(room_id)
        from roomkit.models.vision_event import VisionEvent

        vision_event = VisionEvent(
            session=session,
            description=result.description,
            labels=result.labels,
            confidence=result.confidence,
            text=result.text,
            faces=result.faces,
            elapsed_ms=round(elapsed_ms),
        )
        hook_result = await self._framework.hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.ON_VISION_RESULT,
            vision_event,
            context,
            skip_event_filter=True,
        )
        if not hook_result.allowed:
            logger.info("Vision result blocked by hook: %s", hook_result.reason)
            return

        # Update result from hook (hooks may modify description)
        if isinstance(hook_result.event, VisionEvent):
            result = result.__class__(
                description=hook_result.event.description,
                labels=hook_result.event.labels,
                confidence=hook_result.event.confidence,
                text=hook_result.event.text,
                faces=hook_result.event.faces,
                metadata=result.metadata,
            )

        await self._inject_vision_event(session, result, room_id, elapsed_ms)
        await self._update_ai_vision_context(result, room_id)

    async def _update_ai_vision_context(self, result: VisionResult, room_id: str) -> None:
        """Auto-inject vision description into AI channels in the same room."""
        if not self._framework:
            return
        parts = [f"You can see a live camera feed. Current view: {result.description}"]
        if result.labels:
            parts.append(f"Objects detected: {', '.join(result.labels)}")
        if result.text:
            parts.append(f"Text visible: {result.text}")
        vision_context = "\n".join(parts)

        try:
            from roomkit.channels.ai import AIChannel

            bindings = await self._framework._store.list_bindings(room_id)
            for binding in bindings:
                ch = self._framework._channels.get(binding.channel_id)
                if not isinstance(ch, AIChannel):
                    continue
                meta = dict(binding.metadata)
                base = meta.get("_base_system_prompt")
                if base is None:
                    base = meta.get("system_prompt", "") or getattr(ch, "_system_prompt", "") or ""
                    meta["_base_system_prompt"] = base
                meta["system_prompt"] = f"{base}\n\n{vision_context}" if base else vision_context
                updated = binding.model_copy(update={"metadata": meta})
                await self._framework._store.update_binding(updated)
        except Exception:
            logger.debug("Failed to auto-inject vision context", exc_info=True)
