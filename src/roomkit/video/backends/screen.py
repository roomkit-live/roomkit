"""Screen capture video backend using mss.

Captures the screen (or a region) and delivers raw RGB frames
via the standard ``on_video_received`` callback.

Requires ``mss``::

    pip install roomkit[screen-capture]
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any
from uuid import uuid4

from roomkit.video.backends.base import VideoBackend
from roomkit.video.base import (
    VideoCapability,
    VideoChunk,
    VideoDisconnectCallback,
    VideoReceivedCallback,
    VideoSession,
    VideoSessionReadyCallback,
    VideoSessionState,
)
from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.screen")


def _import_mss() -> Any:
    """Import mss, raising a clear error if missing."""
    try:
        import mss

        return mss
    except ImportError as exc:
        raise ImportError(
            "mss is required for ScreenCaptureBackend. "
            "Install it with: pip install roomkit[screen-capture]"
        ) from exc


class ScreenCaptureBackend(VideoBackend):
    """Screen capture backend using mss.

    Captures frames from the screen (full monitor or a region) and fires
    ``on_video_received`` callbacks with raw RGB frames.
    """

    def __init__(
        self,
        *,
        monitor: int = 1,
        region: tuple[int, int, int, int] | None = None,
        fps: int = 5,
        scale: float = 1.0,
        diff_threshold: float = 0.0,
    ) -> None:
        if fps < 1:
            raise ValueError(f"fps must be >= 1, got {fps}")
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"scale must be in (0.0, 1.0], got {scale}")

        self._mss = _import_mss()
        self._monitor = monitor
        self._region = region
        self._fps = fps
        self._scale = scale
        self._diff_threshold = diff_threshold

        self._sessions: dict[str, VideoSession] = {}
        self._capture_threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}

        self._video_received_callbacks: list[VideoReceivedCallback] = []
        self._session_ready_callbacks: list[VideoSessionReadyCallback] = []
        self._disconnect_callbacks: list[VideoDisconnectCallback] = []

        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def name(self) -> str:
        return "ScreenCaptureBackend"

    @property
    def capabilities(self) -> VideoCapability:
        return VideoCapability.SCREEN_SHARE

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSession:
        session_id = uuid4().hex
        session_meta = {
            "monitor": self._monitor,
            "fps": self._fps,
            "scale": self._scale,
            "backend": self.name,
            **(metadata or {}),
        }
        session = VideoSession(
            id=session_id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VideoSessionState.ACTIVE,
            metadata=session_meta,
        )
        self._sessions[session_id] = session
        logger.info(
            "Screen capture session %s created (monitor=%d, fps=%d, scale=%.1f)",
            session_id[:8],
            self._monitor,
            self._fps,
            self._scale,
        )
        return session

    async def disconnect(self, session: VideoSession) -> None:
        await self.stop_capture(session)
        if session.id in self._sessions:
            self._sessions[session.id] = replace(session, state=VideoSessionState.ENDED)
        for cb in self._disconnect_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result
        logger.info("Screen capture session %s disconnected", session.id[:8])

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        # Screen capture is input-only — no outbound display.
        logger.debug("send_video called on ScreenCaptureBackend (no-op)")

    def get_session(self, session_id: str) -> VideoSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VideoSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        for session in list(self._sessions.values()):
            await self.stop_capture(session)
        self._sessions.clear()
        logger.info("ScreenCaptureBackend closed")

    # -------------------------------------------------------------------------
    # Capture control
    # -------------------------------------------------------------------------

    async def start_capture(self, session: VideoSession) -> None:
        """Start capturing the screen for a session."""
        if session.id in self._capture_threads:
            logger.warning("Capture already active for session %s", session.id[:8])
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        stop_event = threading.Event()
        self._stop_events[session.id] = stop_event

        thread = threading.Thread(
            target=self._capture_loop,
            args=(session, stop_event),
            name=f"screen-capture-{session.id[:8]}",
            daemon=True,
        )
        self._capture_threads[session.id] = thread
        thread.start()

        # Fire session ready callbacks
        for cb in self._session_ready_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result

        logger.info("Screen capture started for session %s", session.id[:8])

    async def stop_capture(self, session: VideoSession) -> None:
        """Stop capturing the screen for a session."""
        stop_event = self._stop_events.pop(session.id, None)
        if stop_event is not None:
            stop_event.set()

        thread = self._capture_threads.pop(session.id, None)
        if thread is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, thread.join, 3.0)
            if thread.is_alive():
                logger.warning(
                    "Capture thread for session %s did not stop in time",
                    session.id[:8],
                )

    # -------------------------------------------------------------------------
    # Capture thread
    # -------------------------------------------------------------------------

    def _capture_loop(
        self,
        session: VideoSession,
        stop_event: threading.Event,
    ) -> None:
        """Background thread: grab screen frames and fire callbacks at target FPS."""
        frame_interval = 1.0 / self._fps
        sequence = 0
        start_time = time.monotonic()
        prev_sample: bytes | None = None

        # Snapshot callbacks once — registrations happen before start_capture
        callbacks = list(self._video_received_callbacks)
        loop_ref = self._loop

        logger.debug("Screen capture loop started for session %s", session.id[:8])

        try:
            sct_ctx = self._mss.mss()
            sct = sct_ctx.__enter__()
        except Exception:
            logger.exception("Failed to initialize mss for session %s", session.id[:8])
            return

        try:
            monitor = self._resolve_monitor(sct)
        except RuntimeError:
            logger.exception("Monitor resolution failed for session %s", session.id[:8])
            sct_ctx.__exit__(None, None, None)
            return

        try:
            while not stop_event.is_set():
                frame_start = time.monotonic()

                screenshot = sct.grab(monitor)
                rgb_bytes = screenshot.rgb
                # Always use actual screenshot dimensions (differs from monitor
                # dict on HiDPI/Retina displays where pixels != logical size).
                width = screenshot.width
                height = screenshot.height

                # Optional scaling
                if self._scale < 1.0:
                    rgb_bytes, width, height = self._downscale(rgb_bytes, width, height)

                # Optional diff-based frame skipping
                skip, sample = self._check_diff(rgb_bytes, prev_sample)
                if skip:
                    self._throttle(frame_start, frame_interval, stop_event)
                    continue
                prev_sample = sample

                timestamp_ms = (time.monotonic() - start_time) * 1000.0

                video_frame = VideoFrame(
                    data=rgb_bytes,
                    codec="raw_rgb24",
                    width=width,
                    height=height,
                    timestamp_ms=timestamp_ms,
                    keyframe=(sequence == 0),
                    sequence=sequence,
                )
                sequence += 1

                for cb in callbacks:
                    if loop_ref is not None and loop_ref.is_running():
                        loop_ref.call_soon_threadsafe(cb, session, video_frame)
                    else:
                        cb(session, video_frame)

                self._throttle(frame_start, frame_interval, stop_event)
        finally:
            sct_ctx.__exit__(None, None, None)

        logger.debug("Screen capture loop ended for session %s", session.id[:8])

    def _resolve_monitor(self, sct: Any) -> dict[str, int]:
        """Resolve the monitor dict from config."""
        if self._region is not None:
            left, top, w, h = self._region
            return {"left": left, "top": top, "width": w, "height": h}

        if self._monitor >= len(sct.monitors):
            raise RuntimeError(
                f"Monitor {self._monitor} not found. "
                f"Available: {len(sct.monitors) - 1} monitors "
                f"(use 0 for all, 1..N for individual)"
            )
        monitor: dict[str, int] = sct.monitors[self._monitor]
        return monitor

    def _downscale(self, rgb_bytes: bytes, width: int, height: int) -> tuple[bytes, int, int]:
        """Downscale RGB bytes via numpy stride slicing."""
        import numpy as np

        step = max(1, int(1.0 / self._scale))
        arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
        arr = arr[::step, ::step, :].copy()
        return arr.tobytes(), arr.shape[1], arr.shape[0]

    # -------------------------------------------------------------------------
    # Diff-based frame skipping
    # -------------------------------------------------------------------------

    _SAMPLE_STRIDE = 300  # Sample every Nth byte for diff comparison

    def _check_diff(self, rgb_bytes: bytes, prev_sample: bytes | None) -> tuple[bool, bytes]:
        """Check if frame differs enough from previous and return the sample.

        Returns:
            (should_skip, current_sample) — skip=True means too similar.
        """
        sample = rgb_bytes[:: self._SAMPLE_STRIDE]

        if self._diff_threshold <= 0 or prev_sample is None:
            return False, sample

        if len(sample) != len(prev_sample):
            return False, sample

        total_diff = sum(abs(a - b) for a, b in zip(sample, prev_sample, strict=False))
        max_diff = len(sample) * 255
        pct = total_diff / max_diff if max_diff > 0 else 0.0
        return pct < self._diff_threshold, sample

    @staticmethod
    def _throttle(frame_start: float, frame_interval: float, stop_event: threading.Event) -> None:
        elapsed = time.monotonic() - frame_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            stop_event.wait(timeout=sleep_time)

    # -------------------------------------------------------------------------
    # Callback registration
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._video_received_callbacks.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)
