"""Local video backend using system webcam via OpenCV.

Requires ``opencv-python-headless``::

    pip install roomkit[local-video]
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any
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

if TYPE_CHECKING:
    import cv2 as _cv2_type

logger = logging.getLogger("roomkit.video.local")


def _import_cv2() -> Any:
    """Import OpenCV, raising a clear error if missing."""
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for LocalVideoBackend. "
            "Install it with: pip install roomkit[local-video]"
        ) from exc


class LocalVideoBackend(VideoBackend):
    """Local webcam video backend using OpenCV.

    Captures frames from a local camera and fires
    ``on_video_received`` callbacks with raw RGB frames.
    """

    def __init__(
        self,
        *,
        device: int = 0,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._cv2 = _import_cv2()
        self._device = device
        self._fps = fps
        self._width = width
        self._height = height

        self._sessions: dict[str, VideoSession] = {}
        self._captures: dict[str, _cv2_type.VideoCapture] = {}
        self._capture_threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}

        self._video_received_callbacks: list[VideoReceivedCallback] = []
        self._session_ready_callbacks: list[VideoSessionReadyCallback] = []
        self._disconnect_callbacks: list[VideoDisconnectCallback] = []

        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def name(self) -> str:
        return "LocalVideoBackend"

    @property
    def capabilities(self) -> VideoCapability:
        return VideoCapability.NONE

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
            "device": self._device,
            "fps": self._fps,
            "width": self._width,
            "height": self._height,
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
            "Video session %s created (device=%d, %dx%d@%dfps)",
            session_id[:8],
            self._device,
            self._width,
            self._height,
            self._fps,
        )
        return session

    async def disconnect(self, session: VideoSession) -> None:
        await self.stop_capture(session)
        if session.id in self._sessions:
            self._sessions[session.id] = replace(session, state=VideoSessionState.ENDED)
        logger.info("Video session %s disconnected", session.id[:8])

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        # Local backend does not support outbound video display (headless).
        # A future version could show frames via cv2.imshow().
        logger.debug("send_video called on LocalVideoBackend (no-op)")

    def get_session(self, session_id: str) -> VideoSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VideoSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        for session in list(self._sessions.values()):
            await self.stop_capture(session)
        self._sessions.clear()
        logger.info("LocalVideoBackend closed")

    # -------------------------------------------------------------------------
    # Capture control
    # -------------------------------------------------------------------------

    async def start_capture(self, session: VideoSession) -> None:
        """Start capturing video from the webcam for a session."""
        if session.id in self._captures:
            logger.warning("Capture already active for session %s", session.id[:8])
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        cap = self._cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self._device}. "
                "Check that a camera is connected and not in use."
            )

        cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(self._cv2.CAP_PROP_FPS, self._fps)

        # Read actual dimensions (camera may not support requested size)
        actual_w = int(cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(self._cv2.CAP_PROP_FPS)
        logger.info(
            "Camera opened: %dx%d @ %.1f fps (requested %dx%d @ %d fps)",
            actual_w,
            actual_h,
            actual_fps,
            self._width,
            self._height,
            self._fps,
        )

        self._captures[session.id] = cap
        stop_event = threading.Event()
        self._stop_events[session.id] = stop_event

        thread = threading.Thread(
            target=self._capture_loop,
            args=(session, cap, stop_event, actual_w, actual_h),
            name=f"video-capture-{session.id[:8]}",
            daemon=True,
        )
        self._capture_threads[session.id] = thread
        thread.start()

        # Fire session ready callbacks
        for cb in self._session_ready_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result

        logger.info("Capture started for session %s", session.id[:8])

    async def stop_capture(self, session: VideoSession) -> None:
        """Stop capturing and release the camera for a session."""
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

        cap = self._captures.pop(session.id, None)
        if cap is not None:
            cap.release()
            logger.info("Capture stopped for session %s", session.id[:8])

    # -------------------------------------------------------------------------
    # Capture thread
    # -------------------------------------------------------------------------

    def _capture_loop(
        self,
        session: VideoSession,
        cap: _cv2_type.VideoCapture,
        stop_event: threading.Event,
        width: int,
        height: int,
    ) -> None:
        """Background thread: read frames and fire callbacks at target FPS."""
        cv2 = self._cv2
        frame_interval = 1.0 / self._fps
        sequence = 0
        start_time = time.monotonic()

        # Snapshot callbacks once — registrations happen before start_capture
        callbacks = list(self._video_received_callbacks)
        loop_ref = self._loop

        logger.debug("Capture loop started for session %s", session.id[:8])

        while not stop_event.is_set():
            frame_start = time.monotonic()

            ret, bgr_frame = cap.read()
            if not ret:
                logger.warning("Camera read failed for session %s", session.id[:8])
                break

            # Convert BGR -> RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            frame_bytes = rgb_frame.tobytes()

            timestamp_ms = (time.monotonic() - start_time) * 1000.0

            video_frame = VideoFrame(
                data=frame_bytes,
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

            # Throttle to target FPS
            elapsed = time.monotonic() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                stop_event.wait(timeout=sleep_time)

        logger.debug("Capture loop ended for session %s", session.id[:8])

    # -------------------------------------------------------------------------
    # Callback registration
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._video_received_callbacks.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)
