"""FastRTC audio+video backend using WebRTC/WebSocket transport.

Extends :class:`FastRTCVoiceBackend` with video transport. Each
:meth:`connect` call creates both a voice session and a video session
(shared session ID). Video frames flow through FastRTC's ``Stream``
with ``modality="audio-video"``.

Requires the ``fastrtc`` optional dependency::

    pip install roomkit[fastrtc]

Usage::

    from roomkit.video.backends.fastrtc import FastRTCVideoBackend, mount_fastrtc_av

    backend = FastRTCVideoBackend()
    channel = AudioVideoChannel("av", backend=backend, ...)
    kit.register_channel(channel)

    # Mount FastRTC A/V endpoints on FastAPI app (in lifespan)
    mount_fastrtc_av(app, backend, path="/av")
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

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
from roomkit.voice.auth import AuthCallback, auth_context
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend
from roomkit.voice.base import VoiceSession

if TYPE_CHECKING:
    import numpy as np
    from fastapi import FastAPI

logger = logging.getLogger("roomkit.video.fastrtc")

__all__ = ["FastRTCVideoBackend", "mount_fastrtc_av"]


class FastRTCVideoBackend(FastRTCVoiceBackend, VideoBackend):  # type: ignore[misc]
    """FastRTC backend for combined audio + video WebRTC transport.

    Extends :class:`FastRTCVoiceBackend` with a parallel video path.
    Audio-only users should use :class:`FastRTCVoiceBackend` directly.

    Args:
        video_width: Default video frame width.
        video_height: Default video frame height.
        video_queue_maxsize: Max pending frames per session video emit queue.
        **kwargs: Forwarded to :class:`FastRTCVoiceBackend`.
    """

    DEFAULT_VIDEO_QUEUE_MAXSIZE: int = 100

    def __init__(
        self,
        *,
        video_width: int = 640,
        video_height: int = 480,
        video_queue_maxsize: int = DEFAULT_VIDEO_QUEUE_MAXSIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._video_width = video_width
        self._video_height = video_height
        self._video_queue_maxsize = video_queue_maxsize

        # Video session tracking
        self._video_sessions: dict[str, VideoSession] = {}
        self._video_received_callback: VideoReceivedCallback | None = None
        self._video_taps: list[VideoReceivedCallback] = []
        self._video_session_ready_callbacks: list[VideoSessionReadyCallback] = []
        self._video_disconnect_callbacks: list[VideoDisconnectCallback] = []
        self._frame_sequences: dict[str, int] = {}

        # Outbound video emit queues: webrtc_id -> asyncio.Queue
        self._video_emit_queues: dict[str, asyncio.Queue[Any | None]] = {}

    @property
    def name(self) -> str:
        return "FastRTC-AV"

    @property
    def video_capabilities(self) -> VideoCapability:
        """Video-specific capabilities."""
        return VideoCapability.NONE

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def connect(  # type: ignore[override]
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        # Audio session via parent
        voice_session = await super().connect(
            room_id, participant_id, channel_id, metadata=metadata
        )

        # Create matching video session with same ID
        video_session = VideoSession(
            id=voice_session.id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VideoSessionState.ACTIVE,
            metadata={
                "width": self._video_width,
                "height": self._video_height,
                "backend": "fastrtc-av",
            },
        )
        self._video_sessions[voice_session.id] = video_session
        self._frame_sequences[voice_session.id] = 0

        logger.info(
            "A/V session created: session=%s, room=%s, participant=%s",
            voice_session.id,
            room_id,
            participant_id,
        )

        for cb in self._video_session_ready_callbacks:
            cb(video_session)

        return voice_session

    async def disconnect(self, session: VoiceSession) -> None:  # type: ignore[override]
        sid = session.id

        # Clean up video state
        video_session = self._video_sessions.pop(sid, None)
        self._frame_sequences.pop(sid, None)

        # Clean up video emit queue
        ws_id = session.metadata.get("websocket_id")
        if ws_id:
            self._video_emit_queues.pop(ws_id, None)

        if video_session is not None:
            video_session.state = VideoSessionState.ENDED
            for cb in self._video_disconnect_callbacks:
                cb(video_session)

        # Audio cleanup via parent
        await super().disconnect(session)

    async def close(self) -> None:
        self._video_sessions.clear()
        self._frame_sequences.clear()
        self._video_emit_queues.clear()
        await super().close()

    # -------------------------------------------------------------------------
    # Inbound video
    # -------------------------------------------------------------------------

    def _handle_video_frame(
        self,
        websocket_id: str,
        video_data: np.ndarray[Any, Any],
        width: int,
        height: int,
    ) -> None:
        """Called by FastRTC handler with raw video frame data.

        Converts numpy HWC array to VideoFrame and fires callbacks.
        """
        if self._video_received_callback is None and not self._video_taps:
            return

        session = self._find_session_by_websocket_id(websocket_id)
        if session is None:
            return

        video_session = self._video_sessions.get(session.id)
        if video_session is None:
            return

        # Convert numpy HWC (RGB) array to raw bytes
        frame_bytes = video_data.tobytes()

        seq = self._frame_sequences.get(session.id, 0)
        self._frame_sequences[session.id] = seq + 1

        frame = VideoFrame(
            data=frame_bytes,
            codec="raw_rgb24",
            width=width,
            height=height,
            timestamp_ms=time.monotonic() * 1000.0,
            keyframe=(seq == 0),
            sequence=seq,
        )

        if self._video_received_callback is not None:
            self._video_received_callback(video_session, frame)
        for tap in self._video_taps:
            tap(video_session, frame)

    # -------------------------------------------------------------------------
    # Outbound video
    # -------------------------------------------------------------------------

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        """Send video frames to a session.

        For WebRTC sessions, frames are queued for the handler's
        ``video_emit()`` method. WebSocket video is not currently
        supported (use WebSocketVideoBackend instead).
        """
        voice_session = self._sessions.get(session.id)
        if voice_session is None:
            logger.warning("send_video: no voice session for %s", session.id)
            return

        ws_id = voice_session.metadata.get("websocket_id")
        if not ws_id:
            logger.warning("send_video: no websocket_id for session %s", session.id)
            return

        queue = self._video_emit_queues.get(ws_id)
        if queue is None:
            logger.warning("send_video: no video emit queue for %s", session.id)
            return

        import numpy as _np

        if isinstance(video, bytes):
            # Store as numpy array with default dimensions
            arr = _np.frombuffer(video, dtype=_np.uint8)
            arr = arr.reshape((self._video_height, self._video_width, 3))
            self._enqueue_video_frame(queue, arr)
        else:
            async for chunk in video:
                arr = _np.frombuffer(chunk.data, dtype=_np.uint8)
                h = chunk.height
                w = chunk.width
                arr = arr.reshape((h, w, 3))
                self._enqueue_video_frame(queue, arr)

    @staticmethod
    def _enqueue_video_frame(
        queue: asyncio.Queue[Any | None], data: Any
    ) -> None:
        """Put a video frame into an emit queue, dropping oldest if full."""
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(data)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._video_received_callback = callback

    def add_video_tap(self, callback: VideoReceivedCallback) -> None:
        self._video_taps.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:  # type: ignore[override]
        """Register callback for video session ready events."""
        self._video_session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:  # type: ignore[override]
        self._video_disconnect_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Session queries
    # -------------------------------------------------------------------------

    def get_video_session(self, session_id: str) -> VideoSession | None:
        return self._video_sessions.get(session_id)

    def list_video_sessions(self, room_id: str) -> list[VideoSession]:
        return [s for s in self._video_sessions.values() if s.room_id == room_id]

    # -------------------------------------------------------------------------
    # WebRTC registration override (creates video emit queue too)
    # -------------------------------------------------------------------------

    def _register_webrtc(self, webrtc_id: str, session_id: str) -> None:
        """Register a WebRTC session and create audio + video emit queues."""
        super()._register_webrtc(webrtc_id, session_id)
        self._video_emit_queues[webrtc_id] = asyncio.Queue(
            maxsize=self._video_queue_maxsize
        )


def mount_fastrtc_av(
    app: FastAPI,
    backend: FastRTCVideoBackend,
    *,
    path: str = "/av",
    session_factory: Any = None,
    auth: AuthCallback | None = None,
    concurrency_limit: int | None = 1,
) -> None:
    """Mount FastRTC audio+video endpoints on a FastAPI app.

    Like :func:`mount_fastrtc_voice` but uses ``modality="audio-video"``
    so that the browser PeerConnection carries both audio and video tracks.

    Args:
        app: FastAPI application.
        backend: The FastRTCVideoBackend instance.
        path: Base path for A/V endpoints (default: /av).
        session_factory: Async callable(websocket_id) -> VoiceSession.
        auth: Optional async auth callback.
        concurrency_limit: Max concurrent connections (None for unlimited).
    """
    from fastrtc import AsyncStreamHandler, Stream

    backend._session_factory = session_factory  # type: ignore[attr-defined]

    class AVPassthroughHandler(AsyncStreamHandler):  # type: ignore[misc,unused-ignore]
        """Passes raw audio + video frames to the backend's callbacks.

        Each connection gets its own handler instance via ``copy()``.
        """

        def __init__(self) -> None:
            super().__init__()
            self._rejected = False
            self._auth_meta: dict[str, Any] | None = None
            self._webrtc_id: str | None = None
            self._is_webrtc = False

        def copy(self) -> AVPassthroughHandler:
            return AVPassthroughHandler()

        async def shutdown(self) -> None:
            """Called by FastRTC when the connection ends."""
            if self._webrtc_id:
                session = backend._find_session_by_websocket_id(self._webrtc_id)
                if session:
                    logger.info(
                        "A/V disconnected: session=%s webrtc_id=%s",
                        session.id,
                        self._webrtc_id,
                    )
                    for cb in backend._client_disconnected_callbacks:
                        try:
                            result = cb(session)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            logger.exception("Client disconnected callback error")
                    # disconnect() fires _video_disconnect_callbacks — no
                    # need to fire them here (would double-fire VoiceChannel's
                    # unbind_session).
                    await backend.disconnect(session)

        async def start_up(self) -> None:
            """Called once per connection — run auth and detect transport."""
            from fastrtc.utils import current_context

            ctx = current_context.get()
            if not ctx:
                return
            self._webrtc_id = ctx.webrtc_id
            self._is_webrtc = ctx.websocket is None

            if auth is not None and ctx.websocket is not None:
                try:
                    result = await auth(ctx.websocket)
                    if result is None:
                        self._rejected = True
                        logger.warning("Auth rejected for id=%s", self._webrtc_id)
                        return
                    self._auth_meta = result
                except Exception:
                    self._rejected = True
                    logger.exception("Auth error for id=%s", self._webrtc_id)
                    return

        async def receive(self, frame: tuple[int, Any]) -> None:
            """Handle inbound audio frames."""
            from fastrtc.utils import current_context

            if self._rejected:
                return

            sample_rate, audio_data = frame

            ctx = current_context.get()
            connection_id = ctx.webrtc_id if ctx else None
            websocket = ctx.websocket if ctx else None

            if not connection_id:
                return

            # Create session if not exists and we have a factory
            session = backend._find_session_by_websocket_id(connection_id)
            if not session and backend._session_factory:  # type: ignore[attr-defined]
                try:
                    token = auth_context.set(self._auth_meta)
                    try:
                        session = await backend._session_factory(connection_id)  # type: ignore[attr-defined]
                    finally:
                        auth_context.reset(token)
                    if session:
                        if websocket:
                            backend._register_websocket(connection_id, session.id, websocket)
                        else:
                            backend._register_webrtc(connection_id, session.id)
                except Exception:
                    logger.exception("Error creating session")

            if not session:
                return

            # Register connection if not already registered
            if session.id not in backend._websockets and "transport" not in session.metadata:
                if websocket:
                    backend._register_websocket(connection_id, session.id, websocket)
                else:
                    backend._register_webrtc(connection_id, session.id)

            backend._handle_audio_frame(connection_id, audio_data, sample_rate)

        async def video_receive(self, frame: tuple[Any, ...]) -> None:
            """Handle inbound video frames from FastRTC.

            FastRTC delivers video as ``(numpy_array_HWC,)`` — a single
            numpy array in Height x Width x Channels (RGB) layout.
            """
            if self._rejected or not self._webrtc_id:
                return

            if not frame:
                return

            video_data = frame[0]
            if video_data is None:
                return

            height, width = video_data.shape[:2]
            backend._handle_video_frame(self._webrtc_id, video_data, width, height)

        async def emit(self) -> tuple[int, Any] | None:
            """Return next audio frame for WebRTC playback."""
            if self._is_webrtc and self._webrtc_id:
                queue = backend._emit_queues.get(self._webrtc_id)
                if queue:
                    try:
                        return await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        return None
                await asyncio.sleep(0.1)
                return None

            await asyncio.sleep(0.1)
            return None

        async def video_emit(self) -> Any | None:
            """Return next video frame for WebRTC playback.

            Returns a numpy array in HWC (RGB) format, or None.
            """
            if not self._is_webrtc or not self._webrtc_id:
                await asyncio.sleep(0.1)
                return None

            queue = backend._video_emit_queues.get(self._webrtc_id)
            if not queue:
                await asyncio.sleep(0.1)
                return None

            try:
                return await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                return None

    # Create FastRTC stream with audio+video modality.
    effective_limit = concurrency_limit if concurrency_limit is not None else 2**31
    stream = Stream(
        handler=AVPassthroughHandler(),
        modality="audio-video",
        mode="send-receive",
        concurrency_limit=effective_limit,
    )

    backend._stream = stream

    stream.mount(app, path=path)
    logger.info("FastRTC A/V backend mounted at %s", path)
