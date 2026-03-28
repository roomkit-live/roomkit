"""WebSocket video backend for browser-based video streaming.

Provides a standalone VideoBackend that receives and sends video frames
over WebSocket connections. Designed for browser clients that stream
video frames separately from audio.

Wire protocol (binary messages)::

    [1 byte flags][4 bytes sequence_be][payload]

    Flags:
      - bit 0:   keyframe (1 = keyframe, 0 = delta)
      - bits 1-3: codec (0=h264, 1=vp8, 2=raw_rgb24, 3=reserved)

JSON control messages::

    {"type": "config", "codec": "h264", "width": 640, "height": 480}

Usage::

    from roomkit.video.backends.websocket import (
        WebSocketVideoBackend,
        mount_websocket_video,
    )

    backend = WebSocketVideoBackend()
    channel = VideoChannel("video", backend=backend, ...)
    kit.register_channel(channel)

    mount_websocket_video(app, backend, path="/video/ws")
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import time
from collections.abc import AsyncIterator, Callable
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
    from fastapi import FastAPI

logger = logging.getLogger("roomkit.video.websocket")

__all__ = ["WebSocketVideoBackend", "mount_websocket_video"]

# Binary header: 1 byte flags + 4 bytes sequence (big-endian)
_HEADER_STRUCT = struct.Struct(">BI")
_HEADER_SIZE = _HEADER_STRUCT.size

# Codec mapping: 3-bit field in flags → codec string
_CODEC_BY_ID: dict[int, str] = {
    0: "h264",
    1: "vp8",
    2: "raw_rgb24",
    3: "raw_rgb24",
}

_ID_BY_CODEC: dict[str, int] = {
    "h264": 0,
    "vp8": 1,
    "raw_rgb24": 2,
}

# Session factory type: connection_id -> VideoSession
SessionFactory = Callable[[str], Any]


class WebSocketVideoBackend(VideoBackend):
    """WebSocket-based video backend for browser frame streaming.

    Accepts WebSocket connections where video frames are exchanged as
    binary messages with a simple header. Control messages (codec
    negotiation, dimensions) use JSON.

    Args:
        default_width: Default frame width (used when not negotiated).
        default_height: Default frame height (used when not negotiated).
        default_codec: Default codec for outbound frames.
    """

    def __init__(
        self,
        *,
        default_width: int = 640,
        default_height: int = 480,
        default_codec: str = "h264",
    ) -> None:
        self._default_width = default_width
        self._default_height = default_height
        self._default_codec = default_codec

        # Session tracking
        self._sessions: dict[str, VideoSession] = {}
        # connection_id -> session_id
        self._connection_sessions: dict[str, str] = {}
        # session_id -> websocket
        self._websockets: dict[str, Any] = {}
        # Per-connection config (codec, width, height)
        self._connection_config: dict[str, dict[str, Any]] = {}

        # Callbacks
        self._video_received_callback: VideoReceivedCallback | None = None
        self._video_taps: list[VideoReceivedCallback] = []
        self._session_ready_callbacks: list[VideoSessionReadyCallback] = []
        self._disconnect_callbacks: list[VideoDisconnectCallback] = []
        self._session_factory: SessionFactory | None = None

        # Frame sequence counters for outbound
        self._outbound_sequences: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "WebSocketVideo"

    @property
    def capabilities(self) -> VideoCapability:
        return VideoCapability.NONE

    # -------------------------------------------------------------------------
    # Session factory
    # -------------------------------------------------------------------------

    def set_session_factory(self, factory: SessionFactory) -> None:
        """Set a factory called when a new WebSocket client connects.

        The factory receives a connection ID and should return a
        :class:`VideoSession` (may be async).
        """
        self._session_factory = factory

    # -------------------------------------------------------------------------
    # Session management
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
        session = VideoSession(
            id=session_id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VideoSessionState.ACTIVE,
            metadata={
                "transport": "websocket",
                "width": self._default_width,
                "height": self._default_height,
                "codec": self._default_codec,
                **(metadata or {}),
            },
        )
        self._sessions[session_id] = session
        self._outbound_sequences[session_id] = 0
        logger.info(
            "Video session created: session=%s, room=%s",
            session_id[:8],
            room_id,
        )
        return session

    async def disconnect(self, session: VideoSession) -> None:
        self._sessions.pop(session.id, None)
        self._websockets.pop(session.id, None)
        self._outbound_sequences.pop(session.id, None)
        session.state = VideoSessionState.ENDED

        # Clean up connection mapping
        to_remove = [cid for cid, sid in self._connection_sessions.items() if sid == session.id]
        for cid in to_remove:
            self._connection_sessions.pop(cid, None)
            self._connection_config.pop(cid, None)

        logger.info("Video session ended: session=%s", session.id[:8])

    def get_session(self, session_id: str) -> VideoSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VideoSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        for session in list(self._sessions.values()):
            await self.disconnect(session)
        self._connection_sessions.clear()
        self._connection_config.clear()

    # -------------------------------------------------------------------------
    # Video send
    # -------------------------------------------------------------------------

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        websocket = self._websockets.get(session.id)
        if websocket is None:
            logger.warning("send_video: no websocket for session %s", session.id[:8])
            return

        if isinstance(video, bytes):
            await self._send_frame_binary(session, websocket, video)
        else:
            async for chunk in video:
                await self._send_frame_binary(
                    session,
                    websocket,
                    chunk.data,
                    keyframe=chunk.keyframe,
                    codec=chunk.codec,
                )

    async def _send_frame_binary(
        self,
        session: VideoSession,
        websocket: Any,
        data: bytes,
        *,
        keyframe: bool = False,
        codec: str | None = None,
    ) -> None:
        """Serialize and send a video frame as a binary WebSocket message."""
        codec = codec or self._default_codec
        codec_id = _ID_BY_CODEC.get(codec, 0)

        seq = self._outbound_sequences.get(session.id, 0)
        self._outbound_sequences[session.id] = seq + 1

        flags = (int(keyframe) & 0x01) | ((codec_id & 0x07) << 1)
        header = _HEADER_STRUCT.pack(flags, seq)

        try:
            await websocket.send_bytes(header + data)
        except Exception:
            logger.debug("Failed to send video frame to session %s", session.id[:8])

    # -------------------------------------------------------------------------
    # Inbound video handling
    # -------------------------------------------------------------------------

    def _handle_binary_frame(self, connection_id: str, data: bytes) -> None:
        """Parse a binary WebSocket message into a VideoFrame."""
        if len(data) <= _HEADER_SIZE:
            return

        flags, seq = _HEADER_STRUCT.unpack_from(data, 0)
        payload = data[_HEADER_SIZE:]

        keyframe = bool(flags & 0x01)
        codec_id = (flags >> 1) & 0x07
        codec = _CODEC_BY_ID.get(codec_id, "h264")

        session_id = self._connection_sessions.get(connection_id)
        if session_id is None:
            return
        session = self._sessions.get(session_id)
        if session is None:
            return

        config = self._connection_config.get(connection_id, {})
        width = config.get("width", self._default_width)
        height = config.get("height", self._default_height)

        frame = VideoFrame(
            data=payload,
            codec=codec,
            width=width,
            height=height,
            timestamp_ms=time.monotonic() * 1000.0,
            keyframe=keyframe,
            sequence=seq,
        )

        if self._video_received_callback is not None:
            self._video_received_callback(session, frame)
        for tap in self._video_taps:
            tap(session, frame)

    def _handle_json_message(self, connection_id: str, message: dict[str, Any]) -> None:
        """Handle a JSON control message from a client."""
        msg_type = message.get("type")
        if msg_type == "config":
            config = self._connection_config.setdefault(connection_id, {})
            if "codec" in message:
                config["codec"] = message["codec"]
            if "width" in message:
                config["width"] = int(message["width"])
            if "height" in message:
                config["height"] = int(message["height"])
            logger.debug(
                "Client config updated: conn=%s config=%s",
                connection_id[:8],
                config,
            )

    # -------------------------------------------------------------------------
    # WebSocket connection lifecycle
    # -------------------------------------------------------------------------

    async def _on_client_connect(self, connection_id: str, websocket: Any) -> VideoSession | None:
        """Handle a new WebSocket client connection."""
        if self._session_factory:
            result = self._session_factory(connection_id)
            if asyncio.iscoroutine(result):
                session = await result
            else:
                session = result
        else:
            session = await self.connect(
                room_id="default",
                participant_id=connection_id,
                channel_id="video",
            )

        if session is None:
            return None

        self._connection_sessions[connection_id] = session.id
        self._websockets[session.id] = websocket

        logger.info(
            "WebSocket video client connected: session=%s conn=%s",
            session.id[:8],
            connection_id[:8],
        )

        for cb in self._session_ready_callbacks:
            with contextlib.suppress(Exception):
                result = cb(session)
                if asyncio.iscoroutine(result):
                    await result

        return session

    async def _on_client_disconnect(self, connection_id: str) -> None:
        """Handle a WebSocket client disconnection."""
        session_id = self._connection_sessions.pop(connection_id, None)
        if session_id is None:
            return

        session = self._sessions.get(session_id)
        if session is None:
            return

        logger.info(
            "WebSocket video client disconnected: session=%s",
            session_id[:8],
        )

        self._connection_config.pop(connection_id, None)

        for cb in self._disconnect_callbacks:
            with contextlib.suppress(Exception):
                result = cb(session)
                if asyncio.iscoroutine(result):
                    await result

        await self.disconnect(session)

    # -------------------------------------------------------------------------
    # Callback registration
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._video_received_callback = callback

    def add_video_tap(self, callback: VideoReceivedCallback) -> None:
        self._video_taps.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)


def mount_websocket_video(
    app: FastAPI,
    backend: WebSocketVideoBackend,
    *,
    path: str = "/video/ws",
) -> None:
    """Mount a WebSocket video endpoint on a FastAPI app.

    Creates a WebSocket endpoint that handles the binary frame protocol
    and JSON control messages.

    Args:
        app: FastAPI application.
        backend: The WebSocketVideoBackend instance.
        path: URL path for the WebSocket endpoint.
    """
    import json as _json

    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket(path)
    async def video_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        connection_id = uuid4().hex

        session = await backend._on_client_connect(connection_id, websocket)
        if session is None:
            await websocket.close(code=1008, reason="Session creation failed")
            return

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"]:
                    backend._handle_binary_frame(connection_id, message["bytes"])
                elif "text" in message and message["text"]:
                    try:
                        data = _json.loads(message["text"])
                        backend._handle_json_message(connection_id, data)
                    except _json.JSONDecodeError:
                        logger.debug("Invalid JSON from client %s", connection_id[:8])
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("WebSocket video error: conn=%s", connection_id[:8])
        finally:
            await backend._on_client_disconnect(connection_id)

    logger.info("WebSocket video endpoint mounted at %s", path)
