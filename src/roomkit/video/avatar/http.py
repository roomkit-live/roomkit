"""HTTP/WebSocket avatar provider — remote GPU inference.

HTTP for lifecycle (start/stop), WebSocket for real-time
audio → video streaming with binary frames.

Usage::

    from roomkit.video.avatar.http import HTTPAvatarProvider

    avatar = HTTPAvatarProvider(base_url="http://gpu-server:8765")
    await avatar.start(image_bytes, width=512, height=512)
    frames = avatar.feed_audio(pcm_chunk, sample_rate=16000)

See ``musetalk-http/server.py`` for the server side.
"""

from __future__ import annotations

import base64
import logging
import struct
from typing import TYPE_CHECKING

import httpx

from roomkit.video.avatar.base import AvatarProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.avatar.http")

_HEADER_FMT = "!HH"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class HTTPAvatarProvider(AvatarProvider):
    """Avatar provider that streams audio/video over WebSocket.

    Args:
        base_url: URL of the avatar service (e.g. ``http://localhost:8765``).
        timeout: HTTP request timeout in seconds (default 30).
        fps: Expected output frame rate (must match the server).
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        fps: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._ws_url = (
            self._base_url.replace("http://", "ws://").replace(
                "https://",
                "wss://",
            )
            + "/ws/stream"
        )
        self._timeout = timeout
        self._fps = fps
        self._http: httpx.Client | None = None
        self._started = False
        self._width = 512
        self._height = 512
        self._idle_frame_cache: VideoFrame | None = None
        self._last_start_args: dict | None = None

    @property
    def name(self) -> str:
        return "http"

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(
        self,
        reference_image: bytes,
        *,
        width: int = 512,
        height: int = 512,
    ) -> None:
        self._width = width
        self._height = height
        self._http = httpx.Client(timeout=self._timeout)

        self._last_start_args = {
            "reference_image": base64.b64encode(reference_image).decode(),
            "width": width,
            "height": height,
        }
        # Lazy start: if the server is not available yet, mark as started
        # and retry on first use. This allows RoomKit to boot without
        # waiting for the GPU service.
        try:
            resp = self._http.post(
                f"{self._base_url}/start",
                json=self._last_start_args,
            )
            resp.raise_for_status()
            result = resp.json()
            self._fps = result.get("fps", self._fps)
            logger.info(
                "HTTP avatar started: %s (%dx%d @ %dfps)",
                self._base_url,
                width,
                height,
                self._fps,
            )
        except Exception:
            logger.warning(
                "Avatar service not available at %s — will retry on first use",
                self._base_url,
            )
        self._started = True
        self._idle_frame_cache = None
        self._idle_error_logged = False

    def feed_audio(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
    ) -> list[VideoFrame]:
        """Send audio to server, receive frames via synchronous WebSocket.

        Protocol:
          1. Client sends audio bytes
          2. Server receives, runs inference, sends frames back
          3. Server sends empty bytes = "done"
          4. Client reads frames until "done" signal
        """
        if not self._started or self._http is None:
            return []

        from websockets.sync.client import connect as ws_connect

        from roomkit.video.video_frame import VideoFrame

        try:
            with ws_connect(self._ws_url, close_timeout=2) as ws:
                ws.send(pcm_data)

                frames: list[VideoFrame] = []
                while True:
                    try:
                        msg = ws.recv(timeout=self._timeout)
                    except Exception:
                        break
                    if not isinstance(msg, bytes):
                        break
                    # Empty message = server signals "done"
                    if len(msg) == 0:
                        break
                    if len(msg) <= _HEADER_SIZE:
                        continue
                    w, h = struct.unpack(_HEADER_FMT, msg[:_HEADER_SIZE])
                    frames.append(
                        VideoFrame(
                            data=msg[_HEADER_SIZE:],
                            codec="raw_rgb24",
                            width=w,
                            height=h,
                            keyframe=True,
                        )
                    )
                return frames
        except Exception:
            logger.debug("WebSocket feed_audio error", exc_info=True)
            # Re-init on connection failure (server may have restarted)
            self._try_restart()
            return []

    def _try_restart(self) -> None:
        """Re-POST /start if server was restarted."""
        if self._http is None or not self._last_start_args:
            return
        try:
            resp = self._http.post(
                f"{self._base_url}/start",
                json=self._last_start_args,
            )
            resp.raise_for_status()
            logger.info("Re-connected to avatar server after restart")
        except Exception:
            logger.debug("Re-start failed", exc_info=True)

    def get_idle_frame(self) -> VideoFrame | None:
        if not self._started or self._http is None:
            return None

        if self._idle_frame_cache is not None:
            return self._idle_frame_cache

        from roomkit.video.video_frame import VideoFrame

        try:
            resp = self._http.get(f"{self._base_url}/idle_frame")
            if resp.status_code == 400:
                # Server restarted — re-send /start
                self._try_restart()
                resp = self._http.get(f"{self._base_url}/idle_frame")
            resp.raise_for_status()
            data = resp.content
            if len(data) <= _HEADER_SIZE:
                return None
            w, h = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
            frame = VideoFrame(
                data=data[_HEADER_SIZE:],
                codec="raw_rgb24",
                width=w,
                height=h,
                keyframe=True,
            )
            self._idle_frame_cache = frame
            return frame
        except Exception:
            # Don't log at 30fps — log once then suppress
            if not getattr(self, "_idle_error_logged", False):
                logger.warning("Failed to fetch idle frame from %s", self._base_url)
                self._idle_error_logged = True
            return None

    def flush(self) -> list[VideoFrame]:
        return []

    async def stop(self) -> None:
        if self._http is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._http.post(f"{self._base_url}/stop")
            self._http.close()
            self._http = None
        self._started = False
        self._idle_frame_cache = None
        logger.info("HTTP avatar stopped")


# Type alias for the websocket
Any = object
