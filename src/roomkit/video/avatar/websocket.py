"""WebSocket avatar provider — remote GPU inference.

HTTP for lifecycle (start/stop), WebSocket for real-time
audio → video streaming with binary frames.

Works with any animation server that speaks the protocol:
  - POST /start     — initialize with reference image + dimensions
  - WS   /ws/stream — stream PCM audio in, receive RGB24 frames out
  - GET  /idle_frame — fetch static idle frame
  - POST /stop      — tear down session

Frame wire format (WS binary + HTTP body)::

    [2B width BE][2B height BE][width × height × 3 bytes RGB24]

Usage::

    from roomkit.video.avatar.websocket import WebSocketAvatarProvider

    avatar = WebSocketAvatarProvider(base_url="http://gpu-server:8765")
    await avatar.start(image_bytes, width=512, height=512)
    frames = avatar.feed_audio(pcm_chunk, sample_rate=16000)
"""

from __future__ import annotations

import base64
import contextlib
import logging
import struct
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

from roomkit.video.avatar.base import AvatarProvider

if TYPE_CHECKING:
    import httpx
    from websockets.sync.client import ClientConnection

    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.avatar.websocket")

_HEADER_FMT = "!HH"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_SCHEME_MAP = {"http": "ws", "https": "wss"}


def _http_to_ws_url(base_url: str, path: str) -> str:
    """Convert an HTTP base URL to a WebSocket URL with the given path."""
    parsed = urlparse(base_url)
    scheme = _SCHEME_MAP.get(parsed.scheme, parsed.scheme)
    return urlunparse(parsed._replace(scheme=scheme, path=path))


class WebSocketAvatarProvider(AvatarProvider):
    """Avatar provider that streams audio/video over WebSocket.

    Connects to a remote animation server via HTTP (lifecycle) and
    a persistent WebSocket (real-time streaming). The server can run
    any model (MuseTalk, Wav2Lip, SadTalker, etc.) as long as it
    implements the expected HTTP+WS protocol.

    Args:
        base_url: URL of the avatar service (e.g. ``http://localhost:8765``).
        timeout: HTTP request timeout in seconds (default 30).
        fps: Expected output frame rate (must match the server).
        sample_rate: Audio sample rate sent to server in ``/start`` payload.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        fps: int = 30,
        sample_rate: int = 16000,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._ws_url = _http_to_ws_url(self._base_url, "/ws/stream")
        self._timeout = timeout
        self._fps = fps
        self._sample_rate = sample_rate
        self._http: httpx.Client | None = None
        self._ws: ClientConnection | None = None
        self._started = False
        self._width = 512
        self._height = 512
        self._idle_frame_cache: VideoFrame | None = None
        self._reference_image: bytes | None = None
        self._idle_error_logged = False

    @property
    def name(self) -> str:
        return "websocket"

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
        import httpx

        self._width = width
        self._height = height
        self._reference_image = reference_image

        payload = self._build_start_payload()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base_url}/start", json=payload)
            resp.raise_for_status()
            self._fps = resp.json().get("fps", self._fps)
            logger.info(
                "WebSocket avatar started: %s (%dx%d @ %dfps)",
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

        # Sync client for thread-pool operations (get_idle_frame, _try_restart)
        self._http = httpx.Client(timeout=self._timeout)
        self._started = True
        self._idle_frame_cache = None
        self._idle_error_logged = False

    def _build_start_payload(self) -> dict[str, Any]:
        """Build the JSON payload for POST /start."""
        return {
            "reference_image": base64.b64encode(self._reference_image or b"").decode(),
            "width": self._width,
            "height": self._height,
            "sample_rate": self._sample_rate,
        }

    # -- WebSocket connection management ---------------------------------------

    def _ensure_ws(self) -> ClientConnection:
        """Return the persistent WS connection, opening one if needed."""
        if self._ws is None:
            from websockets.sync.client import connect

            self._ws = connect(self._ws_url, close_timeout=2)
        return self._ws

    def _close_ws(self) -> None:
        """Close the WS connection (idempotent)."""
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None

    # -- Frame parsing ---------------------------------------------------------

    def _parse_frame(self, data: bytes) -> VideoFrame | None:
        """Parse a binary frame message into a VideoFrame.

        Returns None if the data is too small or pixel count doesn't
        match the declared dimensions.
        """
        if len(data) <= _HEADER_SIZE:
            return None
        from roomkit.video.video_frame import VideoFrame

        w, h = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
        payload = data[_HEADER_SIZE:]
        expected = w * h * 3
        if len(payload) != expected:
            logger.warning(
                "Frame size mismatch: got %d bytes, expected %d (%dx%dx3)",
                len(payload),
                expected,
                w,
                h,
            )
            return None
        return VideoFrame(
            data=payload,
            codec="raw_rgb24",
            width=w,
            height=h,
            keyframe=True,
        )

    def _recv_frames(self) -> list[VideoFrame]:
        """Read frames from WS until the server sends an empty done message."""
        if self._ws is None:
            return []
        frames: list[VideoFrame] = []
        while True:
            try:
                msg = self._ws.recv(timeout=self._timeout)
            except Exception:
                self._close_ws()
                break
            if not isinstance(msg, bytes) or len(msg) == 0:
                break
            frame = self._parse_frame(msg)
            if frame is not None:
                frames.append(frame)
        return frames

    # -- Audio streaming -------------------------------------------------------

    def _stream_audio(self, pcm_data: bytes) -> list[VideoFrame]:
        """Send audio over persistent WS and receive frames."""
        ws = self._ensure_ws()
        ws.send(pcm_data)
        return self._recv_frames()

    def feed_audio(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
    ) -> list[VideoFrame]:
        """Send audio to server, receive frames via persistent WebSocket."""
        if not self._started:
            return []
        try:
            return self._stream_audio(pcm_data)
        except Exception:
            # Connection lost — reconnect and retry once
            self._close_ws()
            try:
                return self._stream_audio(pcm_data)
            except Exception:
                logger.debug("WebSocket feed_audio failed after reconnect", exc_info=True)
                self._close_ws()
                self._try_restart()
                return []

    # -- Server recovery -------------------------------------------------------

    def _try_restart(self) -> None:
        """Re-POST /start if the server was restarted."""
        if self._http is None or self._reference_image is None:
            return
        try:
            resp = self._http.post(
                f"{self._base_url}/start",
                json=self._build_start_payload(),
            )
            resp.raise_for_status()
            logger.info("Re-connected to avatar server after restart")
        except Exception:
            logger.debug("Re-start failed", exc_info=True)

    # -- Idle frame ------------------------------------------------------------

    def get_idle_frame(self) -> VideoFrame | None:
        if not self._started or self._http is None:
            return None
        if self._idle_frame_cache is not None:
            return self._idle_frame_cache
        try:
            resp = self._http.get(f"{self._base_url}/idle_frame")
            if not resp.is_success:
                self._try_restart()
                resp = self._http.get(f"{self._base_url}/idle_frame")
            resp.raise_for_status()
            frame = self._parse_frame(resp.content)
            if frame is not None:
                self._idle_frame_cache = frame
            return frame
        except Exception:
            if not self._idle_error_logged:
                logger.warning("Failed to fetch idle frame from %s", self._base_url)
                self._idle_error_logged = True
            return None

    # -- Flush & stop ----------------------------------------------------------

    def flush(self) -> list[VideoFrame]:
        """Signal the server to flush remaining buffered frames."""
        if not self._started or self._ws is None:
            return []
        try:
            self._ws.send(b"")
            return self._recv_frames()
        except Exception:
            logger.debug("Flush failed", exc_info=True)
            self._close_ws()
            return []

    async def stop(self) -> None:
        self._close_ws()
        if self._http is not None:
            import httpx as _httpx

            with contextlib.suppress(Exception):
                async with _httpx.AsyncClient(timeout=self._timeout) as client:
                    await client.post(f"{self._base_url}/stop")
            self._http.close()
            self._http = None
        self._started = False
        self._idle_frame_cache = None
        self._reference_image = None
        logger.info("WebSocket avatar stopped")
