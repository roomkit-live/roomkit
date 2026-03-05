"""FastRTC VoiceBackend implementation for RoomKit.

This module provides a VoiceBackend that uses FastRTC for WebRTC and WebSocket
audio transport.  The backend is a pure transport — all audio intelligence
(VAD, denoising, diarization) is handled by the AudioPipeline.

The backend supports three FastRTC transport modes:

- **WebRTC** (``/webrtc/offer``): peer-to-peer audio via RTP tracks.
  Outbound audio is returned from the handler's ``emit()`` method.
- **WebSocket** (``/websocket/offer``): mu-law encoded audio in JSON messages.
- **Telephone** (``/telephone/handler``): Twilio-style telephony WebSocket.

Requires the ``fastrtc`` optional dependency::

    pip install roomkit[fastrtc]

Usage::

    from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, mount_fastrtc_voice

    backend = FastRTCVoiceBackend()
    voice_channel = VoiceChannel("voice", stt=stt, tts=tts, backend=backend)
    kit.register_channel(voice_channel)

    # Mount FastRTC endpoints on FastAPI app (in lifespan)
    mount_fastrtc_voice(app, backend, path="/fastrtc")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import struct
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

# Suppress gradio/huggingface telemetry that fires on import — set before
# fastrtc/gradio are imported (both are lazy-imported inside mount_fastrtc_voice).
# Users can override by setting the env vars explicitly before importing this module.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.auth import AuthCallback, auth_context
from roomkit.voice.backends.base import AudioReceivedCallback, SessionReadyCallback, VoiceBackend
from roomkit.voice.base import (
    AudioChunk,
    VoiceCapability,
    VoiceSession,
    VoiceSessionState,
)

if TYPE_CHECKING:
    import numpy as np
    from fastapi import FastAPI

logger = logging.getLogger("roomkit.voice.fastrtc")

# Re-export so existing ``from roomkit.voice.backends.fastrtc import AuthCallback``
# continues to work.
__all__ = ["AuthCallback", "auth_context", "FastRTCVoiceBackend", "mount_fastrtc_voice"]

# ---------------------------------------------------------------------------
# Pure-Python mu-law encoder (replaces audioop.lin2ulaw removed in Python 3.13)
# ---------------------------------------------------------------------------

# ITU-T G.711 mu-law compression bias and clip level
_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635

# Precomputed lookup table: PCM-16 sample (unsigned magnitude) → mu-law byte.
# Built once at import time for O(1) encoding per sample.
_MULAW_TABLE: bytes | None = None


def _build_mulaw_table() -> bytes:
    """Build a 16384-entry lookup table mapping 14-bit magnitude to mu-law 7-bit value.

    Returns the lower 7 bits (exponent + mantissa) with bits inverted.
    The caller must OR in the sign bit (0x80) separately.
    """
    table = bytearray(16384)
    for i in range(16384):
        sample = min(i, _MULAW_CLIP) + _MULAW_BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (sample & mask):
            exponent -= 1
            mask >>= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        table[i] = ~((exponent << 4) | mantissa) & 0x7F
    return bytes(table)


def _pcm16_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert PCM-16 LE bytes to mu-law bytes (pure Python).

    Each pair of bytes in *pcm_data* is interpreted as a signed 16-bit
    little-endian sample and encoded to one mu-law byte per the ITU-T
    G.711 standard.
    """
    global _MULAW_TABLE  # noqa: PLW0603
    if _MULAW_TABLE is None:
        _MULAW_TABLE = _build_mulaw_table()

    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data[: n_samples * 2])
    out = bytearray(n_samples)
    table = _MULAW_TABLE
    for i, s in enumerate(samples):
        sign = 0x80 if s >= 0 else 0x00
        magnitude = -s if s < 0 else s
        magnitude = min(magnitude, _MULAW_CLIP)
        # Shift right once to get a 14-bit index (15-bit magnitude → 14-bit)
        out[i] = table[magnitude >> 1] | sign
    return bytes(out)


class FastRTCVoiceBackend(VoiceBackend):
    """VoiceBackend implementation using FastRTC for WebRTC and WebSocket transport.

    Supports all three FastRTC transport modes (WebRTC, WebSocket, Telephone).
    Delivers raw audio frames via on_audio_received callback.  All audio
    intelligence (VAD, denoising, diarization) is handled by the AudioPipeline.
    """

    #: Default maximum size for per-session audio queues.
    DEFAULT_QUEUE_MAXSIZE: int = 1000

    def __init__(
        self,
        *,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 24000,
        audio_format: str = "mulaw",
        audio_queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        """
        Args:
            input_sample_rate: Expected inbound sample rate.
            output_sample_rate: Target outbound sample rate.
            audio_format: WebSocket audio encoding — ``"mulaw"`` (mu-law in
                JSON, Twilio-compatible) or ``"pcm"`` (raw PCM-16 LE binary
                frames).  Has no effect on WebRTC transport (always PCM via
                RTP).
            audio_queue_maxsize: Max pending frames per session emit queue.
        """
        if audio_format not in ("mulaw", "pcm"):
            msg = f"audio_format must be 'mulaw' or 'pcm', got {audio_format!r}"
            raise ValueError(msg)
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._audio_format = audio_format
        self._audio_queue_maxsize = audio_queue_maxsize

        # Callback for raw audio frames
        self._audio_received_callback: AudioReceivedCallback | None = None

        # Session tracking: session_id -> VoiceSession
        self._sessions: dict[str, VoiceSession] = {}

        # FastRTC stream (set by mount_fastrtc_voice)
        self._stream: Any = None

        # Emit queues for WebRTC sessions: webrtc_id -> asyncio.Queue
        # WebRTC audio goes through emit() → player_worker_decode → RTP track.
        self._emit_queues: dict[str, asyncio.Queue[tuple[int, Any] | None]] = {}

        # WebSocket references: session_id -> websocket
        self._websockets: dict[str, Any] = {}

        # Session ready callbacks
        self._session_ready_callbacks: list[SessionReadyCallback] = []

    @property
    def name(self) -> str:
        return "FastRTC"

    @property
    def capabilities(self) -> VoiceCapability:
        return VoiceCapability.NONE

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        session_id = str(uuid.uuid4())
        session_metadata = {
            "input_sample_rate": self._input_sample_rate,
            "output_sample_rate": self._output_sample_rate,
            **(metadata or {}),
        }
        session = VoiceSession(
            id=session_id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VoiceSessionState.ACTIVE,
            metadata=session_metadata,
        )
        self._sessions[session_id] = session

        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        with telemetry.span(
            SpanKind.BACKEND_CONNECT,
            "backend.connect",
            room_id=room_id,
            session_id=session_id,
            attributes={Attr.BACKEND_TYPE: "FastRTC"},
        ):
            pass  # connection is synchronous for FastRTC

        logger.info(
            "Voice session created: session=%s, room=%s, participant=%s",
            session_id,
            room_id,
            participant_id,
        )
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        session.state = VoiceSessionState.ENDED
        self._sessions.pop(session.id, None)
        ws_id = session.metadata.get("websocket_id")
        if ws_id:
            self._emit_queues.pop(ws_id, None)
        self._websockets.pop(session.id, None)
        logger.info("Voice session ended: session=%s", session.id)

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        """Register callback for raw inbound audio frames."""
        self._audio_received_callback = callback

    def on_session_ready(self, callback: SessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def _resolve_websocket(self, session: VoiceSession) -> Any | None:
        """Resolve WebSocket for a session.

        First checks the explicit registry (populated by _register_websocket).
        Falls back to looking up the websocket from the FastRTC Stream's
        connection registry using the ``websocket_id`` stored in session
        metadata.
        """
        ws = self._websockets.get(session.id)
        if ws is not None:
            return ws

        ws_id = session.metadata.get("websocket_id")
        if ws_id and self._stream and hasattr(self._stream, "connections"):
            handlers = self._stream.connections.get(ws_id)
            if handlers and hasattr(handlers[0], "websocket") and handlers[0].websocket:
                ws = handlers[0].websocket
                self._websockets[session.id] = ws
                logger.info(
                    "Websocket resolved from Stream connections: ws=%s session=%s",
                    ws_id,
                    session.id,
                )
                return ws

        return None

    def _is_webrtc_session(self, session: VoiceSession) -> bool:
        """Check if a session is connected via WebRTC (not WebSocket)."""
        return session.metadata.get("transport") == "webrtc"

    def _get_emit_queue(
        self, session: VoiceSession
    ) -> asyncio.Queue[tuple[int, Any] | None] | None:
        """Get the emit queue for a WebRTC session."""
        ws_id = session.metadata.get("websocket_id")
        if ws_id:
            return self._emit_queues.get(ws_id)
        return None

    def _pcm_to_numpy(self, pcm_data: bytes, sample_rate: int) -> tuple[int, Any]:
        """Convert PCM-16 LE bytes to a (sample_rate, numpy_int16_array) tuple."""
        import numpy as _np

        arr = _np.frombuffer(pcm_data, dtype=_np.int16)
        return (sample_rate, arr)

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        if self._is_webrtc_session(session):
            queue = self._get_emit_queue(session)
            if queue is None:
                logger.warning("No emit queue for WebRTC session %s", session.id)
                return
            sample_rate = session.metadata.get("output_sample_rate", self._output_sample_rate)
            try:
                if isinstance(audio, bytes):
                    queue.put_nowait(self._pcm_to_numpy(audio, sample_rate))
                else:
                    async for chunk in audio:
                        if chunk.data:
                            queue.put_nowait(self._pcm_to_numpy(chunk.data, sample_rate))
            except Exception:
                logger.exception("Error sending audio to WebRTC session %s", session.id)
            return

        # WebSocket path
        websocket = self._resolve_websocket(session)
        if not websocket:
            logger.warning("No WebSocket for session %s", session.id)
            return
        try:
            if isinstance(audio, bytes):
                await self._send_ws_audio(websocket, audio)
            else:
                async for chunk in audio:
                    if chunk.data:
                        await self._send_ws_audio(websocket, chunk.data)
        except Exception:
            logger.exception("Error sending audio to session %s", session.id)

    @staticmethod
    def _ws_is_connected(websocket: Any) -> bool:
        """Return True if *websocket* is still open (Starlette-aware)."""
        state = getattr(websocket, "client_state", None)
        if state is None:
            return True  # not a Starlette WebSocket — assume connected
        try:
            from starlette.websockets import WebSocketState

            return bool(state == WebSocketState.CONNECTED)
        except ImportError:
            return True

    async def _send_ws_audio(self, websocket: Any, pcm_data: bytes) -> None:
        """Send audio to a WebSocket using the configured format."""
        if not self._ws_is_connected(websocket):
            return
        if self._audio_format == "pcm":
            await websocket.send_bytes(pcm_data)
        else:
            mulaw_data = _pcm16_to_mulaw(pcm_data)
            payload = base64.b64encode(mulaw_data).decode("utf-8")
            await websocket.send_json({"event": "media", "media": {"payload": payload}})

    def _prepare_ws_sync(self, pcm_data: bytes) -> tuple[str, bytes | dict[str, Any]]:
        """Prepare a WebSocket message synchronously (for audio thread).

        Returns ``("bytes", data)`` for PCM or ``("json", message)`` for
        mu-law, ready to schedule on the event loop.
        """
        if self._audio_format == "pcm":
            return ("bytes", pcm_data)
        mulaw_data = _pcm16_to_mulaw(pcm_data)
        payload = base64.b64encode(mulaw_data).decode("utf-8")
        return ("json", {"event": "media", "media": {"payload": payload}})

    def send_audio_sync(self, session: VoiceSession, chunk: AudioChunk) -> None:
        """Synchronously send a single audio chunk.

        Used by the audio bridge for frame-by-frame forwarding from audio
        callback threads.

        - **WebRTC**: Puts a numpy frame into the session's emit queue
          (consumed by the handler's ``emit()`` → FastRTC RTP track).
        - **WebSocket (mulaw)**: mu-law + base64 in calling thread, then
          schedules ``send_json`` on the event loop.
        - **WebSocket (pcm)**: schedules ``send_bytes`` on the event loop.
        """
        if self._is_webrtc_session(session):
            queue = self._get_emit_queue(session)
            if queue is None:
                return
            sample_rate = session.metadata.get("output_sample_rate", self._output_sample_rate)
            try:
                queue.put_nowait(self._pcm_to_numpy(chunk.data, sample_rate))
            except Exception:
                logger.warning(
                    "send_audio_sync: emit queue full for session %s",
                    session.id,
                )
            return

        # WebSocket path
        websocket = self._resolve_websocket(session)
        if not websocket or not self._ws_is_connected(websocket):
            return

        mode, data = self._prepare_ws_sync(chunk.data)
        try:
            loop = asyncio.get_running_loop()
            if mode == "bytes":
                loop.create_task(websocket.send_bytes(data))
            else:
                loop.create_task(websocket.send_json(data))
        except RuntimeError:
            logger.warning(
                "send_audio_sync: no event loop for session %s",
                session.id,
            )

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        websocket = self._resolve_websocket(session)
        logger.info(
            "send_transcription: session=%s, role=%s, has_websocket=%s, text=%s",
            session.id,
            role,
            websocket is not None,
            text[:50] if text else "",
        )
        if websocket:
            if not self._ws_is_connected(websocket):
                return
            try:
                await websocket.send_json(
                    {
                        "type": "transcription",
                        "data": {"text": text, "role": role},
                    }
                )
                logger.info("Transcription sent to client")
            except Exception:
                logger.exception("Error sending transcription")
        else:
            logger.warning(
                "No websocket for session %s, registered sessions: %s",
                session.id,
                list(self._websockets.keys()),
            )

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        for session in list(self._sessions.values()):
            await self.disconnect(session)
        self._websockets.clear()
        self._emit_queues.clear()

    # -------------------------------------------------------------------------
    # FastRTC integration methods (called by mount_fastrtc_voice)
    # -------------------------------------------------------------------------

    def _handle_audio_frame(
        self, websocket_id: str, audio_data: np.ndarray[Any, Any], sample_rate: int
    ) -> None:
        """Called by FastRTC handler with raw audio data.

        Converts numpy array to AudioFrame and fires on_audio_received callback.
        """
        import numpy as _np

        session = self._find_session_by_websocket_id(websocket_id)
        if not session or not self._audio_received_callback:
            return

        # Convert numpy array to bytes
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        if audio_data.dtype != _np.int16:
            audio_data = (audio_data * 32767).astype(_np.int16)

        frame = AudioFrame(
            data=audio_data.tobytes(),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
        )
        self._audio_received_callback(session, frame)

    def _register_websocket(self, websocket_id: str, session_id: str, websocket: Any) -> None:
        self._websockets[session_id] = websocket
        session = self._sessions.get(session_id)
        if session:
            session.metadata["websocket_id"] = websocket_id
            session.metadata["transport"] = "websocket"
            # Audio path is now live — fire session ready callbacks
            for cb in self._session_ready_callbacks:
                cb(session)

    def _register_webrtc(self, webrtc_id: str, session_id: str) -> None:
        """Register a WebRTC session and create its emit queue."""
        session = self._sessions.get(session_id)
        if session:
            session.metadata["websocket_id"] = webrtc_id
            session.metadata["transport"] = "webrtc"
            self._emit_queues[webrtc_id] = asyncio.Queue(maxsize=self._audio_queue_maxsize)
            for cb in self._session_ready_callbacks:
                cb(session)

    def _find_session_by_websocket_id(self, websocket_id: str) -> VoiceSession | None:
        for session in self._sessions.values():
            if session.metadata.get("websocket_id") == websocket_id:
                return session
        return None


def mount_fastrtc_voice(
    app: FastAPI,
    backend: FastRTCVoiceBackend,
    *,
    path: str = "/fastrtc",
    session_factory: Any = None,
    auth: AuthCallback | None = None,
) -> None:
    """Mount FastRTC voice endpoints on a FastAPI app.

    Registers three endpoint groups:

    - ``/webrtc/offer`` — WebRTC signaling (SDP offer/answer, ICE)
    - ``/websocket/offer`` — WebSocket audio (mu-law in JSON)
    - ``/telephone/handler`` — Twilio-style telephony WebSocket

    Args:
        app: FastAPI application.
        backend: The FastRTCVoiceBackend instance.
        path: Base path for voice endpoints (default: /fastrtc).
        session_factory: Async callable(websocket_id) -> VoiceSession that creates
            sessions when clients connect. If not provided, sessions must be
            created manually before clients connect.
        auth: Optional async callback for authenticating connections. Receives
            the WebSocket and returns a metadata dict on success or ``None``
            to reject. Auth metadata is available via :data:`auth_context`.
    """
    from fastrtc import AsyncStreamHandler, Stream

    backend._session_factory = session_factory  # type: ignore[attr-defined]

    class AudioPassthroughHandler(AsyncStreamHandler):  # type: ignore[misc,unused-ignore]
        """Passes raw audio frames to the backend's on_audio_received callback.

        Each connection gets its own handler instance via ``copy()``.
        For WebRTC, ``emit()`` returns frames from a per-connection queue.
        For WebSocket, ``emit()`` returns None (audio sent directly on WS).
        """

        def __init__(self) -> None:
            super().__init__()
            self._rejected = False
            self._auth_meta: dict[str, Any] | None = None
            self._webrtc_id: str | None = None
            self._is_webrtc = False

        def copy(self) -> AudioPassthroughHandler:
            return AudioPassthroughHandler()

        async def start_up(self) -> None:
            """Called once per connection — run auth and detect transport."""
            from fastrtc.utils import current_context

            ctx = current_context.get()
            if not ctx:
                return
            self._webrtc_id = ctx.webrtc_id
            # WebRTC connections have no websocket object
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

            # Pass raw audio to pipeline via callback
            backend._handle_audio_frame(connection_id, audio_data, sample_rate)

        async def emit(self) -> tuple[int, Any] | None:
            if self._is_webrtc and self._webrtc_id:
                # WebRTC: pull from per-connection emit queue
                queue = backend._emit_queues.get(self._webrtc_id)
                if queue:
                    try:
                        return await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        return None
                # Queue not created yet — wait for session registration
                await asyncio.sleep(0.1)
                return None

            # WebSocket: audio sent directly via send_audio / send_audio_sync.
            # Sleep to prevent the _emit_to_queue tight loop from spinning CPU.
            await asyncio.sleep(0.1)
            return None

    # Create FastRTC stream with passthrough handler
    stream = Stream(
        handler=AudioPassthroughHandler(),
        modality="audio",
        mode="send-receive",
    )

    backend._stream = stream

    stream.mount(app, path=path)
    logger.info("FastRTC voice backend mounted at %s", path)
