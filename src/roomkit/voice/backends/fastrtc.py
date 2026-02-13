"""FastRTC VoiceBackend implementation for RoomKit.

This module provides a VoiceBackend that uses FastRTC for WebSocket audio transport.
The backend is a pure transport — all audio intelligence (VAD, denoising, diarization)
is handled by the AudioPipeline.

The backend:
- Handles WebSocket connections from clients
- Delivers raw audio frames via on_audio_received callback
- Sends TTS audio back to clients via FastRTC

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
import struct
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import AudioReceivedCallback, VoiceBackend
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
        out[i] = table[magnitude >> 2] | sign
    return bytes(out)


class FastRTCVoiceBackend(VoiceBackend):
    """VoiceBackend implementation using FastRTC for WebSocket audio transport.

    This backend is a pure transport — it delivers raw audio frames via
    the on_audio_received callback. All audio intelligence (VAD, denoising,
    diarization) is handled by the AudioPipeline.
    """

    #: Default maximum size for per-session audio queues.
    DEFAULT_QUEUE_MAXSIZE: int = 1000

    def __init__(
        self,
        *,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 24000,
        audio_queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._audio_queue_maxsize = audio_queue_maxsize

        # Callback for raw audio frames
        self._audio_received_callback: AudioReceivedCallback | None = None

        # Session tracking: session_id -> VoiceSession
        self._sessions: dict[str, VoiceSession] = {}

        # FastRTC stream (set by mount_fastrtc_voice)
        self._stream: Any = None

        # Pending audio to send: session_id -> asyncio.Queue of audio chunks
        self._audio_queues: dict[str, asyncio.Queue[AudioChunk | None]] = {}

        # WebSocket references: session_id -> websocket
        self._websockets: dict[str, Any] = {}

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
        self._audio_queues[session_id] = asyncio.Queue(maxsize=self._audio_queue_maxsize)
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
        self._audio_queues.pop(session.id, None)
        self._websockets.pop(session.id, None)
        logger.info("Voice session ended: session=%s", session.id)

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        """Register callback for raw inbound audio frames."""
        self._audio_received_callback = callback

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

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        websocket = self._resolve_websocket(session)
        if not websocket:
            logger.warning("No WebSocket for session %s", session.id)
            return

        try:
            if isinstance(audio, bytes):
                await self._send_mulaw_audio(websocket, audio)
            else:
                async for chunk in audio:
                    if chunk.data:
                        await self._send_mulaw_audio(websocket, chunk.data)
        except Exception:
            logger.exception("Error sending audio to session %s", session.id)

    async def _send_mulaw_audio(self, websocket: Any, pcm_data: bytes) -> None:
        mulaw_data = _pcm16_to_mulaw(pcm_data)
        payload = base64.b64encode(mulaw_data).decode("utf-8")
        await websocket.send_json(
            {
                "event": "media",
                "media": {"payload": payload},
            }
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
) -> None:
    """Mount FastRTC voice endpoints on a FastAPI app.

    This creates the WebSocket endpoint that FastRTC clients connect to.
    The endpoint handles:
    - WebSocket connection/disconnection
    - Audio streaming with mu-law encoding
    - Raw audio frame passthrough to the pipeline

    Args:
        app: FastAPI application.
        backend: The FastRTCVoiceBackend instance.
        path: Base path for voice endpoints (default: /fastrtc).
        session_factory: Async callable(websocket_id) -> VoiceSession that creates
            sessions when clients connect. If not provided, sessions must be
            created manually before clients connect.
    """
    import numpy as np
    from fastrtc import AsyncStreamHandler, Stream

    backend._session_factory = session_factory  # type: ignore[attr-defined]

    class AudioPassthroughHandler(AsyncStreamHandler):  # type: ignore[misc]
        """Passes raw audio frames to the backend's on_audio_received callback."""

        def copy(self) -> AudioPassthroughHandler:
            return AudioPassthroughHandler()

        async def receive(self, frame: tuple[int, Any]) -> None:
            from fastrtc.utils import current_context

            sample_rate, audio_data = frame

            ctx = current_context.get()
            websocket_id = ctx.webrtc_id if ctx else None
            websocket = ctx.websocket if ctx else None

            if not websocket_id:
                return

            # Create session if not exists and we have a factory
            session = backend._find_session_by_websocket_id(websocket_id)
            if not session and backend._session_factory:  # type: ignore[attr-defined]
                try:
                    session = await backend._session_factory(websocket_id)  # type: ignore[attr-defined]
                    if session and websocket:
                        backend._register_websocket(websocket_id, session.id, websocket)
                except Exception:
                    logger.exception("Error creating session")

            if not session:
                return

            # Register websocket if not already registered
            if websocket and session.id not in backend._websockets:
                backend._register_websocket(websocket_id, session.id, websocket)

            # Pass raw audio to pipeline via callback
            backend._handle_audio_frame(websocket_id, audio_data, sample_rate)

        async def emit(self) -> tuple[int, Any]:
            # Yield silence — actual response comes via direct WebSocket send
            sr = backend._output_sample_rate
            return (sr, np.zeros(sr // 10, dtype=np.int16))

    # Create FastRTC stream with passthrough handler
    stream = Stream(
        handler=AudioPassthroughHandler(),
        modality="audio",
        mode="send-receive",
    )

    backend._stream = stream

    stream.mount(app, path=path)
    logger.info("FastRTC voice backend mounted at %s", path)
