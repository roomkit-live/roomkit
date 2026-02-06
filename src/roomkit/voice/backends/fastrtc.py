"""FastRTC VoiceBackend implementation for RoomKit.

This module provides a VoiceBackend that uses FastRTC for WebSocket audio transport
with built-in VAD (Voice Activity Detection).

The backend:
- Handles WebSocket connections from clients
- Uses FastRTC's ReplyOnPause for VAD
- Calls VoiceBackend callbacks (on_speech_start, on_speech_end)
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
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from roomkit.voice.backends.base import VoiceBackend
from roomkit.voice.base import (
    AudioChunk,
    SpeechEndCallback,
    SpeechStartCallback,
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

    This backend uses FastRTC's ReplyOnPause for Voice Activity Detection.
    When speech is detected:
    1. on_speech_start callback is fired
    2. Audio is accumulated until pause is detected
    3. on_speech_end callback is fired with the audio bytes

    The backend handles session management and audio streaming back to clients.
    """

    def __init__(
        self,
        *,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 24000,
    ) -> None:
        """Initialize the FastRTC backend.

        Args:
            input_sample_rate: Expected sample rate of incoming audio (default 48kHz).
            output_sample_rate: Sample rate for outgoing TTS audio (default 24kHz).
        """
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate

        # Callbacks
        self._speech_start_callback: SpeechStartCallback | None = None
        self._speech_end_callback: SpeechEndCallback | None = None

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
        # FastRTC provides VAD via ReplyOnPause
        return VoiceCapability.NONE

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        """Create a new voice session.

        Note: For FastRTC, sessions are created when clients connect via WebSocket.
        This method is called by the application layer after the WebSocket handshake.
        """
        session_id = str(uuid.uuid4())
        # Include audio parameters in metadata for STT to use
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
        self._audio_queues[session_id] = asyncio.Queue()
        logger.info(
            "Voice session created: session=%s, room=%s, participant=%s",
            session_id,
            room_id,
            participant_id,
        )
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        """End a voice session."""
        session.state = VoiceSessionState.ENDED
        self._sessions.pop(session.id, None)
        self._audio_queues.pop(session.id, None)
        self._websockets.pop(session.id, None)
        logger.info("Voice session ended: session=%s", session.id)

    def on_speech_start(self, callback: SpeechStartCallback) -> None:
        """Register callback for speech start events."""
        self._speech_start_callback = callback

    def on_speech_end(self, callback: SpeechEndCallback) -> None:
        """Register callback for speech end events."""
        self._speech_end_callback = callback

    def _resolve_websocket(self, session: VoiceSession) -> Any | None:
        """Resolve WebSocket for a session.

        First checks the explicit registry (populated by _register_websocket).
        Falls back to looking up the websocket from the FastRTC Stream's
        connection registry using the ``websocket_id`` stored in session
        metadata.  This fallback allows TTS to work even when the user is
        muted (no speech event has fired yet to trigger registration).
        """
        ws = self._websockets.get(session.id)
        if ws is not None:
            return ws

        # Fallback: look up via FastRTC Stream connections
        ws_id = session.metadata.get("websocket_id")
        if ws_id and self._stream and hasattr(self._stream, "connections"):
            handlers = self._stream.connections.get(ws_id)
            if handlers and hasattr(handlers[0], "websocket") and handlers[0].websocket:
                # Auto-register so future lookups are O(1)
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
        """Send audio to a voice session.

        For FastRTC, audio is converted to mu-law and sent via WebSocket.
        """
        websocket = self._resolve_websocket(session)
        if not websocket:
            logger.warning("No WebSocket for session %s", session.id)
            return

        try:
            if isinstance(audio, bytes):
                # Single chunk
                await self._send_mulaw_audio(websocket, audio)
            else:
                # Streaming
                async for chunk in audio:
                    if chunk.data:
                        await self._send_mulaw_audio(websocket, chunk.data)
        except Exception:
            logger.exception("Error sending audio to session %s", session.id)

    async def _send_mulaw_audio(self, websocket: Any, pcm_data: bytes) -> None:
        """Convert PCM to mu-law and send via WebSocket."""
        mulaw_data = _pcm16_to_mulaw(pcm_data)

        # Send as base64 JSON
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
        """Send transcription text to the UI via WebSocket."""
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
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        """List all active sessions in a room."""
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        """Release resources."""
        for session in list(self._sessions.values()):
            await self.disconnect(session)

    # -------------------------------------------------------------------------
    # FastRTC integration methods (called by mount_fastrtc_voice)
    # -------------------------------------------------------------------------

    def _handle_speech_start(self, websocket_id: str) -> None:
        """Called by FastRTC when VAD detects speech start."""
        session = self._find_session_by_websocket_id(websocket_id)
        if session and self._speech_start_callback:
            self._speech_start_callback(session)

    def _handle_speech_end(
        self, websocket_id: str, audio_data: np.ndarray, sample_rate: int
    ) -> None:
        """Called by FastRTC when VAD detects speech end with audio."""
        import numpy as _np

        session = self._find_session_by_websocket_id(websocket_id)
        if session and self._speech_end_callback:
            # Convert numpy array to bytes
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            if audio_data.dtype != _np.int16:
                audio_data = (audio_data * 32767).astype(_np.int16)
            audio_bytes = audio_data.tobytes()
            self._speech_end_callback(session, audio_bytes)

    def _register_websocket(self, websocket_id: str, session_id: str, websocket: Any) -> None:
        """Register a WebSocket connection for a session."""
        self._websockets[session_id] = websocket
        # Store websocket_id -> session_id mapping in session metadata
        session = self._sessions.get(session_id)
        if session:
            session.metadata["websocket_id"] = websocket_id

    def _find_session_by_websocket_id(self, websocket_id: str) -> VoiceSession | None:
        """Find session by FastRTC websocket_id."""
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
    - VAD via FastRTC's ReplyOnPause

    Args:
        app: FastAPI application.
        backend: The FastRTCVoiceBackend instance.
        path: Base path for voice endpoints (default: /fastrtc).
        session_factory: Async callable(websocket_id) -> VoiceSession that creates
            sessions when clients connect. If not provided, sessions must be
            created manually before clients connect.
    """
    import numpy as np
    from fastrtc import ReplyOnPause, Stream

    backend._session_factory = session_factory  # type: ignore[attr-defined]

    async def voice_handler(audio: tuple[int, Any]) -> AsyncGenerator[Any, None]:
        """FastRTC handler that bridges to VoiceBackend callbacks.

        This is called by ReplyOnPause when speech is detected and pause occurs.
        Instead of processing here, we call the backend's callbacks which
        trigger VoiceChannel's pipeline (STT -> hooks -> AI -> TTS).
        """
        from fastrtc.utils import current_context

        sample_rate, audio_data = audio

        # Get the websocket_id and websocket from FastRTC context
        ctx = current_context.get()
        websocket_id = ctx.webrtc_id if ctx else None
        websocket = ctx.websocket if ctx else None

        if not websocket_id:
            logger.warning("No websocket_id in context")
            yield (sample_rate, np.zeros(sample_rate // 10, dtype=np.int16))
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
            logger.warning("No session for websocket_id=%s", websocket_id)
            yield (sample_rate, np.zeros(sample_rate // 10, dtype=np.int16))
            return

        # Register websocket if not already registered
        if websocket and session.id not in backend._websockets:
            backend._register_websocket(websocket_id, session.id, websocket)

        logger.info(
            "Speech ended: session=%s, websocket_id=%s, samples=%d",
            session.id,
            websocket_id,
            audio_data.size,
        )

        # Call the backend's speech end handler
        # This triggers VoiceChannel._on_speech_end -> STT -> hooks -> AI
        backend._handle_speech_end(websocket_id, audio_data, sample_rate)

        # The response audio will be sent back via backend.send_audio()
        # which is called by VoiceChannel._deliver_voice()
        # Yield silence - actual response comes via direct WebSocket send
        yield (sample_rate, np.zeros(sample_rate // 10, dtype=np.int16))

    # Create FastRTC stream
    stream = Stream(
        handler=ReplyOnPause(
            voice_handler,
            input_sample_rate=backend._input_sample_rate,
            output_sample_rate=backend._output_sample_rate,
        ),
        modality="audio",
        mode="send-receive",
    )

    backend._stream = stream

    # Mount the stream
    stream.mount(app, path=path)
    logger.info("FastRTC voice backend mounted at %s", path)
