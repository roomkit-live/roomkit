"""Twilio Media Streams WebSocket voice backend.

Bridges Twilio-style WebSocket audio to RoomKit's VoiceChannel pipeline.
Accepts JSON-framed mu-law 8 kHz audio and converts to/from PCM for the
internal pipeline.

Twilio protocol (JSON over WebSocket)::

    Client -> Server:
        {"event": "connected", "protocol": "...", "conversation_id": "..."}
        {"event": "start", "conversation_id": "..."}
        {"event": "media", "conversation_id": "...", "media": {"payload": "<base64 mulaw>"}}
        {"event": "stop", "conversation_id": "..."}

    Server -> Client:
        {"event": "media", "conversation_id": "...", "media": {"payload": "<base64 mulaw>"}}

Usage::

    from roomkit.voice.backends.twilio_ws import TwilioWebSocketBackend

    backend = TwilioWebSocketBackend(output_sample_rate=24000)
    voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # type: ignore[import-untyped]

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import (
    AudioReceivedCallback,
    SessionReadyCallback,
    TransportDisconnectCallback,
    VoiceBackend,
)
from roomkit.voice.base import AudioChunk, VoiceSession

logger = logging.getLogger("roomkit.voice.backend.twilio_ws")

TWILIO_SAMPLE_RATE = 8000
"""Twilio Media Streams audio rate: 8 kHz mu-law."""


class TwilioWebSocketBackend(VoiceBackend):
    """Voice backend for Twilio-style WebSocket audio streams.

    Accepts any WebSocket object with ``send_json()`` for outbound frames
    and exposes ``feed_twilio_audio()`` for inbound Twilio media payloads.
    The caller is responsible for the WebSocket lifecycle and message
    routing — this backend handles only the audio encoding/decoding and
    session management.

    Args:
        output_sample_rate: PCM sample rate used by the pipeline (default 24000).
    """

    def __init__(self, output_sample_rate: int = 24000) -> None:
        self._output_sample_rate = output_sample_rate
        self._audio_received_cb: AudioReceivedCallback | None = None
        self._session_ready_cb: SessionReadyCallback | None = None
        self._transport_disconnect_cb: TransportDisconnectCallback | None = None
        self._websocket: Any | None = None
        self._sessions: dict[str, VoiceSession] = {}
        # Outbound uses audioop (consistent chunk sizes for Twilio protocol).
        self._outbound_ratecv_state: Any = None
        # Outbound write queue + dedicated writer task — prevents send_json()
        # from blocking inbound receive_json() on the same WebSocket.
        self._write_queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        # Inbound uses soxr stream resampler (high quality, same as Pipecat).
        # Falls back to stateful audioop.ratecv if soxr unavailable.
        self._resample_inbound = self._build_inbound_resampler(output_sample_rate)

    @property
    def name(self) -> str:
        return "twilio-ws"

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        session = VoiceSession(
            id=f"twilio-{room_id}-{participant_id}",
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            metadata=metadata or {},
        )
        self._sessions[session.id] = session
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        self._sessions.pop(session.id, None)
        # Stop the writer task
        if self._write_queue is not None:
            await self._write_queue.put(None)  # sentinel
        if self._writer_task is not None:
            with contextlib.suppress(Exception):
                await self._writer_task
            self._writer_task = None
        # Clear stale state so a reconnect starts clean
        self._write_queue = None
        self._websocket = None
        self._outbound_ratecv_state = None
        self._resample_inbound = self._build_inbound_resampler(self._output_sample_rate)
        if self._transport_disconnect_cb:
            self._transport_disconnect_cb(session)

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        """Encode PCM audio to mu-law and send as Twilio media frames."""
        if not self._websocket:
            return
        if isinstance(audio, bytes):
            await self._send_mulaw_frame(session, audio)
        else:
            async for chunk in audio:
                await self._send_mulaw_frame(session, chunk.data)

    async def _send_mulaw_frame(self, session: VoiceSession, pcm_data: bytes) -> None:
        if not self._websocket or not pcm_data:
            return
        # Resample from pipeline rate to 8 kHz for Twilio (stateful)
        if self._output_sample_rate != TWILIO_SAMPLE_RATE:
            pcm_data, self._outbound_ratecv_state = audioop.ratecv(
                pcm_data,
                2,
                1,
                self._output_sample_rate,
                TWILIO_SAMPLE_RATE,
                self._outbound_ratecv_state,
            )
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        payload = base64.b64encode(mulaw_data).decode("ascii")
        msg = {
            "event": "media",
            "conversation_id": session.room_id,
            "media": {"payload": payload},
        }
        # Queue for async writer instead of sending directly
        if self._write_queue is not None:
            await self._write_queue.put(msg)
        else:
            await self._websocket.send_json(msg)

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_cb = callback

    def on_session_ready(self, callback: SessionReadyCallback) -> None:
        self._session_ready_cb = callback

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._transport_disconnect_cb = callback

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    # ------------------------------------------------------------------
    # Public helpers for the WebSocket handler
    # ------------------------------------------------------------------

    def bind_websocket(self, websocket: Any) -> None:
        """Bind a WebSocket connection for audio I/O.

        Starts a dedicated writer task so that outbound send_json() calls
        never block inbound receive_json() on the same WebSocket.

        The websocket must support ``await ws.send_json(dict)``.
        """
        self._websocket = websocket
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._write_queue = queue

        async def _writer() -> None:
            try:
                while True:
                    msg = await queue.get()
                    if msg is None:
                        break
                    await websocket.send_json(msg)
            except Exception as e:
                logger.debug("Writer task ended: %s", e)

        self._writer_task = asyncio.create_task(_writer())

    def notify_session_ready(self, session: VoiceSession) -> None:
        """Signal that the session's audio path is live."""
        if self._session_ready_cb:
            self._session_ready_cb(session)

    async def feed_twilio_audio(self, session: VoiceSession, mulaw_payload: str) -> None:
        """Decode a Twilio media payload and feed it into the audio pipeline.

        Args:
            session: The voice session this audio belongs to.
            mulaw_payload: Base64-encoded mu-law audio from a Twilio media frame.
        """
        mulaw_data = base64.b64decode(mulaw_payload)
        pcm_8k = audioop.ulaw2lin(mulaw_data, 2)
        pcm_out = self._resample_inbound(pcm_8k)
        # Stream resamplers may buffer internally — skip empty output.
        if pcm_out and self._audio_received_cb:
            frame = AudioFrame(
                data=pcm_out,
                sample_rate=self._output_sample_rate,
                channels=1,
                sample_width=2,
            )
            self._audio_received_cb(session, frame)

    @staticmethod
    def _build_inbound_resampler(output_rate: int) -> Callable[[bytes], bytes]:
        """Build a stateful inbound resampler (soxr preferred, audioop fallback)."""
        if output_rate == TWILIO_SAMPLE_RATE:
            return lambda data: data
        try:
            import numpy as np
            import soxr

            stream = soxr.ResampleStream(TWILIO_SAMPLE_RATE, output_rate, 1, dtype=np.int16)
            logger.info(
                "Inbound resampler: soxr stream (%d -> %d Hz)", TWILIO_SAMPLE_RATE, output_rate
            )

            def _soxr(data: bytes) -> bytes:
                out = stream.resample_chunk(np.frombuffer(data, dtype=np.int16))
                return out.tobytes()

            return _soxr
        except ImportError:
            logger.info(
                "Inbound resampler: audioop.ratecv (%d -> %d Hz)", TWILIO_SAMPLE_RATE, output_rate
            )
            state: list[Any] = [None]

            def _ratecv(data: bytes) -> bytes:
                result, state[0] = audioop.ratecv(
                    data, 2, 1, TWILIO_SAMPLE_RATE, output_rate, state[0]
                )
                return result

            return _ratecv
