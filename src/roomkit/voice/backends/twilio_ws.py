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

import base64
import logging
from collections.abc import AsyncIterator
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
        # Stateful stream resampler for high-quality inbound audio conversion
        self._inbound_resampler = self._create_resampler()
        self._outbound_resampler = self._create_resampler()

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
        # Resample from pipeline rate to 8 kHz for Twilio
        if self._output_sample_rate != TWILIO_SAMPLE_RATE:
            if self._outbound_resampler is not None:
                pcm_data = self._outbound_resampler.resample(
                    pcm_data, self._output_sample_rate, TWILIO_SAMPLE_RATE
                )
            else:
                pcm_data, _ = audioop.ratecv(
                    pcm_data, 2, 1, self._output_sample_rate, TWILIO_SAMPLE_RATE, None
                )
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        payload = base64.b64encode(mulaw_data).decode("ascii")
        await self._websocket.send_json(
            {
                "event": "media",
                "conversation_id": session.room_id,
                "media": {"payload": payload},
            }
        )

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_cb = callback

    def on_session_ready(self, callback: SessionReadyCallback) -> None:
        self._session_ready_cb = callback

    def on_transport_disconnect(self, callback: TransportDisconnectCallback) -> None:
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

        The websocket must support ``await ws.send_json(dict)``.
        """
        self._websocket = websocket

    def notify_session_ready(self, session: VoiceSession) -> None:
        """Signal that the session's audio path is live."""
        if self._session_ready_cb:
            self._session_ready_cb(session)

    async def feed_twilio_audio(self, session: VoiceSession, mulaw_payload: str) -> None:
        """Decode a Twilio media payload and feed it into the audio pipeline.

        Decodes mu-law to 8 kHz PCM, then uses a stateful soxr stream
        resampler to upsample to the pipeline's output rate with high
        quality and no inter-frame discontinuities.

        Args:
            session: The voice session this audio belongs to.
            mulaw_payload: Base64-encoded mu-law audio from a Twilio media frame.
        """
        mulaw_data = base64.b64decode(mulaw_payload)
        pcm_8k = audioop.ulaw2lin(mulaw_data, 2)

        # Resample with stateful stream resampler (maintains context across calls)
        if self._output_sample_rate != TWILIO_SAMPLE_RATE and self._inbound_resampler is not None:
            pcm_resampled = self._inbound_resampler.resample(
                pcm_8k, TWILIO_SAMPLE_RATE, self._output_sample_rate
            )
            sample_rate = self._output_sample_rate
        else:
            pcm_resampled = pcm_8k
            sample_rate = TWILIO_SAMPLE_RATE

        frame = AudioFrame(
            data=pcm_resampled,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
        )
        if self._audio_received_cb:
            self._audio_received_cb(session, frame)

    @staticmethod
    def _create_resampler() -> Any:
        """Create a stateful soxr stream resampler if available."""
        try:
            import numpy as np
            import soxr

            class _SoxrStreamResampler:
                """Stateful soxr resampler that maintains context across calls."""

                def __init__(self) -> None:
                    self._resampler: Any = None
                    self._in_rate: int = 0
                    self._out_rate: int = 0

                def resample(self, data: bytes, in_rate: int, out_rate: int) -> bytes:
                    if in_rate == out_rate:
                        return data
                    # Recreate resampler if rates changed
                    if self._in_rate != in_rate or self._out_rate != out_rate:
                        self._resampler = soxr.ResampleStream(
                            in_rate,
                            out_rate,
                            1,
                            dtype=np.int16,
                        )
                        self._in_rate = in_rate
                        self._out_rate = out_rate
                    samples = np.frombuffer(data, dtype=np.int16)
                    resampled = self._resampler.resample_chunk(samples)
                    return resampled.tobytes()

            return _SoxrStreamResampler()
        except ImportError:
            logger.warning("soxr/numpy not available — falling back to audioop resampling")
            return None
