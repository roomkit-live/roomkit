"""RTP voice backend using aiortp.

This backend sends and receives voice audio over RTP, for integration
with PBX/SIP gateways or any system that speaks RTP.

Requires the ``aiortp`` optional dependency::

    pip install roomkit[rtp]

Usage::

    from roomkit.voice.backends.rtp import RTPVoiceBackend

    backend = RTPVoiceBackend(
        local_addr=("0.0.0.0", 10000),
        remote_addr=("192.168.1.100", 20000),
    )
    voice_channel = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
    kit.register_channel(voice_channel)

    session = await backend.connect("room-1", "user-1", "voice-1")
    # ... pipeline processes inbound RTP audio, AI responds, TTS sends outbound RTP ...
    await backend.disconnect(session)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import AudioReceivedCallback, VoiceBackend
from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
    VoiceCapability,
    VoiceSession,
    VoiceSessionState,
)
from roomkit.voice.pipeline.dtmf.base import DTMFEvent

logger = logging.getLogger("roomkit.voice.rtp")

# Type alias for DTMF callbacks fired by this backend
DTMFReceivedCallback = Callable[["VoiceSession", DTMFEvent], Any]


def _import_aiortp() -> Any:
    """Import aiortp, raising a clear error if missing."""
    try:
        import aiortp

        return aiortp
    except ImportError as exc:
        raise ImportError(
            "aiortp is required for RTPVoiceBackend. Install it with: pip install roomkit[rtp]"
        ) from exc


class RTPVoiceBackend(VoiceBackend):
    """VoiceBackend that sends and receives audio over RTP.

    Each :meth:`connect` call creates a new ``aiortp.RTPSession`` bound to
    the configured local address and sending to the remote address.

    Inbound audio is decoded by aiortp (G.711, L16, Opus) and delivered as
    ``AudioFrame`` objects via the ``on_audio_received`` callback.  DTMF
    digits are received out-of-band via RFC 4733 and delivered via
    ``on_dtmf_received`` callbacks.

    Args:
        local_addr: ``(host, port)`` to bind RTP.  Use port ``0`` for
            OS-assigned.
        remote_addr: ``(host, port)`` to send RTP to.  May be ``None``
            if supplied per-session via ``metadata["remote_addr"]`` in
            :meth:`connect`.
        payload_type: RTP payload type number (default ``0`` = PCMU).
        clock_rate: Clock rate in Hz (default ``8000`` for G.711).
        dtmf_payload_type: RTP payload type for RFC 4733 DTMF events
            (default ``101``).
        rtcp_interval: Seconds between RTCP sender reports.
    """

    def __init__(
        self,
        *,
        local_addr: tuple[str, int] = ("0.0.0.0", 0),  # nosec B104
        remote_addr: tuple[str, int] | None = None,
        payload_type: int = 0,
        clock_rate: int = 8000,
        dtmf_payload_type: int = 101,
        rtcp_interval: float = 5.0,
    ) -> None:
        self._aiortp = _import_aiortp()

        self._local_addr = local_addr
        self._remote_addr = remote_addr
        self._payload_type = payload_type
        self._clock_rate = clock_rate
        self._dtmf_payload_type = dtmf_payload_type
        self._rtcp_interval = rtcp_interval

        # Callback registrations
        self._audio_received_callback: AudioReceivedCallback | None = None
        self._barge_in_callbacks: list[BargeInCallback] = []
        self._dtmf_callbacks: list[DTMFReceivedCallback] = []

        # Session tracking
        self._sessions: dict[str, VoiceSession] = {}
        self._rtp_sessions: dict[str, Any] = {}  # str -> aiortp.RTPSession

        # Outbound timestamp tracking per session
        self._send_timestamps: dict[str, int] = {}

        # Playback tracking for interruption support
        self._playing_sessions: set[str] = set()
        self._playback_tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def name(self) -> str:
        return "RTP"

    @property
    def capabilities(self) -> VoiceCapability:
        return VoiceCapability.DTMF_SIGNALING | VoiceCapability.INTERRUPTION

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
    ) -> VoiceSession:
        metadata = metadata or {}

        # Determine remote address: per-session override or constructor default
        remote_addr = metadata.pop("remote_addr", None) or self._remote_addr
        if remote_addr is None:
            raise ValueError(
                "remote_addr must be provided either in RTPVoiceBackend() "
                "constructor or via metadata['remote_addr'] in connect()"
            )

        rtp_session = await self._aiortp.RTPSession.create(
            local_addr=self._local_addr,
            remote_addr=remote_addr,
            payload_type=self._payload_type,
            clock_rate=self._clock_rate,
            dtmf_payload_type=self._dtmf_payload_type,
            rtcp_interval=self._rtcp_interval,
        )

        session_id = str(uuid.uuid4())
        session_metadata = {
            "payload_type": self._payload_type,
            "clock_rate": self._clock_rate,
            "remote_addr": remote_addr,
            "backend": "rtp",
            **metadata,
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
        self._rtp_sessions[session_id] = rtp_session
        self._send_timestamps[session_id] = 0

        # Wire inbound audio callback
        rtp_session.on_audio = self._make_audio_handler(session)

        # Wire inbound DTMF callback
        rtp_session.on_dtmf = self._make_dtmf_handler(session)

        logger.info(
            "RTP session created: session=%s, room=%s, participant=%s, remote=%s, pt=%d",
            session_id,
            room_id,
            participant_id,
            remote_addr,
            self._payload_type,
        )
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        # Cancel any ongoing playback
        await self.cancel_audio(session)

        rtp_session = self._rtp_sessions.pop(session.id, None)
        if rtp_session is not None:
            await rtp_session.close()

        session.state = VoiceSessionState.ENDED
        self._sessions.pop(session.id, None)
        self._send_timestamps.pop(session.id, None)
        logger.info("RTP session ended: session=%s", session.id)

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        for session in list(self._sessions.values()):
            await self.disconnect(session)

    # -------------------------------------------------------------------------
    # Inbound audio / DTMF handlers
    # -------------------------------------------------------------------------

    def _make_audio_handler(self, session: VoiceSession) -> Any:  # Callable[[bytes, int], None]
        """Create an on_audio callback bound to *session*."""

        def _on_audio(pcm_data: bytes, timestamp: int) -> None:
            if self._audio_received_callback is None:
                return
            frame = AudioFrame(
                data=pcm_data,
                sample_rate=self._clock_rate,
                channels=1,
                sample_width=2,
            )
            self._audio_received_callback(session, frame)

        return _on_audio

    def _make_dtmf_handler(self, session: VoiceSession) -> Any:  # Callable[[str, int], None]
        """Create an on_dtmf callback bound to *session*."""

        def _on_dtmf(digit: str, duration: int) -> None:
            # Convert aiortp duration (RTP timestamp units) to milliseconds
            duration_ms = (duration / self._clock_rate) * 1000
            event = DTMFEvent(
                digit=digit,
                duration_ms=duration_ms,
            )
            for cb in self._dtmf_callbacks:
                cb(session, event)

        return _on_dtmf

    # -------------------------------------------------------------------------
    # Outbound audio
    # -------------------------------------------------------------------------

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        rtp_session = self._rtp_sessions.get(session.id)
        if rtp_session is None:
            logger.warning("send_audio: no RTP session for %s", session.id)
            return

        self._playing_sessions.add(session.id)
        try:
            if isinstance(audio, bytes):
                await asyncio.get_running_loop().run_in_executor(
                    None, self._send_pcm_bytes, session, rtp_session, audio
                )
            else:
                await self._send_pcm_stream(session, rtp_session, audio)
        except Exception:
            logger.exception("Error sending audio for session %s", session.id)
        finally:
            self._playing_sessions.discard(session.id)

    def _send_pcm_bytes(self, session: VoiceSession, rtp_session: Any, pcm_data: bytes) -> None:
        """Send a complete PCM-16 LE buffer as RTP packets."""
        # Frame size: 20ms worth of samples
        samples_per_frame = self._clock_rate // 50  # 20ms frames
        bytes_per_frame = samples_per_frame * 2  # 16-bit samples

        ts = self._send_timestamps.get(session.id, 0)
        offset = 0
        while offset < len(pcm_data):
            chunk = pcm_data[offset : offset + bytes_per_frame]
            rtp_session.send_audio_pcm(chunk, ts)
            ts += samples_per_frame
            offset += bytes_per_frame

        self._send_timestamps[session.id] = ts

    async def _send_pcm_stream(
        self,
        session: VoiceSession,
        rtp_session: Any,
        chunks: AsyncIterator[AudioChunk],
    ) -> None:
        """Stream AudioChunks as RTP packets."""
        samples_per_frame = self._clock_rate // 50  # 20ms frames
        bytes_per_frame = samples_per_frame * 2

        ts = self._send_timestamps.get(session.id, 0)

        async def _run() -> None:
            nonlocal ts
            buf = bytearray()
            async for chunk in chunks:
                if session.id not in self._playing_sessions:
                    return
                if chunk.data:
                    buf.extend(chunk.data)
                # Flush complete frames from buffer
                while len(buf) >= bytes_per_frame:
                    frame_data = bytes(buf[:bytes_per_frame])
                    del buf[:bytes_per_frame]
                    rtp_session.send_audio_pcm(frame_data, ts)
                    ts += samples_per_frame

            # Flush remaining data (may be a partial frame)
            if buf and session.id in self._playing_sessions:
                rtp_session.send_audio_pcm(bytes(buf), ts)
                ts += len(buf) // 2

        task = asyncio.create_task(_run())
        self._playback_tasks[session.id] = task
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._playback_tasks.pop(session.id, None)
            self._send_timestamps[session.id] = ts

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        """Log transcription text (no UI channel in RTP mode)."""
        label = "User" if role == "user" else "Assistant"
        logger.info("[%s] %s", label, text)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_callback = callback

    def on_barge_in(self, callback: BargeInCallback) -> None:
        self._barge_in_callbacks.append(callback)

    def on_dtmf_received(self, callback: DTMFReceivedCallback) -> None:
        """Register a callback for inbound DTMF digits (RFC 4733).

        Args:
            callback: Function called with ``(session, dtmf_event)``.
        """
        self._dtmf_callbacks.append(callback)

    async def cancel_audio(self, session: VoiceSession) -> bool:
        was_playing = session.id in self._playing_sessions
        if was_playing:
            self._playing_sessions.discard(session.id)
            task = self._playback_tasks.pop(session.id, None)
            if task is not None:
                task.cancel()
            logger.info("Audio cancelled for session %s", session.id)
        return was_playing

    def is_playing(self, session: VoiceSession) -> bool:
        return session.id in self._playing_sessions
