"""SIP voice backend using aiosipua + aiortp.

This backend listens for incoming SIP calls (INVITE), negotiates codecs
via SDP, creates RTP sessions for audio streaming, and handles the full
call lifecycle (BYE, CANCEL).  Calls are routed to roomkit rooms using
X-headers (X-Room-ID, X-Session-ID) set by the PBX/proxy.

Requires the ``aiosipua[rtp]`` optional dependency::

    pip install roomkit[sip]

Usage::

    from roomkit.voice.backends.sip import SIPVoiceBackend

    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", 5060),
        local_rtp_ip="10.0.0.5",
        rtp_port_start=10000,
    )
    backend.on_call(on_incoming_call)
    backend.on_call_disconnected(on_remote_hangup)
    await backend.start()
"""

from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger("roomkit.voice.sip")

# Callback types
DTMFReceivedCallback = Callable[["VoiceSession", DTMFEvent], Any]
CallCallback = Callable[["VoiceSession"], Any]


def _import_aiosipua() -> Any:
    """Import aiosipua, raising a clear error if missing."""
    try:
        import aiosipua

        return aiosipua
    except ImportError as exc:
        raise ImportError(
            "aiosipua is required for SIPVoiceBackend. Install it with: pip install roomkit[sip]"
        ) from exc


def _import_rtp_bridge() -> Any:
    """Import aiosipua.rtp_bridge, raising a clear error if missing."""
    try:
        from aiosipua import rtp_bridge

        return rtp_bridge
    except ImportError as exc:
        raise ImportError(
            "aiosipua[rtp] is required for SIPVoiceBackend. "
            "Install it with: pip install roomkit[sip]"
        ) from exc


class SIPVoiceBackend(VoiceBackend):
    """VoiceBackend that handles incoming SIP calls with full lifecycle.

    Listens for SIP INVITE requests, negotiates codecs via SDP, creates
    RTP sessions for audio streaming, and handles BYE/CANCEL for call
    teardown.  Incoming calls are auto-accepted; an ``on_call`` callback
    lets the application route the session to a room.

    Args:
        local_sip_addr: ``(host, port)`` to bind the SIP listener.
        local_rtp_ip: IP address for RTP media binding.
        rtp_port_start: First RTP port to allocate.
        rtp_port_end: Last RTP port in the allocation range.
        supported_codecs: List of payload type numbers to accept
            (default ``[0, 8]`` = PCMU, PCMA).
        dtmf_payload_type: RTP payload type for RFC 4733 DTMF events.
    """

    def __init__(
        self,
        *,
        local_sip_addr: tuple[str, int] = ("0.0.0.0", 5060),  # nosec B104
        local_rtp_ip: str = "0.0.0.0",  # nosec B104
        rtp_port_start: int = 10000,
        rtp_port_end: int = 20000,
        supported_codecs: list[int] | None = None,
        dtmf_payload_type: int = 101,
    ) -> None:
        self._aiosipua = _import_aiosipua()
        self._rtp_bridge = _import_rtp_bridge()

        self._local_sip_addr = local_sip_addr
        self._local_rtp_ip = local_rtp_ip
        self._rtp_port_start = rtp_port_start
        self._rtp_port_end = rtp_port_end
        self._supported_codecs = supported_codecs or [9, 0, 8]
        self._dtmf_payload_type = dtmf_payload_type

        # SIP components (created in start())
        self._transport: Any = None
        self._uas: Any = None
        self._uac: Any = None

        # Session tracking
        self._sessions: dict[str, VoiceSession] = {}
        self._call_sessions: dict[str, Any] = {}  # session_id -> CallSession
        self._incoming_calls: dict[str, Any] = {}  # session_id -> IncomingCall
        self._call_to_session: dict[str, str] = {}  # call_id -> session_id

        # Per-session codec info (populated after call_session.start())
        self._codec_rates: dict[str, int] = {}  # actual audio sample rate
        self._clock_rates: dict[str, int] = {}  # RTP clock rate

        # Outbound timestamp tracking per session
        self._send_timestamps: dict[str, int] = {}

        # Playback tracking for interruption support
        self._playing_sessions: set[str] = set()
        self._playback_tasks: dict[str, asyncio.Task[None]] = {}

        # Callback registrations
        self._audio_received_callback: AudioReceivedCallback | None = None
        self._barge_in_callbacks: list[BargeInCallback] = []
        self._dtmf_callbacks: list[DTMFReceivedCallback] = []
        self._on_call_callback: CallCallback | None = None
        self._on_disconnect_callback: CallCallback | None = None

        # Port allocator
        self._next_rtp_port = rtp_port_start

    @property
    def name(self) -> str:
        return "SIP"

    @property
    def capabilities(self) -> VoiceCapability:
        return VoiceCapability.DTMF_SIGNALING | VoiceCapability.INTERRUPTION

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the SIP listener and prepare for incoming calls."""
        transport_cls = self._aiosipua.UdpSipTransport
        uas_cls = self._aiosipua.SipUAS
        uac_cls = self._aiosipua.SipUAC

        self._transport = transport_cls(local_addr=self._local_sip_addr)
        self._uas = uas_cls(self._transport)
        self._uac = uac_cls(self._transport)

        self._uas.on_invite = lambda call: asyncio.get_running_loop().create_task(
            self._handle_invite(call)
        )
        self._uas.on_bye = self._handle_bye

        await self._uas.start()
        logger.info(
            "SIP backend listening on %s:%d",
            self._local_sip_addr[0],
            self._local_sip_addr[1],
        )

    async def _handle_invite(self, call: Any) -> None:
        """Handle an incoming SIP INVITE."""
        if call.sdp_offer is None:
            call.reject(488, "Not Acceptable Here")
            return

        rtp_port = self._allocate_rtp_port()

        try:
            call_session = self._rtp_bridge.CallSession(
                local_ip=self._local_rtp_ip,
                rtp_port=rtp_port,
                offer=call.sdp_offer,
                supported_codecs=self._supported_codecs,
                dtmf_payload_type=self._dtmf_payload_type,
            )
        except Exception:
            logger.exception("SDP negotiation failed for call %s", call.call_id)
            call.reject(488, "Not Acceptable Here")
            return

        call.ringing()
        call.accept(call_session.sdp_answer)
        await call_session.start()

        # Store per-session codec info for sample rate awareness
        codec_rate = call_session.codec_sample_rate  # 16000 for G.722, 8000 for G.711
        clock_rate = call_session.clock_rate  # 8000 for both (G.722 RFC 3551 quirk)

        # Extract routing metadata from X-headers
        room_id = call.room_id or call.call_id
        participant_id = call.session_id or call.caller

        session_id = call.session_id or call.call_id
        session = VoiceSession(
            id=session_id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
            metadata={
                "backend": "sip",
                "call_id": call.call_id,
                "caller": call.caller,
                "callee": call.callee,
                "room_id": room_id,
                "x_headers": call.x_headers,
            },
        )

        # Wire audio callback
        call_session.on_audio = self._make_audio_handler(session)
        call_session.on_dtmf = self._make_dtmf_handler(session)

        # Store all mappings
        self._sessions[session.id] = session
        self._call_sessions[session.id] = call_session
        self._incoming_calls[session.id] = call
        self._call_to_session[call.call_id] = session.id
        self._send_timestamps[session.id] = 0
        self._codec_rates[session.id] = codec_rate
        self._clock_rates[session.id] = clock_rate

        logger.info(
            "SIP call accepted: session=%s, room=%s, call_id=%s",
            session.id,
            room_id,
            call.call_id,
        )

        # Fire on_call callback so the app can route to a room
        if self._on_call_callback is not None:
            self._on_call_callback(session)

    def _handle_bye(self, call: Any, request: Any) -> None:
        """Handle a remote BYE (call hangup)."""
        session_id = self._call_to_session.get(call.call_id)
        if session_id is None:
            logger.warning("BYE for unknown call_id: %s", call.call_id)
            return

        session = self._sessions.get(session_id)
        call_session = self._call_sessions.get(session_id)

        if call_session is not None:
            asyncio.get_running_loop().create_task(call_session.close())

        self._cleanup_session(session_id)

        if session is not None:
            session.state = VoiceSessionState.ENDED
            logger.info("SIP call ended (remote BYE): session=%s", session_id)
            if self._on_disconnect_callback is not None:
                self._on_disconnect_callback(session)

    def _cleanup_session(self, session_id: str) -> None:
        """Remove all tracking state for a session."""
        session = self._sessions.pop(session_id, None)
        self._call_sessions.pop(session_id, None)
        call = self._incoming_calls.pop(session_id, None)
        if call is not None:
            self._call_to_session.pop(call.call_id, None)
        self._send_timestamps.pop(session_id, None)
        self._codec_rates.pop(session_id, None)
        self._clock_rates.pop(session_id, None)
        self._playing_sessions.discard(session_id)
        task = self._playback_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
        if session is not None:
            session.state = VoiceSessionState.ENDED

    def _allocate_rtp_port(self) -> int:
        """Allocate the next even RTP port (RTP + RTCP pair)."""
        port = self._next_rtp_port
        self._next_rtp_port += 2
        if self._next_rtp_port >= self._rtp_port_end:
            self._next_rtp_port = self._rtp_port_start
        return port

    # -------------------------------------------------------------------------
    # Session lifecycle (VoiceBackend interface)
    # -------------------------------------------------------------------------

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        """Return a pre-created session by metadata lookup.

        For the SIP backend, sessions are created during INVITE handling.
        ``connect()`` is called after the INVITE handler has already set
        up the session.  Pass ``session_id`` in *metadata* to look up the
        pre-created session.
        """
        metadata = metadata or {}
        session_id = metadata.get("session_id")
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.room_id = room_id
            session.channel_id = channel_id
            return session

        raise ValueError(
            "SIPVoiceBackend.connect() requires metadata['session_id'] "
            "matching a pre-created session from an incoming INVITE."
        )

    async def disconnect(self, session: VoiceSession) -> None:
        """Disconnect a SIP session, sending BYE if the call is still active."""
        await self.cancel_audio(session)

        call = self._incoming_calls.get(session.id)
        call_session = self._call_sessions.get(session.id)

        # Send BYE if dialog is still confirmed
        if call is not None and self._uac is not None:
            try:
                from aiosipua import DialogState

                if call.dialog.state == DialogState.CONFIRMED:
                    self._uac.send_bye(call.dialog, call.source_addr)
            except Exception:
                logger.exception("Failed to send BYE for session %s", session.id)

        # Close RTP session
        if call_session is not None:
            await call_session.close()

        self._cleanup_session(session.id)
        session.state = VoiceSessionState.ENDED
        logger.info("SIP session disconnected: session=%s", session.id)

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        """Disconnect all sessions, stop UAS and transport."""
        for session in list(self._sessions.values()):
            await self.disconnect(session)

        if self._uas is not None:
            await self._uas.stop()
        logger.info("SIP backend closed")

    # -------------------------------------------------------------------------
    # Inbound audio / DTMF handlers
    # -------------------------------------------------------------------------

    def _make_audio_handler(self, session: VoiceSession) -> Any:
        """Create an on_audio callback bound to *session*."""

        def _on_audio(pcm_data: bytes, timestamp: int) -> None:
            if self._audio_received_callback is None:
                return
            frame = AudioFrame(
                data=pcm_data,
                sample_rate=self._codec_rates.get(session.id, 8000),
                channels=1,
                sample_width=2,
            )
            self._audio_received_callback(session, frame)

        return _on_audio

    def _make_dtmf_handler(self, session: VoiceSession) -> Any:
        """Create an on_dtmf callback bound to *session*."""

        def _on_dtmf(digit: str, duration: int) -> None:
            clock_rate = self._clock_rates.get(session.id, 8000)
            duration_ms = (duration / clock_rate) * 1000
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
        call_session = self._call_sessions.get(session.id)
        if call_session is None:
            logger.warning("send_audio: no call session for %s", session.id)
            return

        self._playing_sessions.add(session.id)
        try:
            if isinstance(audio, bytes):
                await asyncio.get_running_loop().run_in_executor(
                    None, self._send_pcm_bytes, session, call_session, audio
                )
            else:
                await self._send_pcm_stream(session, call_session, audio)
        except Exception:
            logger.exception("Error sending audio for session %s", session.id)
        finally:
            self._playing_sessions.discard(session.id)

    def _send_pcm_bytes(self, session: VoiceSession, call_session: Any, pcm_data: bytes) -> None:
        """Send a complete PCM-16 LE buffer as RTP packets."""
        codec_rate = self._codec_rates.get(session.id, 8000)
        clock_rate = self._clock_rates.get(session.id, 8000)
        pcm_samples_per_frame = codec_rate // 50  # 320 for G.722, 160 for G.711
        bytes_per_frame = pcm_samples_per_frame * 2  # 640 for G.722, 320 for G.711
        ts_increment = clock_rate // 50  # 160 for both (G.722 RTP clock = 8000)

        ts = self._send_timestamps.get(session.id, 0)
        offset = 0
        while offset < len(pcm_data):
            chunk = pcm_data[offset : offset + bytes_per_frame]
            call_session.send_audio_pcm(chunk, ts)
            ts += ts_increment
            offset += bytes_per_frame

        self._send_timestamps[session.id] = ts

    async def _send_pcm_stream(
        self,
        session: VoiceSession,
        call_session: Any,
        chunks: AsyncIterator[AudioChunk],
    ) -> None:
        """Stream AudioChunks as RTP packets."""
        codec_rate = self._codec_rates.get(session.id, 8000)
        clock_rate = self._clock_rates.get(session.id, 8000)
        pcm_samples_per_frame = codec_rate // 50
        bytes_per_frame = pcm_samples_per_frame * 2
        ts_increment = clock_rate // 50

        ts = self._send_timestamps.get(session.id, 0)

        async def _run() -> None:
            nonlocal ts
            buf = bytearray()
            async for chunk in chunks:
                if session.id not in self._playing_sessions:
                    return
                if chunk.data:
                    buf.extend(chunk.data)
                while len(buf) >= bytes_per_frame:
                    frame_data = bytes(buf[:bytes_per_frame])
                    del buf[:bytes_per_frame]
                    call_session.send_audio_pcm(frame_data, ts)
                    ts += ts_increment

            if buf and session.id in self._playing_sessions:
                call_session.send_audio_pcm(bytes(buf), ts)
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
        """Log transcription text (no UI channel in SIP mode)."""
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

    def on_call(self, callback: CallCallback) -> None:
        """Register a callback for incoming SIP calls.

        Fired after the INVITE has been accepted and the RTP session is
        active.  Use this to route the session to a room.

        Args:
            callback: Function called with ``(session)``.
        """
        self._on_call_callback = callback

    def on_call_disconnected(self, callback: CallCallback) -> None:
        """Register a callback for remote BYE (call hangup).

        Fired when the remote party sends BYE.  Use this to clean up
        the room and release resources.

        Args:
            callback: Function called with ``(session)``.
        """
        self._on_disconnect_callback = callback

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
