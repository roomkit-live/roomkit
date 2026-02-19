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
    @backend.on_call
    async def handle_call(session):
        await kit.process_inbound(parse_voice_session(session, channel_id="voice"))

    await backend.start()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import time
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

# Well-known RTP payload types (RFC 3551)
PT_PCMU = 0  # G.711 µ-law, 8 kHz
PT_PCMA = 8  # G.711 A-law, 8 kHz
PT_G722 = 9  # G.722, 16 kHz audio (8 kHz RTP clock)

# Codec info: payload_type → (name, rtp_clock_rate, audio_sample_rate)
_CODEC_INFO: dict[int, tuple[str, int, int]] = {
    PT_PCMU: ("PCMU", 8000, 8000),
    PT_PCMA: ("PCMA", 8000, 8000),
    PT_G722: ("G722", 8000, 16000),  # RTP clock 8000, audio rate 16000
}

# Audio diagnostics interval (seconds)
_STATS_INTERVAL = 5.0

# Callback types
DTMFReceivedCallback = Callable[["VoiceSession", DTMFEvent], Any]
CallCallback = Callable[["VoiceSession"], Any]


class _AudioStats:
    """Per-session audio diagnostics counters."""

    __slots__ = (
        "inbound_packets",
        "inbound_bytes",
        "inbound_first_ts",
        "inbound_last_ts",
        "inbound_gaps",
        "inbound_max_gap_ms",
        "outbound_frames",
        "outbound_bytes",
        "outbound_first_ts",
        "outbound_last_ts",
        "outbound_max_burst",
        "outbound_calls",
    )

    def __init__(self) -> None:
        self.inbound_packets = 0
        self.inbound_bytes = 0
        self.inbound_first_ts: float = 0.0
        self.inbound_last_ts: float = 0.0
        self.inbound_gaps = 0
        self.inbound_max_gap_ms: float = 0.0
        self.outbound_frames = 0
        self.outbound_bytes = 0
        self.outbound_first_ts: float = 0.0
        self.outbound_last_ts: float = 0.0
        self.outbound_max_burst = 0  # max frames in a single _send_pcm_bytes call
        self.outbound_calls = 0  # number of _send_pcm_bytes calls


def _wrap_async(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async callback so it can be fired from sync context.

    If *callback* is a coroutine function it is wrapped in
    ``asyncio.create_task``.  Sync callbacks are returned unchanged.
    """
    if asyncio.iscoroutinefunction(callback):
        orig = callback

        def _wrapper(*args: Any, **kwargs: Any) -> None:
            asyncio.get_running_loop().create_task(orig(*args, **kwargs))

        return _wrapper
    return callback


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
            (default ``[PT_G722, PT_PCMU, PT_PCMA]``).
        dtmf_payload_type: RTP payload type for RFC 4733 DTMF events.
        user_agent: Value for the SIP ``User-Agent`` header in responses.
        server_name: SDP session name (``s=`` line) in answers.
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
        user_agent: str | None = None,
        server_name: str = "-",
    ) -> None:
        self._aiosipua = _import_aiosipua()
        self._rtp_bridge = _import_rtp_bridge()

        self._local_sip_addr = local_sip_addr
        self._local_rtp_ip = local_rtp_ip
        self._rtp_port_start = rtp_port_start
        self._rtp_port_end = rtp_port_end
        self._supported_codecs = supported_codecs or [PT_G722, PT_PCMU, PT_PCMA]
        self._dtmf_payload_type = dtmf_payload_type
        self._user_agent = user_agent
        self._server_name = server_name

        # SIP components (created in start())
        self._transport: Any = None
        self._uas: Any = None
        self._uac: Any = None

        # Session tracking
        self._sessions: dict[str, VoiceSession] = {}
        self._call_sessions: dict[str, Any] = {}  # session_id -> CallSession
        self._incoming_calls: dict[str, Any] = {}  # session_id -> IncomingCall
        self._outgoing_calls: dict[str, Any] = {}  # session_id -> OutgoingCall
        self._call_to_session: dict[str, str] = {}  # call_id -> session_id
        self._pending_reinvite_sdp: dict[str, Any] = {}  # session_id -> SdpMessage

        # Per-session codec info (populated after call_session.start())
        self._codec_rates: dict[str, int] = {}  # actual audio sample rate
        self._clock_rates: dict[str, int] = {}  # RTP clock rate

        # Outbound timestamp tracking and PCM carry buffer per session
        self._send_timestamps: dict[str, int] = {}
        self._send_buffers: dict[str, bytearray] = {}
        self._send_frame_count: dict[str, int] = {}  # RTP frames sent per session
        self._last_rtp_send_time: dict[str, float] = {}  # wall-clock of last RTP send

        # Playback tracking for interruption support
        self._playing_sessions: set[str] = set()
        self._playback_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_pacers: dict[str, Any] = {}  # OutboundAudioPacer per session

        # Callback registrations
        self._audio_received_callback: AudioReceivedCallback | None = None
        self._barge_in_callbacks: list[BargeInCallback] = []
        self._dtmf_callbacks: list[DTMFReceivedCallback] = []
        self._on_call_callback: CallCallback | None = None
        self._on_disconnect_callback: CallCallback | None = None

        # Audio diagnostics
        self._audio_stats: dict[str, _AudioStats] = {}
        self._stats_task: asyncio.Task[None] | None = None

        # Protocol trace emitter (set by channel when trace is enabled)
        self._trace_emitter: Callable[..., Any] | None = None

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
        self._uac = uac_cls(self._transport)
        self._uas = uas_cls(self._transport, user_agent=self._user_agent, uac=self._uac)

        self._uas.on_invite = lambda call: asyncio.get_running_loop().create_task(
            self._handle_invite(call)
        )
        self._uas.on_reinvite = self._handle_reinvite
        self._uas.on_bye = self._handle_bye

        await self._uas.start()
        self._stats_task = asyncio.get_running_loop().create_task(
            self._audio_stats_loop(), name="sip_audio_stats"
        )
        logger.info(
            "SIP backend listening on %s:%d",
            self._local_sip_addr[0],
            self._local_sip_addr[1],
        )

    async def _handle_invite(self, call: Any) -> None:
        """Handle an incoming SIP INVITE."""
        # Detect re-INVITE for an outbound call: the UAS only checks its
        # own _calls for existing dialogs, but outbound calls live in the
        # UAC.  If the Call-ID already maps to a session, route to the
        # re-INVITE handler instead of creating a duplicate session.
        if call.call_id in self._call_to_session:
            self._handle_reinvite(call)
            return

        if call.sdp_offer is None:
            call.reject(488, "Not Acceptable Here")
            return

        rtp_port = self._allocate_rtp_port()
        local_ip = self._resolve_local_ip(call.source_addr)

        # Fix SIP Contact header: if transport is bound to 0.0.0.0,
        # replace with the resolved IP so responses use a routable address.
        if self._transport is not None and self._transport.local_addr[0] in ("0.0.0.0", ""):  # nosec B104
            self._transport.local_addr = (local_ip, self._transport.local_addr[1])

        try:
            call_session = self._rtp_bridge.CallSession(
                local_ip=local_ip,
                rtp_port=rtp_port,
                offer=call.sdp_offer,
                supported_codecs=self._supported_codecs,
                dtmf_payload_type=self._dtmf_payload_type,
                session_name=self._server_name,
                jitter_capacity=32,
                jitter_prefetch=0,
                skip_audio_gaps=True,
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
        self._send_frame_count[session.id] = 0
        self._codec_rates[session.id] = codec_rate
        self._clock_rates[session.id] = clock_rate

        logger.info(
            "SIP call accepted: session=%s, room=%s, call_id=%s",
            session.id,
            room_id,
            call.call_id,
        )

        # Emit protocol traces for the INVITE + 200 OK
        if self._trace_emitter is not None:
            from roomkit.models.trace import ProtocolTrace

            invite_raw = call.invite.serialize() if hasattr(call, "invite") else None
            self._trace_emitter(
                ProtocolTrace(
                    channel_id=session.channel_id,
                    direction="inbound",
                    protocol="sip",
                    summary=f"INVITE from {call.caller} to {call.callee}",
                    raw=invite_raw,
                    metadata={
                        "call_id": call.call_id,
                        "caller": call.caller,
                        "callee": call.callee,
                        "x_headers": call.x_headers,
                    },
                    session_id=session.id,
                    room_id=room_id,
                )
            )
            self._trace_emitter(
                ProtocolTrace(
                    channel_id=session.channel_id,
                    direction="outbound",
                    protocol="sip",
                    summary=f"200 OK (codec={codec_rate}Hz, rtp_clock={clock_rate}Hz)",
                    raw=call_session.sdp_answer if hasattr(call_session, "sdp_answer") else None,
                    metadata={
                        "call_id": call.call_id,
                        "codec_sample_rate": codec_rate,
                        "clock_rate": clock_rate,
                        "rtp_port": rtp_port,
                    },
                    session_id=session.id,
                    room_id=room_id,
                )
            )

        # Fire on_call callback so the app can route to a room
        if self._on_call_callback is not None:
            self._on_call_callback(session)

    def _handle_reinvite(self, call: Any) -> None:
        """Handle a re-INVITE (session timer refresh or media update).

        Asterisk and other PBXes send re-INVITEs for session timer
        refresh (RFC 4028) or to update the media path.  We accept
        with the existing SDP answer.  If the re-INVITE contains a
        new SDP offer with an updated RTP address, we update the
        CallSession's remote address accordingly.
        """
        session_id = self._call_to_session.get(call.call_id)
        if session_id is None:
            logger.warning("re-INVITE for unknown call_id: %s", call.call_id)
            return

        call_session = self._call_sessions.get(session_id)
        if call_session is None:
            # CallSession not created yet (race: re-INVITE arrived before
            # dial() finished RTP setup).  Queue the SDP so we can apply
            # the remote address update once the CallSession is ready.
            if call.sdp_offer is not None:
                self._pending_reinvite_sdp[session_id] = call.sdp_offer
            # Accept with our original SDP offer (stored as pending)
            # We need to build a minimal response — the remote expects 200 OK.
            logger.info("re-INVITE queued for session %s (CallSession pending)", session_id)
            return

        # Accept with the existing SDP answer
        call.accept(call_session.sdp_answer)

        # If the re-INVITE updated the remote RTP address, apply it
        if call.sdp_offer is not None:
            rtp_addr = call.sdp_offer.rtp_address
            if rtp_addr is not None and rtp_addr != call_session.remote_addr:
                logger.info(
                    "re-INVITE updated RTP target: %s → %s (session=%s)",
                    call_session.remote_addr,
                    rtp_addr,
                    session_id,
                )
                call_session.update_remote(rtp_addr)

        logger.info("re-INVITE accepted for session %s", session_id)

        if self._trace_emitter is not None:
            session = self._sessions.get(session_id)
            if session is not None:
                from roomkit.models.trace import ProtocolTrace

                self._trace_emitter(
                    ProtocolTrace(
                        channel_id=session.channel_id,
                        direction="inbound",
                        protocol="sip",
                        summary=f"re-INVITE from {call.caller} (session refresh)",
                        raw=None,
                        metadata={"call_id": call.call_id},
                        session_id=session.id,
                        room_id=session.metadata.get("room_id"),
                    )
                )

    def _handle_bye(self, call: Any, request: Any) -> None:
        """Handle a remote BYE (call hangup)."""
        session_id = self._call_to_session.get(call.call_id)
        if session_id is None:
            logger.warning("BYE for unknown call_id: %s", call.call_id)
            return

        session = self._sessions.get(session_id)
        call_session = self._call_sessions.get(session_id)

        # Emit trace before cleanup removes session data
        if self._trace_emitter is not None and session is not None:
            from roomkit.models.trace import ProtocolTrace

            bye_raw = request.serialize() if hasattr(request, "serialize") else None
            self._trace_emitter(
                ProtocolTrace(
                    channel_id=session.channel_id,
                    direction="inbound",
                    protocol="sip",
                    summary=f"BYE from {call.caller}",
                    raw=bye_raw,
                    metadata={"call_id": call.call_id},
                    session_id=session.id,
                    room_id=session.metadata.get("room_id"),
                )
            )

        if call_session is not None:
            asyncio.get_running_loop().create_task(call_session.close())

        self._cleanup_session(session_id)

        if session is not None:
            session.state = VoiceSessionState.ENDED
            logger.info("SIP call ended (remote BYE): session=%s", session_id)
            if self._on_disconnect_callback is not None:
                self._on_disconnect_callback(session)

    # -------------------------------------------------------------------------
    # Outbound calling
    # -------------------------------------------------------------------------

    async def dial(
        self,
        to_uri: str,
        from_uri: str,
        proxy_addr: tuple[str, int],
        *,
        room_id: str | None = None,
        channel_id: str = "voice",
        codec: int = PT_PCMU,
        auth: Any | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> VoiceSession:
        """Initiate an outbound SIP call.

        Builds an SDP offer, sends INVITE via the UAC, waits for the
        remote party to answer (200 OK), then sets up an RTP session and
        returns a :class:`VoiceSession`.

        Args:
            to_uri: SIP URI of the callee (e.g. ``"sip:alice@example.com"``).
            from_uri: SIP URI of the caller (e.g. ``"sip:bot@example.com"``).
            proxy_addr: ``(host, port)`` of the outbound SIP proxy.
            room_id: Room ID for the session (defaults to the call ID).
            channel_id: Channel ID for the session.
            codec: RTP payload type number (default :data:`PT_PCMU`).
            auth: Optional :class:`~aiosipua.SipDigestAuth` for 401/407 retry.
            extra_headers: Extra SIP headers to include in the INVITE.
            timeout: Seconds to wait for the call to be answered.

        Returns:
            An active :class:`VoiceSession` for the established call.

        Raises:
            RuntimeError: If the call is rejected or the backend is not started.
            TimeoutError: If the call is not answered within *timeout* seconds.
        """
        if self._uac is None or self._transport is None:
            raise RuntimeError("SIPVoiceBackend.start() must be called before dial()")

        codec_info = _CODEC_INFO.get(codec)
        if codec_info is None:
            raise ValueError(f"Unsupported codec payload type: {codec}")
        codec_name, clock_rate, codec_rate = codec_info

        rtp_port = self._allocate_rtp_port()
        local_ip = self._resolve_local_ip(proxy_addr)

        # Fix SIP Contact/Via: if transport is bound to 0.0.0.0,
        # replace with the resolved IP so INVITE uses a routable address.
        if self._transport.local_addr[0] in ("0.0.0.0", ""):  # nosec B104
            self._transport.local_addr = (local_ip, self._transport.local_addr[1])

        # Build SDP offer
        from aiosipua import build_sdp

        sdp_offer = build_sdp(
            local_ip=local_ip,
            rtp_port=rtp_port,
            payload_type=codec,
            codec_name=codec_name,
            sample_rate=clock_rate,
            dtmf_payload_type=self._dtmf_payload_type,
        )

        # Send INVITE
        out_call = self._uac.send_invite(
            from_uri=from_uri,
            to_uri=to_uri,
            remote_addr=proxy_addr,
            sdp_offer=sdp_offer,
            extra_headers=extra_headers,
            auth=auth,
        )

        # Emit INVITE trace
        if self._trace_emitter is not None:
            from roomkit.models.trace import ProtocolTrace

            self._trace_emitter(
                ProtocolTrace(
                    channel_id=channel_id,
                    direction="outbound",
                    protocol="sip",
                    summary=f"INVITE from {from_uri} to {to_uri}",
                    raw=None,
                    metadata={
                        "call_id": out_call.call_id,
                        "from_uri": from_uri,
                        "to_uri": to_uri,
                    },
                    session_id=None,
                    room_id=room_id,
                )
            )

        # Wait for answer or rejection
        try:
            await out_call.wait_answered(timeout=timeout)
        except (TimeoutError, RuntimeError):
            self._uac.remove_call(out_call.call_id)
            raise

        # Store call_id → session_id mapping immediately so that
        # re-INVITEs arriving before full session setup are not ignored.
        self._call_to_session[out_call.call_id] = out_call.call_id

        # 200 OK received — set up RTP session using answer SDP
        call_session = self._rtp_bridge.CallSession(
            local_ip=local_ip,
            rtp_port=rtp_port,
            offer=out_call.sdp_answer,
            supported_codecs=[codec],
            dtmf_payload_type=self._dtmf_payload_type,
            session_name=self._server_name,
            jitter_capacity=32,
            jitter_prefetch=0,
            skip_audio_gaps=True,
        )
        await call_session.start()

        actual_codec_rate = call_session.codec_sample_rate
        actual_clock_rate = call_session.clock_rate

        session_id = out_call.call_id
        effective_room_id = room_id or session_id
        session = VoiceSession(
            id=session_id,
            room_id=effective_room_id,
            participant_id=to_uri,
            channel_id=channel_id,
            state=VoiceSessionState.ACTIVE,
            metadata={
                "backend": "sip",
                "call_id": out_call.call_id,
                "caller": from_uri,
                "callee": to_uri,
                "room_id": effective_room_id,
                "direction": "outbound",
            },
        )

        # Wire audio/DTMF callbacks
        call_session.on_audio = self._make_audio_handler(session)
        call_session.on_dtmf = self._make_dtmf_handler(session)

        # Store all mappings
        self._sessions[session.id] = session
        self._call_sessions[session.id] = call_session
        self._outgoing_calls[session.id] = out_call
        self._call_to_session[out_call.call_id] = session.id
        self._send_timestamps[session.id] = 0
        self._send_frame_count[session.id] = 0
        self._codec_rates[session.id] = actual_codec_rate
        self._clock_rates[session.id] = actual_clock_rate

        # Apply any re-INVITE SDP that arrived during RTP setup
        pending_sdp = self._pending_reinvite_sdp.pop(session.id, None)
        if pending_sdp is not None:
            rtp_addr = pending_sdp.rtp_address
            if rtp_addr is not None and rtp_addr != call_session.remote_addr:
                logger.info(
                    "Applying queued re-INVITE RTP target: %s → %s (session=%s)",
                    call_session.remote_addr,
                    rtp_addr,
                    session.id,
                )
                call_session.update_remote(rtp_addr)

        logger.info(
            "SIP outbound call established: session=%s, to=%s, call_id=%s, "
            "local_rtp=%s:%d, remote_rtp=%s:%d, codec=%s(%dHz), clock=%dHz",
            session.id,
            to_uri,
            out_call.call_id,
            local_ip,
            rtp_port,
            call_session.remote_addr[0],
            call_session.remote_addr[1],
            codec_name,
            actual_codec_rate,
            actual_clock_rate,
        )

        # Emit 200 OK trace
        if self._trace_emitter is not None:
            from roomkit.models.trace import ProtocolTrace

            self._trace_emitter(
                ProtocolTrace(
                    channel_id=channel_id,
                    direction="inbound",
                    protocol="sip",
                    summary=f"200 OK (codec={actual_codec_rate}Hz, "
                    f"rtp_clock={actual_clock_rate}Hz)",
                    raw=None,
                    metadata={
                        "call_id": out_call.call_id,
                        "codec_sample_rate": actual_codec_rate,
                        "clock_rate": actual_clock_rate,
                        "rtp_port": rtp_port,
                    },
                    session_id=session.id,
                    room_id=effective_room_id,
                )
            )

        # Fire on_call callback
        if self._on_call_callback is not None:
            self._on_call_callback(session)

        return session

    def _cleanup_session(self, session_id: str) -> None:
        """Remove all tracking state for a session."""
        session = self._sessions.pop(session_id, None)
        self._call_sessions.pop(session_id, None)
        call = self._incoming_calls.pop(session_id, None)
        out_call = self._outgoing_calls.pop(session_id, None)
        if call is not None:
            self._call_to_session.pop(call.call_id, None)
        if out_call is not None:
            self._call_to_session.pop(out_call.call_id, None)
        self._pending_reinvite_sdp.pop(session_id, None)
        # Log final audio stats for this session
        stats = self._audio_stats.pop(session_id, None)
        if stats is not None:
            logger.info(
                "SIP session %s final stats: IN pkts=%d bytes=%d gaps=%d"
                " max_gap=%.0fms | OUT frames=%d bytes=%d",
                session_id[:8],
                stats.inbound_packets,
                stats.inbound_bytes,
                stats.inbound_gaps,
                stats.inbound_max_gap_ms,
                stats.outbound_frames,
                stats.outbound_bytes,
            )
        frames_sent = self._send_frame_count.pop(session_id, 0)
        if frames_sent:
            logger.info("SIP session %s: sent %d RTP frames total", session_id, frames_sent)
        self._send_timestamps.pop(session_id, None)
        self._send_buffers.pop(session_id, None)
        self._last_rtp_send_time.pop(session_id, None)
        self._codec_rates.pop(session_id, None)
        self._clock_rates.pop(session_id, None)
        self._playing_sessions.discard(session_id)
        task = self._playback_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
        pacer = self._session_pacers.pop(session_id, None)
        if pacer is not None:
            pacer.interrupt()
            # Schedule proper shutdown so the background task exits cleanly
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(
                    pacer.stop(),
                    name=f"pacer_stop:{session_id}",
                )
        if session is not None:
            session.state = VoiceSessionState.ENDED

    def _allocate_rtp_port(self) -> int:
        """Allocate the next even RTP port (RTP + RTCP pair)."""
        port = self._next_rtp_port
        self._next_rtp_port += 2
        if self._next_rtp_port >= self._rtp_port_end:
            self._next_rtp_port = self._rtp_port_start
        return port

    def _resolve_local_ip(self, remote_addr: tuple[str, int]) -> str:
        """Return the local IP to advertise in SDP.

        If *local_rtp_ip* was set to a specific address, use it as-is.
        Otherwise (``0.0.0.0``), probe the OS routing table by opening a
        UDP socket towards the caller to discover the correct local IP.
        """
        if self._local_rtp_ip and self._local_rtp_ip != "0.0.0.0":  # nosec B104
            return self._local_rtp_ip
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((remote_addr[0], remote_addr[1] or 5060))
                result: str = s.getsockname()[0]
                return result
        except OSError:
            logger.warning(
                "Could not auto-detect local IP for remote %s; "
                "falling back to 0.0.0.0 — set local_rtp_ip explicitly",
                remote_addr,
            )
            return self._local_rtp_ip

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

            from roomkit.telemetry.base import Attr, SpanKind
            from roomkit.telemetry.noop import NoopTelemetryProvider

            telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            with telemetry.span(
                SpanKind.BACKEND_CONNECT,
                "backend.connect",
                room_id=room_id,
                session_id=session_id,
                attributes={Attr.BACKEND_TYPE: "SIP"},
            ):
                pass  # SIP session pre-created during INVITE

            return session

        raise ValueError(
            "SIPVoiceBackend.connect() requires metadata['session_id'] "
            "matching a pre-created session from an incoming INVITE."
        )

    async def disconnect(self, session: VoiceSession) -> None:
        """Disconnect a SIP session, sending BYE if the call is still active."""
        await self.cancel_audio(session)

        call = self._incoming_calls.get(session.id)
        out_call = self._outgoing_calls.get(session.id)
        call_session = self._call_sessions.get(session.id)

        # Send BYE for incoming call
        if call is not None and self._uac is not None:
            try:
                from aiosipua import DialogState

                if call.dialog.state == DialogState.CONFIRMED:
                    self._uac.send_bye(call.dialog, call.source_addr)

                    if self._trace_emitter is not None:
                        from roomkit.models.trace import ProtocolTrace

                        self._trace_emitter(
                            ProtocolTrace(
                                channel_id=session.channel_id,
                                direction="outbound",
                                protocol="sip",
                                summary="BYE (local hangup)",
                                raw=None,
                                metadata={"call_id": call.call_id},
                                session_id=session.id,
                                room_id=session.metadata.get("room_id"),
                            )
                        )
            except Exception:
                logger.exception("Failed to send BYE for session %s", session.id)

        # Send BYE for outgoing call
        if out_call is not None and self._uac is not None:
            try:
                out_call.hangup(self._uac)

                if self._trace_emitter is not None:
                    from roomkit.models.trace import ProtocolTrace

                    self._trace_emitter(
                        ProtocolTrace(
                            channel_id=session.channel_id,
                            direction="outbound",
                            protocol="sip",
                            summary="BYE (local hangup)",
                            raw=None,
                            metadata={"call_id": out_call.call_id},
                            session_id=session.id,
                            room_id=session.metadata.get("room_id"),
                        )
                    )
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
        if self._stats_task is not None:
            self._stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stats_task
            self._stats_task = None

        for session in list(self._sessions.values()):
            await self.disconnect(session)

        if self._uas is not None:
            await self._uas.stop()
        logger.info("SIP backend closed")

    # -------------------------------------------------------------------------
    # Audio diagnostics
    # -------------------------------------------------------------------------

    async def _audio_stats_loop(self) -> None:
        """Periodically log per-session audio diagnostics."""
        try:
            while True:
                await asyncio.sleep(_STATS_INTERVAL)
                for sid, stats in list(self._audio_stats.items()):
                    # Gather RTP-level stats from call session
                    rtp_info = ""
                    cs = self._call_sessions.get(sid)
                    if cs is not None:
                        rtp_stats = cs.stats
                        if rtp_stats:
                            recv = rtp_stats.get("packets_received", 0)
                            lost = rtp_stats.get("packets_lost", 0)
                            jitter = rtp_stats.get("jitter", 0.0)
                            rtp_info = f" rtp_recv={recv} rtp_lost={lost} jitter={jitter:.1f}"

                    in_dur = 0.0
                    if stats.inbound_packets > 1:
                        in_dur = stats.inbound_last_ts - stats.inbound_first_ts

                    out_dur = 0.0
                    if stats.outbound_frames > 1:
                        out_dur = stats.outbound_last_ts - stats.outbound_first_ts

                    logger.info(
                        "SIP audio [%s] IN: pkts=%d bytes=%d gaps=%d"
                        " max_gap=%.0fms dur=%.1fs |"
                        " OUT: frames=%d bytes=%d dur=%.1fs"
                        " max_burst=%d calls=%d%s",
                        sid[:8],
                        stats.inbound_packets,
                        stats.inbound_bytes,
                        stats.inbound_gaps,
                        stats.inbound_max_gap_ms,
                        in_dur,
                        stats.outbound_frames,
                        stats.outbound_bytes,
                        out_dur,
                        stats.outbound_max_burst,
                        stats.outbound_calls,
                        rtp_info,
                    )
        except asyncio.CancelledError:
            pass

    # -------------------------------------------------------------------------
    # Inbound audio / DTMF handlers
    # -------------------------------------------------------------------------

    def _make_audio_handler(self, session: VoiceSession) -> Any:
        """Create an on_audio callback bound to *session*."""
        stats = _AudioStats()
        self._audio_stats[session.id] = stats

        def _on_audio(pcm_data: bytes, timestamp: int) -> None:
            now = time.monotonic()
            # Track inter-packet gap
            if stats.inbound_packets > 0:
                gap_ms = (now - stats.inbound_last_ts) * 1000
                if gap_ms > 40:  # >2x expected 20ms ptime
                    stats.inbound_gaps += 1
                if gap_ms > stats.inbound_max_gap_ms:
                    stats.inbound_max_gap_ms = gap_ms
            else:
                stats.inbound_first_ts = now
            stats.inbound_last_ts = now
            stats.inbound_packets += 1
            stats.inbound_bytes += len(pcm_data)

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

    def _ensure_pacer(self, session: VoiceSession) -> Any:
        """Return the existing pacer for *session*, or create one."""
        from roomkit.voice.realtime.pacer import OutboundAudioPacer

        pacer = self._session_pacers.get(session.id)
        if pacer is not None:
            return pacer

        call_session = self._call_sessions[session.id]
        codec_rate = self._codec_rates.get(session.id, 8000)

        async def rtp_send(data: bytes) -> None:
            self._send_pcm_bytes(session, call_session, data)

        pacer = OutboundAudioPacer(send_fn=rtp_send, sample_rate=codec_rate)
        self._session_pacers[session.id] = pacer
        asyncio.get_running_loop().create_task(pacer.start(), name=f"sip_pacer_{session.id}")
        return pacer

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        call_session = self._call_sessions.get(session.id)
        if call_session is None:
            logger.warning("send_audio: no call session for %s", session.id)
            return

        pacer = self._ensure_pacer(session)

        if isinstance(audio, bytes):
            # Push to pacer (fire-and-forget, used by realtime transport)
            self._playing_sessions.add(session.id)
            pacer.push(audio)
        else:
            # Stream: iterate + push + await flush
            self._playing_sessions.add(session.id)
            task = asyncio.create_task(self._feed_stream(session, pacer, audio))
            self._playback_tasks[session.id] = task
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self._playback_tasks.pop(session.id, None)
                self._playing_sessions.discard(session.id)

    def _send_pcm_bytes(self, session: VoiceSession, call_session: Any, pcm_data: bytes) -> None:
        """Send a PCM-16 LE buffer as RTP packets.

        Buffers partial frames across calls so that only complete 20 ms
        frames are sent.  This avoids codec encoding artifacts and RTP
        timestamp jumps that confuse the remote jitter buffer.

        Between TTS responses there is a silence gap where no audio is
        sent.  The RTP timestamp must still advance during these gaps so
        the remote jitter buffer does not see stale timestamps and drop
        or delay the new audio.
        """
        codec_rate = self._codec_rates.get(session.id, 8000)
        clock_rate = self._clock_rates.get(session.id, 8000)
        pcm_samples_per_frame = codec_rate // 50  # 320 for G.722, 160 for G.711
        bytes_per_frame = pcm_samples_per_frame * 2  # 640 for G.722, 320 for G.711
        ts_increment = clock_rate // 50  # 160 for both (G.722 RTP clock = 8000)

        buf = self._send_buffers.get(session.id)
        if buf is None:
            buf = bytearray()
            self._send_buffers[session.id] = buf
        buf.extend(pcm_data)

        ts = self._send_timestamps.get(session.id, 0)

        # Advance timestamp across silence gaps so the remote jitter
        # buffer sees monotonically increasing timestamps.  Normal
        # pacing sends one frame every ~20 ms; anything beyond 100 ms
        # indicates a silence gap between TTS responses.
        now = time.monotonic()
        last_send = self._last_rtp_send_time.get(session.id)
        if last_send is not None:
            gap = now - last_send
            if gap > 0.100:
                gap_ticks = int(gap * clock_rate)
                ts += gap_ticks
                logger.debug(
                    "RTP timestamp gap: %.0fms → +%d ticks (session %s)",
                    gap * 1000,
                    gap_ticks,
                    session.id[:8],
                )

        frame_count = self._send_frame_count.get(session.id, 0)
        frames_this_call = 0
        while len(buf) >= bytes_per_frame:
            call_session.send_audio_pcm(bytes(buf[:bytes_per_frame]), ts)
            del buf[:bytes_per_frame]
            ts += ts_increment
            frame_count += 1
            frames_this_call += 1

        self._send_timestamps[session.id] = ts
        self._send_frame_count[session.id] = frame_count
        if frames_this_call > 0:
            self._last_rtp_send_time[session.id] = now

        # Track outbound stats
        stats = self._audio_stats.get(session.id)
        if stats is not None:
            stats.outbound_calls += 1
            if frames_this_call > 0:
                if stats.outbound_frames == 0:
                    stats.outbound_first_ts = now
                stats.outbound_last_ts = now
                stats.outbound_frames += frames_this_call
                stats.outbound_bytes += frames_this_call * bytes_per_frame
                if frames_this_call > stats.outbound_max_burst:
                    stats.outbound_max_burst = frames_this_call

    async def _feed_stream(
        self,
        session: VoiceSession,
        pacer: Any,
        chunks: AsyncIterator[AudioChunk],
    ) -> None:
        """Iterate TTS chunks, push to the session pacer, await flush."""
        try:
            chunk_count = 0
            total_bytes = 0
            max_gap_ms = 0.0
            t_start = time.monotonic()
            t_last = t_start

            async for chunk in chunks:
                if session.id not in self._playing_sessions:
                    break
                if chunk.data:
                    now = time.monotonic()
                    if chunk_count > 0:
                        gap_ms = (now - t_last) * 1000
                        if gap_ms > max_gap_ms:
                            max_gap_ms = gap_ms
                    t_last = now
                    chunk_count += 1
                    total_bytes += len(chunk.data)
                    pacer.push(chunk.data)

            dur = time.monotonic() - t_start
            codec_rate = self._codec_rates.get(session.id, 8000)
            audio_ms = total_bytes / (codec_rate * 2) * 1000
            logger.info(
                "TTS stream [%s]: %d chunks, %d bytes (%.0fms audio) in %.1fs, max_gap=%.0fms",
                session.id[:8],
                chunk_count,
                total_bytes,
                audio_ms,
                dur,
                max_gap_ms,
            )

            pacer.end_of_response()
            await pacer.wait_for_response_done()
        except asyncio.CancelledError:
            pass

    def end_of_response(self, session: VoiceSession) -> None:
        """Signal end of an AI response to the session pacer."""
        pacer = self._session_pacers.get(session.id)
        if pacer is not None:
            pacer.end_of_response()

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        """Log transcription text (no UI channel in SIP mode)."""
        label = "User" if role == "user" else "Assistant"
        logger.info("[%s] %s", label, text)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_trace_emitter(self, emitter: Callable[..., Any] | None) -> None:
        self._trace_emitter = emitter

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_callback = callback

    def on_barge_in(self, callback: BargeInCallback) -> None:
        self._barge_in_callbacks.append(callback)

    def on_dtmf_received(self, callback: DTMFReceivedCallback) -> DTMFReceivedCallback:
        """Register a callback for inbound DTMF digits (RFC 4733).

        Accepts both sync and async callbacks.  Can be used as a decorator::

            @backend.on_dtmf_received
            async def handle_dtmf(session, event):
                ...

        Args:
            callback: Function called with ``(session, dtmf_event)``.
        """
        self._dtmf_callbacks.append(_wrap_async(callback))
        return callback

    def on_call(self, callback: CallCallback) -> CallCallback:
        """Register a callback for incoming SIP calls.

        Fired after the INVITE has been accepted and the RTP session is
        active.  Accepts both sync and async callbacks.  Can be used as
        a decorator::

            @backend.on_call
            async def handle_call(session):
                await kit.process_inbound(
                    parse_voice_session(session, channel_id="voice")
                )

        Args:
            callback: Function called with ``(session)``.
        """
        self._on_call_callback = _wrap_async(callback)
        return callback

    def on_call_disconnected(self, callback: CallCallback) -> CallCallback:
        """Register a callback for remote BYE (call hangup).

        Fired when the remote party sends BYE.  Accepts both sync and
        async callbacks.  Can be used as a decorator::

            @backend.on_call_disconnected
            async def handle_disconnect(session):
                ...

        Args:
            callback: Function called with ``(session)``.
        """
        self._on_disconnect_callback = _wrap_async(callback)
        return callback

    async def cancel_audio(self, session: VoiceSession) -> bool:
        was_playing = session.id in self._playing_sessions
        if was_playing:
            self._playing_sessions.discard(session.id)
            # Interrupt the pacer (drains its queue immediately) but keep
            # it in the dict so it is reused for the next response.
            # Popping it would orphan the background _run() task.
            pacer = self._session_pacers.get(session.id)
            if pacer is not None:
                pacer.interrupt()
            task = self._playback_tasks.pop(session.id, None)
            if task is not None:
                task.cancel()
            logger.info("Audio cancelled for session %s", session.id)
        # Flush carry buffer so stale PCM doesn't leak into the next response
        buf = self._send_buffers.get(session.id)
        if buf is not None:
            buf.clear()
        return was_playing

    def is_playing(self, session: VoiceSession) -> bool:
        return session.id in self._playing_sessions
