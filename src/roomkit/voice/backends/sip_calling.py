"""SIP inbound/outbound call handling mixin."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from roomkit.models.trace import ProtocolTrace
from roomkit.voice.backends._sip_types import (
    CODEC_INFO,
    PT_PCMU,
    SIPSessionState,
    logger,
    resolve_local_ip,
)
from roomkit.voice.base import VoiceSession, VoiceSessionState

__all__ = ["SIPCallingMixin"]


class SIPCallingMixin:
    """Mixin providing inbound INVITE handling and outbound dial() for SIPVoiceBackend.

    Attribute declarations below are for mypy — actual values are set
    by :class:`SIPVoiceBackend.__init__`.
    """

    _aiosipua: Any
    _rtp_bridge: Any
    _transport: Any
    _uac: Any
    _local_rtp_ip: str
    _advertised_ip: str | None
    _supported_codecs: list[int]
    _dtmf_payload_type: int
    _server_name: str
    _jitter_capacity: int
    _jitter_prefetch: int
    _skip_audio_gaps: bool
    _user_agent: str | None
    _auth_users: dict[str, str] | None
    _session_states: dict[str, SIPSessionState]
    _call_to_session: dict[str, str]
    _pending_reinvite_calls: dict[str, Any]
    _available_ports: set[int]
    _allocated_ports: set[int]
    _transport_addr_resolved: bool
    _trace_emitter: Any
    _on_call_callback: Any
    _session_ready_callbacks: list[Any]
    _disconnect_callbacks: list[Any]
    _validate_invite_auth: Any  # from SIPAuthMixin
    _make_audio_handler: Any  # from SIPAudioMixin
    _make_dtmf_handler: Any  # from SIPAudioMixin

    # -------------------------------------------------------------------------
    # Inbound call handling
    # -------------------------------------------------------------------------

    async def _handle_invite(self, call: Any) -> None:
        """Handle an incoming SIP INVITE."""
        # Detect re-INVITE for an outbound call: the UAS only checks its
        # own _calls for existing dialogs, but outbound calls live in the
        # UAC.  If the Call-ID already maps to a session, route to the
        # re-INVITE handler instead of creating a duplicate session.
        if call.call_id in self._call_to_session:
            self._handle_reinvite(call)
            return

        # Challenge unauthenticated callers when auth is configured
        if self._auth_users and not self._validate_invite_auth(call):
            return

        if call.sdp_offer is None:
            call.reject(488, "Not Acceptable Here")
            return

        rtp_port = self._allocate_rtp_port()
        bind_ip = self._resolve_local_ip(call.source_addr)
        sdp_ip = self._advertised_ip or bind_ip

        # Resolve transport address once so SIP Contact headers use a
        # routable IP instead of 0.0.0.0.  Without this, the remote
        # party (PBX) cannot route BYE back to us and sessions linger.
        if (
            not self._transport_addr_resolved
            and self._transport is not None
            and self._transport.local_addr[0] in ("0.0.0.0", "")  # nosec B104
        ):
            self._transport.local_addr = (sdp_ip, self._transport.local_addr[1])
            self._transport_addr_resolved = True

        try:
            call_session = self._rtp_bridge.CallSession(
                local_ip=bind_ip,
                rtp_port=rtp_port,
                offer=call.sdp_offer,
                advertised_ip=self._advertised_ip,
                supported_codecs=self._supported_codecs,
                dtmf_payload_type=self._dtmf_payload_type,
                session_name=self._server_name,
                jitter_capacity=self._jitter_capacity,
                jitter_prefetch=self._jitter_prefetch,
                skip_audio_gaps=self._skip_audio_gaps,
            )
        except Exception:
            logger.exception("SDP negotiation failed for call %s", call.call_id)
            self._release_rtp_port(rtp_port)
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

        # Extract display name and user from SIP From header
        from_addr = call.invite.from_addr
        caller_display_name = from_addr.display_name if from_addr else None
        caller_user = from_addr.uri.user if from_addr else None

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
                "caller_display_name": caller_display_name,
                "caller_user": caller_user,
                "room_id": room_id,
                "x_headers": call.x_headers,
                "input_sample_rate": codec_rate,
                "output_sample_rate": codec_rate,
                "codec_sample_rate": codec_rate,
            },
        )

        # Store consolidated session state
        state = SIPSessionState(
            session=session,
            call_session=call_session,
            incoming_call=call,
            codec_rate=codec_rate,
            clock_rate=clock_rate,
            rtp_port=rtp_port,
        )
        self._session_states[session.id] = state
        self._call_to_session[call.call_id] = session.id

        # Wire audio callback (after state is stored so handlers can access it)
        call_session.on_audio = self._make_audio_handler(session)
        call_session.on_dtmf = self._make_dtmf_handler(session)

        # Apply any re-INVITE that arrived before session state was ready
        _pending_call = (
            self._pending_reinvite_calls.pop(session.id, None) or state.pending_reinvite_call
        )
        state.pending_reinvite_call = None
        if _pending_call is not None:
            _pending_call.accept(call_session.sdp_answer)
            _pending_sdp = getattr(_pending_call, "sdp_offer", None)
            if _pending_sdp is None:
                _pending_sdp = state.pending_reinvite_sdp
            if _pending_sdp is not None:
                rtp_addr = _pending_sdp.rtp_address
                if rtp_addr is not None and rtp_addr != call_session.remote_addr:
                    call_session.update_remote(rtp_addr)
            state.pending_reinvite_sdp = None

        logger.info(
            "SIP call accepted: session=%s, room=%s, call_id=%s",
            session.id,
            room_id,
            call.call_id,
        )

        # Emit protocol traces for the INVITE + 200 OK
        if self._trace_emitter is not None:
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

        # Audio path is live — fire session ready callbacks
        for cb in self._session_ready_callbacks:
            cb(session)

    def _handle_reinvite(self, call: Any) -> None:
        """Handle a re-INVITE (session timer refresh or media update)."""
        session_id = self._call_to_session.get(call.call_id)
        if session_id is None:
            logger.warning("re-INVITE for unknown call_id: %s", call.call_id)
            return

        state = self._session_states.get(session_id)
        if state is None or state.call_session is None:
            if state is not None:
                state.pending_reinvite_call = call
                if call.sdp_offer is not None:
                    state.pending_reinvite_sdp = call.sdp_offer
            else:
                self._pending_reinvite_calls[session_id] = call
            logger.info("re-INVITE queued for session %s (CallSession pending)", session_id)
            return

        call_session = state.call_session
        call.accept(call_session.sdp_answer)

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
            session = state.session
            if session is not None:
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

        state = self._session_states.get(session_id)
        session = state.session if state is not None else None
        call_session = state.call_session if state is not None else None

        if self._trace_emitter is not None and session is not None:
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
            task = asyncio.get_running_loop().create_task(
                call_session.close(), name=f"sip_close:{session_id}"
            )
            task.add_done_callback(self._log_task_exception)

        self._cleanup_session(session_id)

        if session is not None:
            logger.info("SIP call ended (remote BYE): session=%s", session_id)
            for cb in self._disconnect_callbacks:
                cb(session)

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

        codec_info = CODEC_INFO.get(codec)
        if codec_info is None:
            raise ValueError(f"Unsupported codec payload type: {codec}")
        codec_name, clock_rate, codec_rate = codec_info

        rtp_port = self._allocate_rtp_port()
        bind_ip = self._resolve_local_ip(proxy_addr)
        sdp_ip = self._advertised_ip or bind_ip

        if not self._transport_addr_resolved and self._transport.local_addr[0] in ("0.0.0.0", ""):  # nosec B104
            self._transport.local_addr = (sdp_ip, self._transport.local_addr[1])
            self._transport_addr_resolved = True

        from aiosipua import build_sdp

        sdp_offer = build_sdp(
            local_ip=bind_ip,
            rtp_port=rtp_port,
            payload_type=codec,
            codec_name=codec_name,
            sample_rate=clock_rate,
            dtmf_payload_type=self._dtmf_payload_type,
            advertised_ip=self._advertised_ip,
        )

        out_call = self._uac.send_invite(
            from_uri=from_uri,
            to_uri=to_uri,
            remote_addr=proxy_addr,
            sdp_offer=sdp_offer,
            extra_headers=extra_headers,
            auth=auth,
        )

        if self._trace_emitter is not None:
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

        try:
            await out_call.wait_answered(timeout=timeout)
        except (TimeoutError, RuntimeError):
            self._release_rtp_port(rtp_port)
            self._uac.remove_call(out_call.call_id)
            raise

        self._call_to_session[out_call.call_id] = out_call.call_id

        try:
            call_session = self._rtp_bridge.CallSession(
                local_ip=bind_ip,
                rtp_port=rtp_port,
                offer=out_call.sdp_answer,
                advertised_ip=self._advertised_ip,
                supported_codecs=[codec],
                dtmf_payload_type=self._dtmf_payload_type,
                session_name=self._server_name,
                jitter_capacity=self._jitter_capacity,
                jitter_prefetch=self._jitter_prefetch,
                skip_audio_gaps=self._skip_audio_gaps,
            )
            await call_session.start()
        except Exception:
            self._release_rtp_port(rtp_port)
            self._call_to_session.pop(out_call.call_id, None)
            raise

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
                "input_sample_rate": actual_codec_rate,
                "output_sample_rate": actual_codec_rate,
                "codec_sample_rate": actual_codec_rate,
            },
        )

        state = SIPSessionState(
            session=session,
            call_session=call_session,
            outgoing_call=out_call,
            codec_rate=actual_codec_rate,
            clock_rate=actual_clock_rate,
            rtp_port=rtp_port,
        )
        self._session_states[session.id] = state
        self._call_to_session[out_call.call_id] = session.id

        call_session.on_audio = self._make_audio_handler(session)
        call_session.on_dtmf = self._make_dtmf_handler(session)

        # Apply any re-INVITE that arrived during RTP setup
        pending_call = (
            self._pending_reinvite_calls.pop(session.id, None) or state.pending_reinvite_call
        )
        state.pending_reinvite_call = None
        if pending_call is not None:
            pending_call.accept(call_session.sdp_answer)

        pending_sdp = state.pending_reinvite_sdp
        state.pending_reinvite_sdp = None
        if pending_sdp is None and pending_call is not None:
            pending_sdp = getattr(pending_call, "sdp_offer", None)
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
            bind_ip,
            rtp_port,
            call_session.remote_addr[0],
            call_session.remote_addr[1],
            codec_name,
            actual_codec_rate,
            actual_clock_rate,
        )

        if self._trace_emitter is not None:
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

        if self._on_call_callback is not None:
            self._on_call_callback(session)

        for cb in self._session_ready_callbacks:
            cb(session)

        return session

    # -------------------------------------------------------------------------
    # Session cleanup & port management
    # -------------------------------------------------------------------------

    def _cleanup_session(self, session_id: str) -> None:
        """Remove all tracking state for a session."""
        state = self._session_states.pop(session_id, None)
        if state is None:
            return

        if state.rtp_port is not None:
            self._release_rtp_port(state.rtp_port)

        if state.incoming_call is not None:
            self._call_to_session.pop(state.incoming_call.call_id, None)
        if state.outgoing_call is not None:
            self._call_to_session.pop(state.outgoing_call.call_id, None)

        stats = state.audio_stats
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
        if state.send_frame_count:
            logger.info(
                "SIP session %s: sent %d RTP frames total",
                session_id,
                state.send_frame_count,
            )

        if state.playback_task is not None:
            state.playback_task.cancel()
        if state.pacer is not None:
            state.pacer.interrupt()
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(
                    state.pacer.stop(),
                    name=f"pacer_stop:{session_id}",
                )

        state.session.state = VoiceSessionState.ENDED

    def _allocate_rtp_port(self) -> int:
        """Allocate an even RTP port from the available pool."""
        if not self._available_ports:
            raise RuntimeError(
                f"No RTP ports available (all {len(self._allocated_ports)} ports in use)"
            )
        port = self._available_ports.pop()
        self._allocated_ports.add(port)
        return port

    def _release_rtp_port(self, port: int) -> None:
        """Return an RTP port to the available pool."""
        self._allocated_ports.discard(port)
        self._available_ports.add(port)

    def _resolve_local_ip(self, remote_addr: tuple[str, int]) -> str:
        """Return the local IP to advertise in SDP."""
        return resolve_local_ip(self._local_rtp_ip, remote_addr)

    @staticmethod
    def _log_task_exception(task: asyncio.Task[Any]) -> None:
        """Done callback — log exceptions from fire-and-forget tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("SIP background task %s failed: %s", task.get_name(), exc)
