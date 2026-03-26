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
from collections.abc import Callable
from typing import Any

from roomkit.voice.backends._sip_types import (
    PT_G722,
    PT_PCMA,
    PT_PCMU,
    CallCallback,
    DTMFReceivedCallback,
    SIPSessionState,
    import_aiosipua,
    import_rtp_bridge,
    logger,
    wrap_async,
)
from roomkit.voice.backends.base import (
    AudioReceivedCallback,
    SessionReadyCallback,
    TransportDisconnectCallback,
    VoiceBackend,
)
from roomkit.voice.backends.sip_audio import SIPAudioMixin
from roomkit.voice.backends.sip_auth import SIPAuthMixin
from roomkit.voice.backends.sip_calling import SIPCallingMixin
from roomkit.voice.base import (
    BargeInCallback,
    VoiceCapability,
)

__all__ = [
    "PT_G722",
    "PT_PCMA",
    "PT_PCMU",
    "SIPVoiceBackend",
]


class SIPVoiceBackend(SIPAuthMixin, SIPCallingMixin, SIPAudioMixin, VoiceBackend):
    """VoiceBackend that handles incoming SIP calls with full lifecycle.

    Listens for SIP INVITE requests, negotiates codecs via SDP, creates
    RTP sessions for audio streaming, and handles BYE/CANCEL for call
    teardown.  Incoming calls are auto-accepted; an ``on_call`` callback
    lets the application route the session to a room.

    Args:
        local_sip_addr: ``(host, port)`` to bind the SIP listener.
        local_rtp_ip: IP address for RTP media binding.
        advertised_ip: Public IP to advertise in SDP ``c=``/``o=`` lines
            and SIP Contact/Via headers when behind NAT.  RTP sockets
            still bind to *local_rtp_ip*.  Default ``None`` (use the
            resolved local IP for everything).
        rtp_port_start: First RTP port to allocate.
        rtp_port_end: Last RTP port in the allocation range.
        supported_codecs: List of payload type numbers to accept
            (default ``[PT_G722, PT_PCMU, PT_PCMA]``).
        dtmf_payload_type: RTP payload type for RFC 4733 DTMF events.
        user_agent: Value for the SIP ``User-Agent`` header in responses.
        server_name: SDP session name (``s=`` line) in answers.
        jitter_capacity: Maximum number of packets the RTP jitter buffer
            can hold.  Default 32 (~640 ms at 20 ms/packet).
        jitter_prefetch: Number of packets to accumulate before starting
            playout.  Default 0 (start immediately, optimised for low
            latency).
        skip_audio_gaps: When ``True`` (default), gaps in the RTP stream
            are skipped rather than filled with silence.
        rtp_inactivity_timeout: Seconds of RTP silence before forcing
            session disconnect (safety net for missed BYE).  Set to 0
            to disable.  Default 30.
        auth_users: Optional mapping of ``username → password`` for
            inbound digest authentication.  When set, incoming INVITEs
            without valid credentials are challenged with 401.
        auth_realm: Realm string used in the ``WWW-Authenticate``
            challenge header (default ``"roomkit"``).
    """

    def __init__(
        self,
        *,
        local_sip_addr: tuple[str, int] = ("0.0.0.0", 5060),  # nosec B104
        local_rtp_ip: str = "0.0.0.0",  # nosec B104
        advertised_ip: str | None = None,
        rtp_port_start: int = 10000,
        rtp_port_end: int = 20000,
        supported_codecs: list[int] | None = None,
        dtmf_payload_type: int = 101,
        user_agent: str | None = None,
        server_name: str = "-",
        jitter_capacity: int = 32,
        jitter_prefetch: int = 0,
        skip_audio_gaps: bool = True,
        rtp_inactivity_timeout: float = 30.0,
        auth_users: dict[str, str] | None = None,
        auth_realm: str = "roomkit",
    ) -> None:
        self._aiosipua = import_aiosipua()
        self._rtp_bridge = import_rtp_bridge()

        self._local_sip_addr = local_sip_addr
        self._local_rtp_ip = local_rtp_ip
        self._advertised_ip = advertised_ip
        self._rtp_port_start = rtp_port_start
        self._rtp_port_end = rtp_port_end
        self._supported_codecs = supported_codecs or [PT_G722, PT_PCMU, PT_PCMA]
        self._dtmf_payload_type = dtmf_payload_type
        self._user_agent = user_agent
        self._server_name = server_name
        self._jitter_capacity = jitter_capacity
        self._jitter_prefetch = jitter_prefetch
        self._skip_audio_gaps = skip_audio_gaps
        self._rtp_inactivity_timeout = rtp_inactivity_timeout

        # Inbound authentication
        self._auth_users = auth_users
        self._auth_realm = auth_realm
        self._auth_nonces: dict[str, float] = {}

        # Outbound registration state
        self._register_params: dict[str, Any] | None = None
        self._register_response_future: asyncio.Future[Any] | None = None
        self._registration_task: asyncio.Task[None] | None = None
        self._registered = False

        # SIP components (created in start())
        self._transport: Any = None
        self._uas: Any = None
        self._uac: Any = None

        # Per-session state
        self._session_states: dict[str, SIPSessionState] = {}
        self._call_to_session: dict[str, str] = {}
        self._pending_reinvite_calls: dict[str, Any] = {}

        # Callback registrations
        self._audio_received_callback: AudioReceivedCallback | None = None
        self._barge_in_callbacks: list[BargeInCallback] = []
        self._dtmf_callbacks: list[DTMFReceivedCallback] = []
        self._session_ready_callbacks: list[SessionReadyCallback] = []
        self._on_call_callback: CallCallback | None = None
        self._disconnect_callbacks: list[CallCallback] = []

        # Background tasks
        self._stats_task: asyncio.Task[None] | None = None

        # Protocol trace emitter
        self._trace_emitter: Callable[..., Any] | None = None

        # Port allocator
        self._available_ports: set[int] = set(range(rtp_port_start, rtp_port_end, 2))
        self._allocated_ports: set[int] = set()

        self._transport_addr_resolved = False

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

        # Wrap transport handler to intercept REGISTER responses
        self._transport.on_message = self._sip_message_handler

        self._stats_task = asyncio.get_running_loop().create_task(
            self._audio_stats_loop(), name="sip_audio_stats"
        )
        logger.info(
            "SIP backend listening on %s:%d",
            self._local_sip_addr[0],
            self._local_sip_addr[1],
        )

    async def close(self) -> None:
        """Disconnect all sessions, unregister, and stop UAS/transport."""
        # Cancel registration renewal
        if self._registration_task is not None:
            self._registration_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._registration_task
            self._registration_task = None

        # Unregister (expires=0) if currently registered
        if self._registered and self._transport is not None:
            try:
                await self._do_register(expires=0)
            except Exception:
                logger.debug("Failed to unregister on close", exc_info=True)
            self._registered = False

        if self._stats_task is not None:
            self._stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stats_task
            self._stats_task = None

        for state in list(self._session_states.values()):
            await self.disconnect(state.session)

        if self._uas is not None:
            await self._uas.stop()
        logger.info("SIP backend closed")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_trace_emitter(self, emitter: Callable[..., Any] | None) -> None:
        self._trace_emitter = emitter

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_callback = callback

    def on_session_ready(self, callback: SessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def on_barge_in(self, callback: BargeInCallback) -> None:
        self._barge_in_callbacks.append(callback)

    def on_dtmf_received(self, callback: DTMFReceivedCallback) -> DTMFReceivedCallback:
        """Register a callback for inbound DTMF digits (RFC 4733)."""
        self._dtmf_callbacks.append(wrap_async(callback))
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
        """
        self._on_call_callback = wrap_async(callback)
        return callback

    def on_call_disconnected(self, callback: CallCallback) -> CallCallback:
        """Register a callback for remote BYE (call hangup)."""
        self._disconnect_callbacks.append(wrap_async(callback))
        return callback

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        """Register callback for client disconnection (base-class API)."""
        self._disconnect_callbacks.append(wrap_async(callback))
