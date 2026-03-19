"""SIP audio+video backend using aiosipua + aiortp.

Extends :class:`SIPVoiceBackend` with video transport.  When a SIP
INVITE contains an ``m=video`` line the backend negotiates both audio
and video codecs, creates parallel RTP sessions, and delivers video
frames via :meth:`on_video_received`.

Requires ``aiosipua[rtp]``::

    pip install roomkit[sip]

Usage::

    from roomkit.video.backends.sip import SIPVideoBackend

    backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", 5060),
        local_rtp_ip="10.0.0.5",
    )
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from roomkit.video.backends.base import VideoBackend
from roomkit.video.base import (
    VideoChunk,
    VideoDisconnectCallback,
    VideoReceivedCallback,
    VideoSession,
    VideoSessionReadyCallback,
    VideoSessionState,
)
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.backends.sip import SIPVoiceBackend, _SIPSessionState
from roomkit.voice.base import VoiceSession, VoiceSessionState

logger = logging.getLogger("roomkit.video.sip")


def _import_video_bridge() -> Any:
    """Import aiosipua.video_bridge for VideoCallSession."""
    try:
        from aiosipua import video_bridge

        return video_bridge
    except ImportError as exc:
        raise ImportError(
            "aiosipua[rtp] is required for SIPVideoBackend. Install with: pip install roomkit[sip]"
        ) from exc


class SIPVideoBackend(SIPVoiceBackend, VideoBackend):  # type: ignore[misc]
    """SIP backend with audio + video support.

    Audio-only calls (no ``m=video`` in INVITE) are handled by the
    parent :class:`SIPVoiceBackend`.  Calls with video get parallel
    audio and video RTP sessions.

    Args:
        supported_video_codecs: Video codec names to accept
            (default ``["H264"]``).
        **kwargs: Forwarded to :class:`SIPVoiceBackend`.
    """

    def __init__(
        self,
        *,
        supported_video_codecs: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._supported_video_codecs = supported_video_codecs or ["H264"]
        self._video_bridge = _import_video_bridge()

        # Video state (keyed by voice session ID)
        self._video_call_sessions: dict[str, Any] = {}
        self._combined_answers: dict[str, Any] = {}
        self._video_sessions: dict[str, VideoSession] = {}
        self._video_rtp_ports: dict[str, int] = {}
        self._frame_sequences: dict[str, int] = {}

        # Video callbacks
        self._video_received_callback: VideoReceivedCallback | None = None
        self._video_taps: list[VideoReceivedCallback] = []
        self._video_session_ready_callbacks: list[VideoSessionReadyCallback] = []
        self._video_disconnect_callbacks: list[VideoDisconnectCallback] = []

        # Event loop reference (set during start())
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def name(self) -> str:
        return "SIP-AV"

    async def start(self) -> None:
        """Start SIP listener and capture the event loop reference."""
        self._loop = asyncio.get_running_loop()
        await super().start()

    # -------------------------------------------------------------------------
    # INVITE handling
    # -------------------------------------------------------------------------

    async def _handle_invite(self, call: Any) -> None:
        """Dispatch: A/V for video offers, parent for audio-only."""
        if call.sdp_offer is not None and call.sdp_offer.video is not None:
            await self._handle_av_invite(call)
        else:
            await super()._handle_invite(call)

    async def _handle_av_invite(self, call: Any) -> None:
        """Handle an incoming SIP INVITE with audio + video."""
        # Detect re-INVITE for existing call
        if call.call_id in self._call_to_session:
            self._handle_reinvite(call)
            return

        if call.sdp_offer is None:
            call.reject(488, "Not Acceptable Here")
            return

        audio_rtp_port = self._allocate_rtp_port()
        video_rtp_port = self._allocate_rtp_port()
        local_ip = self._resolve_local_ip(call.source_addr)

        # Log what the remote is offering for video
        video_offer = call.sdp_offer.video
        if video_offer is not None:
            codecs_str = ", ".join(
                f"{c.encoding_name}/{c.clock_rate} (pt={c.payload_type})"
                for c in getattr(video_offer, "codecs", [])
            )
            logger.info(
                "Video offer from %s: port=%s, codecs=[%s]",
                call.source_addr,
                getattr(video_offer, "port", "?"),
                codecs_str,
            )

        if (
            not self._transport_addr_resolved
            and self._transport is not None
            and self._transport.local_addr[0] in ("0.0.0.0", "")  # nosec B104
        ):
            self._transport.local_addr = (local_ip, self._transport.local_addr[1])
            self._transport_addr_resolved = True

        # Audio negotiation via CallSession
        try:
            call_session = self._rtp_bridge.CallSession(
                local_ip=local_ip,
                rtp_port=audio_rtp_port,
                offer=call.sdp_offer,
                supported_codecs=self._supported_codecs,
                dtmf_payload_type=self._dtmf_payload_type,
                session_name=self._server_name,
                jitter_capacity=32,
                jitter_prefetch=0,
                skip_audio_gaps=True,
            )
        except Exception:
            logger.exception("Audio SDP negotiation failed: call_id=%s", call.call_id)
            self._release_rtp_port(audio_rtp_port)
            self._release_rtp_port(video_rtp_port)
            call.reject(488, "Not Acceptable Here")
            return

        # Video negotiation via VideoCallSession
        video_call_session = None
        try:
            video_call_session = self._video_bridge.VideoCallSession(
                local_ip=local_ip,
                rtp_port=video_rtp_port,
                offer=call.sdp_offer,
                supported_video_codecs=self._supported_video_codecs,
                session_name=self._server_name,
            )
        except Exception:
            logger.warning("Video negotiation failed, audio-only: call_id=%s", call.call_id)
            self._release_rtp_port(video_rtp_port)
            video_rtp_port = None  # type: ignore[assignment]

        # Build combined SDP answer (mutates audio answer in place,
        # matching negotiate_av_sdp() behavior)
        combined = call_session.sdp_answer
        if video_call_session is not None:
            video_media = video_call_session.sdp_answer.video
            if video_media is not None:
                combined.media.append(video_media)
            logger.info(
                "Video negotiated: pt=%d, remote=%s, local=%s:%d",
                video_call_session.chosen_payload_type,
                video_call_session.remote_addr,
                local_ip,
                video_rtp_port,
            )

        # Accept and start sessions
        call.ringing()
        call.accept(combined)
        await call_session.start()
        if video_call_session is not None:
            await video_call_session.start()
            logger.info(
                "Video RTP session started: listening on %s:%d for remote %s",
                local_ip,
                video_rtp_port,
                video_call_session.remote_addr,
            )

        # Codec info
        codec_rate = call_session.codec_sample_rate
        clock_rate = call_session.clock_rate

        # Routing metadata
        room_id = call.room_id or call.call_id
        participant_id = call.session_id or call.caller
        session_id = call.session_id or call.call_id
        from_addr = call.invite.from_addr

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
                "caller_display_name": from_addr.display_name if from_addr else None,
                "caller_user": from_addr.uri.user if from_addr else None,
                "room_id": room_id,
                "x_headers": call.x_headers,
                "input_sample_rate": codec_rate,
                "output_sample_rate": codec_rate,
                "codec_sample_rate": codec_rate,
                "has_video": video_call_session is not None,
            },
        )

        # Store audio state (parent's structure)
        state = _SIPSessionState(
            session=session,
            call_session=call_session,
            incoming_call=call,
            codec_rate=codec_rate,
            clock_rate=clock_rate,
            rtp_port=audio_rtp_port,
        )
        self._session_states[session.id] = state
        self._call_to_session[call.call_id] = session.id

        # Wire audio/DTMF callbacks
        call_session.on_audio = self._make_audio_handler(session)
        call_session.on_dtmf = self._make_dtmf_handler(session)

        # Store video state
        if video_call_session is not None:
            self._video_call_sessions[session.id] = video_call_session
            self._combined_answers[session.id] = combined
            self._video_rtp_ports[session.id] = video_rtp_port
            self._frame_sequences[session.id] = 0

            # Determine negotiated video codec name from SDP answer
            video_codec = "h264"
            video_answer = video_call_session.sdp_answer.video
            if video_answer is not None:
                for c in getattr(video_answer, "codecs", []):
                    if c.payload_type == video_call_session.chosen_payload_type:
                        video_codec = c.encoding_name.lower()
                        break

            video_session = VideoSession(
                id=session.id,
                room_id=room_id,
                participant_id=participant_id,
                channel_id="voice",
                state=VideoSessionState.ACTIVE,
                metadata={
                    "backend": "sip-av",
                    "call_id": call.call_id,
                    "video_codec": video_codec,
                },
            )
            self._video_sessions[session.id] = video_session
            video_call_session.on_frame = self._make_video_handler(
                session.id,
                video_codec,
            )

        # Handle deferred re-INVITEs
        pending_call = (
            self._pending_reinvite_calls.pop(session.id, None) or state.pending_reinvite_call
        )
        state.pending_reinvite_call = None
        if pending_call is not None:
            pending_call.accept(combined)
            pending_sdp = getattr(pending_call, "sdp_offer", None) or state.pending_reinvite_sdp
            if pending_sdp is not None:
                rtp_addr = pending_sdp.rtp_address
                if rtp_addr is not None and rtp_addr != call_session.remote_addr:
                    call_session.update_remote(rtp_addr)
            state.pending_reinvite_sdp = None

        logger.info(
            "SIP A/V call accepted: session=%s, room=%s, video=%s",
            session.id,
            room_id,
            video_call_session is not None,
        )

        if self._on_call_callback is not None:
            self._on_call_callback(session)
        for cb in self._session_ready_callbacks:
            cb(session)

    # -------------------------------------------------------------------------
    # re-INVITE override
    # -------------------------------------------------------------------------

    def _handle_reinvite(self, call: Any) -> None:
        session_id = self._call_to_session.get(call.call_id)
        if session_id is None or session_id not in self._combined_answers:
            # Audio-only session — delegate to parent
            super()._handle_reinvite(call)
            return

        state = self._session_states.get(session_id)
        if state is None or state.call_session is None:
            super()._handle_reinvite(call)
            return

        # Accept with combined A/V answer
        call.accept(self._combined_answers[session_id])

        if call.sdp_offer is not None:
            # Update audio RTP address
            rtp_addr = call.sdp_offer.rtp_address
            if rtp_addr is not None and rtp_addr != state.call_session.remote_addr:
                state.call_session.update_remote(rtp_addr)

            # Update video RTP address
            video_rtp_addr = call.sdp_offer.video_rtp_address
            vcs = self._video_call_sessions.get(session_id)
            if (
                video_rtp_addr is not None
                and vcs is not None
                and video_rtp_addr != vcs.remote_addr
            ):
                vcs.update_remote(video_rtp_addr)

        logger.info("re-INVITE (A/V) accepted: session=%s", session_id)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def _cleanup_session(self, session_id: str) -> None:
        vcs = self._video_call_sessions.pop(session_id, None)
        if vcs is not None:
            stats = vcs.stats
            logger.info(
                "Video RTP stats for session %s: %s",
                session_id[:8],
                stats,
            )
            loop = self._loop
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    logger.warning("Video session close skipped — no event loop")
            if loop is not None:
                task = loop.create_task(vcs.close(), name=f"sip_video_close:{session_id}")
                task.add_done_callback(self._log_task_exception)

        video_rtp_port = self._video_rtp_ports.pop(session_id, None)
        if video_rtp_port is not None:
            self._release_rtp_port(video_rtp_port)

        video_session = self._video_sessions.pop(session_id, None)
        if video_session is not None:
            video_session.state = VideoSessionState.ENDED
            for cb in self._video_disconnect_callbacks:
                cb(video_session)

        self._combined_answers.pop(session_id, None)
        frame_count = self._frame_sequences.pop(session_id, None)
        if frame_count is not None and frame_count == 0:
            logger.warning(
                "No video frames received for session %s — remote may not be sending video RTP",
                session_id[:8],
            )

        super()._cleanup_session(session_id)

    async def disconnect(self, session: VoiceSession) -> None:  # type: ignore[override]
        await super().disconnect(session)

    # -------------------------------------------------------------------------
    # Inbound video
    # -------------------------------------------------------------------------

    def _make_video_handler(
        self,
        session_id: str,
        codec: str = "h264",
    ) -> Callable[..., None]:
        frame_count = 0

        def _on_frame(nal_data: bytes, timestamp: int, is_keyframe: bool) -> None:
            nonlocal frame_count
            frame_count += 1
            if frame_count == 1:
                logger.info(
                    "First video frame received: session=%s, codec=%s, %d bytes, keyframe=%s",
                    session_id[:8],
                    codec,
                    len(nal_data),
                    is_keyframe,
                )
            if self._video_received_callback is None and not self._video_taps:
                return
            video_session = self._video_sessions.get(session_id)
            if video_session is None:
                return
            seq = self._frame_sequences.get(session_id, 0)
            self._frame_sequences[session_id] = seq + 1
            frame = VideoFrame(
                data=nal_data,
                codec=codec,
                timestamp_ms=timestamp / 90.0,
                keyframe=is_keyframe,
                sequence=seq,
            )
            try:
                if self._video_received_callback is not None:
                    self._video_received_callback(video_session, frame)
                for tap in self._video_taps:
                    tap(video_session, frame)
            except Exception:
                logger.exception(
                    "Error in video callback: session=%s, seq=%d",
                    session_id[:8],
                    seq,
                )

        return _on_frame

    # -------------------------------------------------------------------------
    # Outbound video
    # -------------------------------------------------------------------------

    async def send_video(
        self,
        session: VideoSession,
        video: bytes | AsyncIterator[VideoChunk],
    ) -> None:
        vcs = self._video_call_sessions.get(session.id)
        if vcs is None:
            logger.warning("send_video: no video session for %s", session.id)
            return

        if isinstance(video, bytes):
            vcs.send_frame([video], 0)
        else:
            async for chunk in video:
                ts = int((chunk.timestamp_ms or 0) * 90)
                vcs.send_frame([chunk.data], ts, chunk.keyframe)

    def send_video_sync(self, session: VideoSession, frame: VideoFrame) -> None:
        """Synchronously send a video frame via SIP/RTP.

        Called by the video bridge from the RTP receive thread where
        ``asyncio.get_running_loop()`` is not available.  Calls
        ``vcs.send_frame()`` directly — no event loop required.
        """
        vcs = self._video_call_sessions.get(session.id)
        if vcs is None:
            logger.debug("send_video_sync: no VCS for session %s", session.id[:8])
            return
        ts = int((frame.timestamp_ms or 0) * 90)  # ms → 90kHz RTP clock
        try:
            vcs.send_frame([frame.data], ts, frame.keyframe)
        except Exception:
            logger.exception(
                "send_video_sync: send_frame failed for session %s",
                session.id[:8],
            )

    def request_keyframe(self, session: VideoSession) -> None:
        """Send a PLI (Picture Loss Indication) to the remote endpoint.

        Requests the remote encoder to produce a keyframe so a new
        bridge participant's decoder can start immediately.
        """
        vcs = self._video_call_sessions.get(session.id)
        if vcs is None:
            return
        try:
            vcs.request_keyframe()
            logger.info("PLI sent: session=%s", session.id[:8])
        except Exception:
            logger.debug("Failed to send PLI for session %s", session.id[:8], exc_info=True)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_video_received(self, callback: VideoReceivedCallback) -> None:
        self._video_received_callback = callback

    def add_video_tap(self, callback: VideoReceivedCallback) -> None:
        self._video_taps.append(callback)

    def on_session_ready(self, callback: VideoSessionReadyCallback) -> None:  # type: ignore[override]
        self._video_session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: VideoDisconnectCallback) -> None:  # type: ignore[override]
        self._video_disconnect_callbacks.append(callback)

    def get_video_session(self, session_id: str) -> VideoSession | None:
        return self._video_sessions.get(session_id)

    def list_video_sessions(self, room_id: str) -> list[VideoSession]:
        return [s for s in self._video_sessions.values() if s.room_id == room_id]
