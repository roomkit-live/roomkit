"""SIP audio I/O, DTMF, diagnostics, and session lifecycle mixin."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from roomkit.core.task_utils import log_task_exception
from roomkit.models.trace import ProtocolTrace
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends._sip_types import STATS_INTERVAL, SIPSessionState, logger
from roomkit.voice.base import AudioChunk, VoiceSession, VoiceSessionState
from roomkit.voice.pipeline.dtmf.base import DTMFEvent
from roomkit.voice.realtime.pacer import OutboundAudioPacer

__all__ = ["SIPAudioMixin"]


@runtime_checkable
class SIPAudioHost(Protocol):
    """Contract: capabilities a host class must provide for SIPAudioMixin.

    Attributes provided by the host's ``__init__``:
        _aiosipua: The aiosipua module (lazy-imported, no type stubs).
        _transport: The SIP UDP transport.
        _uac: The SIP User Agent Client.
        _rtp_inactivity_timeout: Seconds before RTP inactivity triggers disconnect.
        _session_states: Consolidated per-session state.
        _audio_received_callback: Callback for inbound audio frames.
        _dtmf_callbacks: Callbacks for DTMF digit events.
        _disconnect_callbacks: Callbacks for session disconnect.
        _trace_emitter: Protocol trace emitter callback.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _cleanup_session: Remove all tracking state for a session (SIPCallingMixin).
        _log_task_exception: Done callback for fire-and-forget tasks (SIPCallingMixin).
    """

    _aiosipua: Any
    _transport: Any
    _uac: Any
    _rtp_inactivity_timeout: float
    _session_states: dict[str, SIPSessionState]
    _audio_received_callback: Any
    _dtmf_callbacks: list[Any]
    _disconnect_callbacks: list[Any]
    _trace_emitter: Any

    def _cleanup_session(self, session_id: str) -> None: ...

    @staticmethod
    def _log_task_exception(task: Any) -> None: ...


class SIPAudioMixin:
    """Mixin providing audio I/O, DTMF, diagnostics, and session lifecycle.

    Host contract: :class:`SIPAudioHost`.
    """

    _aiosipua: Any
    _transport: Any
    _uac: Any
    _rtp_inactivity_timeout: float
    _session_states: dict[str, SIPSessionState]
    _audio_received_callback: Any
    _dtmf_callbacks: list[Any]
    _disconnect_callbacks: list[Any]
    _trace_emitter: Any
    _cleanup_session: Any  # see SIPAudioHost — cross-mixin, from SIPCallingMixin
    _log_task_exception: Any  # see SIPAudioHost — cross-mixin, from SIPCallingMixin

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
        """Return a pre-created session by metadata lookup."""
        metadata = metadata or {}
        session_id = metadata.get("session_id")
        state = self._session_states.get(session_id) if session_id else None
        if session_id and state is not None:
            session = state.session
            session.room_id = room_id
            session.channel_id = channel_id

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

        state = self._session_states.get(session.id)
        call = state.incoming_call if state is not None else None
        out_call = state.outgoing_call if state is not None else None
        call_session = state.call_session if state is not None else None

        # Send BYE for incoming call
        if call is not None and self._uac is not None:
            try:
                from aiosipua import DialogState

                if call.dialog.state == DialogState.CONFIRMED:
                    self._uac.send_bye(call.dialog, call.source_addr)

                    if self._trace_emitter is not None:
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

        if call_session is not None:
            await call_session.close()

        self._cleanup_session(session.id)
        session.state = VoiceSessionState.ENDED
        logger.info("SIP session disconnected: session=%s", session.id)

    def get_session(self, session_id: str) -> VoiceSession | None:
        state = self._session_states.get(session_id)
        return state.session if state is not None else None

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [
            st.session for st in self._session_states.values() if st.session.room_id == room_id
        ]

    def get_codec_rate(self, session_id: str) -> int:
        """Return the negotiated codec sample rate for a session (8000 or 16000)."""
        state = self._session_states.get(session_id)
        return state.codec_rate if state is not None else 8000

    # -------------------------------------------------------------------------
    # Audio diagnostics
    # -------------------------------------------------------------------------

    async def _audio_stats_loop(self) -> None:
        """Periodically log per-session audio diagnostics.

        Also detects RTP inactivity: if no inbound audio arrives for
        longer than ``rtp_inactivity_timeout`` seconds, the session is
        treated as a missed BYE and cleaned up automatically.
        """
        try:
            while True:
                await asyncio.sleep(STATS_INTERVAL)
                now = time.monotonic()
                inactive_sessions: list[tuple[str, SIPSessionState]] = []

                for sid, st in list(self._session_states.items()):
                    stats = st.audio_stats
                    rtp_info = ""
                    cs = st.call_session
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

                    logger.debug(
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

                    if (
                        self._rtp_inactivity_timeout > 0
                        and stats.inbound_packets > 0
                        and (now - stats.inbound_last_ts) > self._rtp_inactivity_timeout
                    ):
                        inactive_sessions.append((sid, st))

                for sid, st in inactive_sessions:
                    idle = now - st.audio_stats.inbound_last_ts
                    logger.warning(
                        "RTP inactivity timeout: session=%s idle=%.1fs "
                        "(threshold=%.0fs) — forcing disconnect",
                        sid[:8],
                        idle,
                        self._rtp_inactivity_timeout,
                    )
                    session = st.session
                    call_session = st.call_session

                    if call_session is not None:
                        with contextlib.suppress(Exception):
                            await call_session.close()

                    self._cleanup_session(sid)

                    for cb in self._disconnect_callbacks:
                        cb(session)

        except asyncio.CancelledError:
            pass

    # -------------------------------------------------------------------------
    # Inbound audio / DTMF handlers
    # -------------------------------------------------------------------------

    def _make_audio_handler(self, session: VoiceSession) -> Any:
        """Create an on_audio callback bound to *session*."""
        state = self._session_states[session.id]
        stats = state.audio_stats

        def _on_audio(pcm_data: bytes, timestamp: int) -> None:
            now = time.monotonic()
            if stats.inbound_packets > 0:
                gap_ms = (now - stats.inbound_last_ts) * 1000
                if gap_ms > 40:
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
                sample_rate=state.codec_rate,
                channels=1,
                sample_width=2,
            )
            self._audio_received_callback(session, frame)

        return _on_audio

    def _make_dtmf_handler(self, session: VoiceSession) -> Any:
        """Create an on_dtmf callback bound to *session*."""
        state = self._session_states[session.id]

        def _on_dtmf(digit: str, duration: int) -> None:
            duration_ms = (duration / state.clock_rate) * 1000
            event = DTMFEvent(digit=digit, duration_ms=duration_ms)
            for cb in self._dtmf_callbacks:
                cb(session, event)

        return _on_dtmf

    # -------------------------------------------------------------------------
    # Outbound DTMF
    # -------------------------------------------------------------------------

    def send_dtmf(self, session: VoiceSession, digit: str, duration_ms: int = 160) -> None:
        state = self._session_states.get(session.id)
        if state is None or state.call_session is None:
            logger.warning("send_dtmf: no call session for %s", session.id)
            return
        state.call_session.send_dtmf(digit, duration_ms)
        logger.info(
            "DTMF sent: digit=%s, duration=%dms, session=%s", digit, duration_ms, session.id
        )

    # -------------------------------------------------------------------------
    # Outbound audio
    # -------------------------------------------------------------------------

    def send_audio_sync(self, session: VoiceSession, chunk: AudioChunk) -> None:
        """Synchronously send a single audio chunk via SIP/RTP."""
        state = self._session_states.get(session.id)
        if state is None or state.call_session is None:
            return
        self._send_pcm_bytes(session, state.call_session, chunk.data)

    def _ensure_pacer(self, session: VoiceSession) -> Any:
        """Return the existing pacer for *session*, or create one."""
        state = self._session_states[session.id]
        if state.pacer is not None:
            return state.pacer

        call_session = state.call_session

        async def rtp_send(data: bytes) -> None:
            self._send_pcm_bytes(session, call_session, data)

        pacer = OutboundAudioPacer(send_fn=rtp_send, sample_rate=state.codec_rate)
        state.pacer = pacer
        task = asyncio.get_running_loop().create_task(
            pacer.start(), name=f"sip_pacer_{session.id}"
        )
        task.add_done_callback(log_task_exception)
        return pacer

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        state = self._session_states.get(session.id)
        if state is None or state.call_session is None:
            logger.warning("send_audio: no call session for %s", session.id)
            return

        pacer = self._ensure_pacer(session)

        if isinstance(audio, bytes):
            state.is_playing = True
            pacer.push(audio)
        else:
            state.is_playing = True
            task = asyncio.create_task(self._feed_stream(session, pacer, audio))
            state.playback_task = task
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                if state.playback_task is task:
                    state.playback_task = None
                    state.is_playing = False

    def _send_pcm_bytes(self, session: VoiceSession, call_session: Any, pcm_data: bytes) -> None:
        """Send a PCM-16 LE buffer as RTP packets."""
        state = self._session_states.get(session.id)
        if state is None:
            return

        codec_rate = state.codec_rate
        clock_rate = state.clock_rate
        pcm_samples_per_frame = codec_rate // 50
        bytes_per_frame = pcm_samples_per_frame * 2
        ts_increment = clock_rate // 50

        buf = state.send_buffer
        buf.extend(pcm_data)

        ts = state.send_timestamp

        now = time.monotonic()
        last_send = state.last_rtp_send_time
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

        frames_this_call = 0
        while len(buf) >= bytes_per_frame:
            call_session.send_audio_pcm(bytes(buf[:bytes_per_frame]), ts)
            del buf[:bytes_per_frame]
            ts += ts_increment
            state.send_frame_count += 1
            frames_this_call += 1

        state.send_timestamp = ts
        if frames_this_call > 0:
            state.last_rtp_send_time = now

        stats = state.audio_stats
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
                state = self._session_states.get(session.id)
                if state is None or not state.is_playing:
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
            state = self._session_states.get(session.id)
            codec_rate = state.codec_rate if state is not None else 8000
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
        state = self._session_states.get(session.id)
        if state is not None and state.pacer is not None:
            state.pacer.end_of_response()

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        """Log transcription text (no UI channel in SIP mode)."""
        label = "User" if role == "user" else "Assistant"
        logger.info("[%s] %s", label, text)

    async def cancel_audio(self, session: VoiceSession) -> bool:
        state = self._session_states.get(session.id)
        if state is None:
            return False
        was_playing = state.is_playing
        if was_playing:
            state.is_playing = False
            if state.pacer is not None:
                state.pacer.interrupt()
            if state.playback_task is not None:
                state.playback_task.cancel()
                state.playback_task = None
            logger.info("Audio cancelled for session %s", session.id)
        state.send_buffer.clear()
        return was_playing

    def is_playing(self, session: VoiceSession) -> bool:
        state = self._session_states.get(session.id)
        return state.is_playing if state is not None else False
