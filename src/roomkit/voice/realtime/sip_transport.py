"""SIP-based realtime audio transport.

Bridges a :class:`~roomkit.voice.backends.sip.SIPVoiceBackend` to the
:class:`RealtimeAudioTransport` interface so that incoming SIP calls can
be wired to a :class:`~roomkit.channels.realtime_voice.RealtimeVoiceChannel`
(e.g. Gemini Live, OpenAI Realtime).

Audio passes through without resampling — sample-rate conversion is
handled by :class:`RealtimeVoiceChannel` via its ``transport_sample_rate``
parameter.

Requires ``roomkit[sip]``.
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.voice.realtime.base import RealtimeSession
from roomkit.voice.realtime.pacer import OutboundAudioPacer
from roomkit.voice.realtime.transport import (
    RealtimeAudioTransport,
    TransportAudioCallback,
    TransportDisconnectCallback,
)

logger = logging.getLogger("roomkit.voice.realtime.sip_transport")


class SIPRealtimeTransport(RealtimeAudioTransport):
    """Audio transport bridging SIP calls to a realtime voice provider.

    Accepts a :class:`~roomkit.voice.backends.sip.SIPVoiceBackend` and
    passes audio through without resampling.  Use
    ``RealtimeVoiceChannel(transport_sample_rate=...)`` for automatic
    sample-rate conversion.

    The ``connection`` argument to :meth:`accept` is the
    :class:`~roomkit.voice.base.VoiceSession` created by the SIP backend
    during INVITE handling.

    Args:
        backend: A started :class:`SIPVoiceBackend`.

    Example::

        backend = SIPVoiceBackend(...)
        transport = SIPRealtimeTransport(backend)

        realtime = RealtimeVoiceChannel(
            "realtime-voice",
            provider=gemini_provider,
            transport=transport,
            input_sample_rate=16000,
            output_sample_rate=24000,
            transport_sample_rate=8000,
        )
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

        # Mapping: realtime_session_id -> VoiceSession (SIP session)
        self._voice_sessions: dict[str, Any] = {}
        # Mapping: realtime_session_id -> RealtimeSession
        self._rt_sessions: dict[str, RealtimeSession] = {}
        # Reverse mapping: voice_session_id -> realtime_session_id
        self._voice_to_rt: dict[str, str] = {}
        # Per-session outbound audio pacers
        self._pacers: dict[str, OutboundAudioPacer] = {}

        self._audio_callbacks: list[TransportAudioCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []

        # Wire into the SIP backend's audio callback
        self._prev_audio_callback = backend._audio_received_callback
        backend.on_audio_received(self._on_sip_audio)

    @property
    def name(self) -> str:
        return "SIPRealtimeTransport"

    async def accept(self, session: RealtimeSession, connection: Any) -> None:
        """Accept a SIP call as a realtime session.

        Args:
            session: The :class:`RealtimeSession` created by the channel.
            connection: The :class:`VoiceSession` from the SIP backend.
        """
        voice_session = connection
        self._voice_sessions[session.id] = voice_session
        self._rt_sessions[session.id] = session
        self._voice_to_rt[voice_session.id] = session.id

        # Expose the negotiated codec sample rate so the channel can
        # create resamplers at the correct rate (16 kHz for G.722, 8 kHz
        # for G.711) instead of relying on a static transport_sample_rate.
        codec_rate = self._backend._codec_rates.get(voice_session.id, 8000)
        session.metadata["transport_sample_rate"] = codec_rate

        # Create and start the outbound audio pacer for this session
        pacer = OutboundAudioPacer(
            send_fn=self._make_send_fn(session),
            sample_rate=codec_rate,
        )
        self._pacers[session.id] = pacer
        await pacer.start()

        logger.info(
            "SIP realtime transport: accepted session %s (SIP %s, codec_rate=%d)",
            session.id,
            voice_session.id,
            codec_rate,
        )

    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        """Enqueue audio for paced delivery to the SIP caller."""
        pacer = self._pacers.get(session.id)
        if pacer is not None:
            pacer.push(audio)

    async def send_message(self, session: RealtimeSession, message: dict[str, Any]) -> None:
        """No-op — SIP has no metadata/signaling channel for JSON messages."""

    def end_of_response(self, session: RealtimeSession) -> None:
        """Signal end of AI response to the pacer."""
        pacer = self._pacers.get(session.id)
        if pacer is not None:
            pacer.end_of_response()

    def interrupt(self, session: RealtimeSession) -> None:
        """Signal interruption — drain pacer queue, stop playback."""
        pacer = self._pacers.get(session.id)
        if pacer is not None:
            pacer.interrupt()

    async def disconnect(self, session: RealtimeSession) -> None:
        """Disconnect a realtime session (does not send SIP BYE)."""
        pacer = self._pacers.pop(session.id, None)
        if pacer is not None:
            await pacer.stop()
        voice_session = self._voice_sessions.pop(session.id, None)
        self._rt_sessions.pop(session.id, None)
        if voice_session is not None:
            self._voice_to_rt.pop(voice_session.id, None)
        logger.info("SIP realtime transport: disconnected session %s", session.id)

    def on_audio_received(self, callback: TransportAudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    async def close(self) -> None:
        """Disconnect all sessions."""
        for session_id in list(self._rt_sessions.keys()):
            session = self._rt_sessions.get(session_id)
            if session:
                await self.disconnect(session)

    # -- Internal --

    def _make_send_fn(self, session: RealtimeSession) -> Any:
        """Create a send function bound to a specific session."""

        async def _send(audio: bytes) -> None:
            voice_session = self._voice_sessions.get(session.id)
            if voice_session is not None and audio:
                await self._backend.send_audio(voice_session, audio)

        return _send

    def _on_sip_audio(self, voice_session: Any, frame: Any) -> None:
        """Handle inbound SIP audio: pass through to callbacks."""
        rt_session_id = self._voice_to_rt.get(voice_session.id)
        if rt_session_id is None:
            return
        rt_session = self._rt_sessions.get(rt_session_id)
        if rt_session is None:
            return

        if frame.data:
            self._fire_audio_callbacks(rt_session, frame.data)

    def _fire_audio_callbacks(self, session: RealtimeSession, audio: bytes) -> None:
        """Fire all registered audio callbacks (sync)."""
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    import asyncio

                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                    else:
                        loop.create_task(result)
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)

    def _fire_disconnect_callbacks(self, session: RealtimeSession) -> None:
        """Fire all registered disconnect callbacks."""
        for cb in self._disconnect_callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    import asyncio

                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                    else:
                        loop.create_task(result)
            except Exception:
                logger.exception("Error in disconnect callback for session %s", session.id)
