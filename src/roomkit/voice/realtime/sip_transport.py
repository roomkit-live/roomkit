"""SIP-based realtime audio transport.

Bridges a :class:`~roomkit.voice.backends.sip.SIPVoiceBackend` to the
:class:`~roomkit.voice.backends.base.VoiceBackend` interface so that incoming SIP calls can
be wired to a :class:`~roomkit.channels.realtime_voice.RealtimeVoiceChannel`
(e.g. Gemini Live, OpenAI Realtime).

Audio passes through without resampling — sample-rate conversion is
handled by :class:`RealtimeVoiceChannel` via its ``transport_sample_rate``
parameter.

Requires ``roomkit[sip]``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from roomkit.voice.backends.base import (
    TransportDisconnectCallback,
    VoiceBackend,
)
from roomkit.voice.base import AudioChunk, VoiceSession

logger = logging.getLogger("roomkit.voice.realtime.sip_transport")

TransportAudioCallback = Callable[["VoiceSession", bytes], Any]


class SIPRealtimeTransport(VoiceBackend):
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
        # Mapping: realtime_session_id -> VoiceSession
        self._rt_sessions: dict[str, VoiceSession] = {}
        # Reverse mapping: voice_session_id -> realtime_session_id
        self._voice_to_rt: dict[str, str] = {}

        self._audio_callbacks: list[TransportAudioCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []

        # Wire into the SIP backend's audio callback
        self._prev_audio_callback = backend._audio_received_callback
        backend.on_audio_received(self._on_sip_audio)

    @property
    def name(self) -> str:
        return "SIPRealtimeTransport"

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        """Accept a SIP call as a realtime session.

        Args:
            session: The :class:`VoiceSession` created by the channel.
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

        logger.info(
            "SIP realtime transport: accepted session %s (SIP %s, codec_rate=%d)",
            session.id,
            voice_session.id,
            codec_rate,
        )

    async def send_audio(
        self, session: VoiceSession, audio: bytes | AsyncIterator[AudioChunk]
    ) -> None:
        """Delegate audio delivery to the SIP backend's session pacer."""
        if not isinstance(audio, bytes):
            return
        voice_session = self._voice_sessions.get(session.id)
        if voice_session is not None:
            await self._backend.send_audio(voice_session, audio)

    async def send_message(self, session: VoiceSession, message: dict[str, Any]) -> None:
        """No-op — SIP has no metadata/signaling channel for JSON messages."""

    def end_of_response(self, session: VoiceSession) -> None:
        """Signal end of AI response to the backend pacer."""
        voice_session = self._voice_sessions.get(session.id)
        if voice_session is not None:
            self._backend.end_of_response(voice_session)

    def interrupt(self, session: VoiceSession) -> None:
        """Signal interruption — delegate to backend cancel_audio."""
        voice_session = self._voice_sessions.get(session.id)
        if voice_session is not None:
            asyncio.get_running_loop().create_task(self._backend.cancel_audio(voice_session))

    async def disconnect(self, session: VoiceSession) -> None:
        """Disconnect a realtime session (does not send SIP BYE)."""
        voice_session = self._voice_sessions.pop(session.id, None)
        self._rt_sessions.pop(session.id, None)
        if voice_session is not None:
            self._voice_to_rt.pop(voice_session.id, None)
        logger.info("SIP realtime transport: disconnected session %s", session.id)

    def on_audio_received(self, callback: TransportAudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    def set_trace_emitter(self, emitter: Any) -> None:
        """Forward trace emitter to the underlying SIP backend."""
        if hasattr(self._backend, "set_trace_emitter"):
            self._backend.set_trace_emitter(emitter)

    async def close(self) -> None:
        """Disconnect all sessions."""
        for session_id in list(self._rt_sessions.keys()):
            session = self._rt_sessions.get(session_id)
            if session:
                await self.disconnect(session)

    # -- Internal --

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

    def _fire_audio_callbacks(self, session: VoiceSession, audio: bytes) -> None:
        """Fire all registered audio callbacks (sync)."""
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                    else:
                        loop.create_task(result)
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)

    def _fire_disconnect_callbacks(self, session: VoiceSession) -> None:
        """Fire all registered disconnect callbacks."""
        for cb in self._disconnect_callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                    else:
                        loop.create_task(result)
            except Exception:
                logger.exception("Error in disconnect callback for session %s", session.id)
