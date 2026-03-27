"""Manual VAD support for RealtimeVoiceChannel.

When the transport has a local audio pipeline with VAD, the channel
operates in "manual mode": local VAD drives speech start/end events
instead of relying on the provider's server-side VAD.  The channel
sends ``activityStart`` / ``activityEnd`` signals to the provider so
it knows when the user is speaking.

Threading model:
    VAD callbacks fire on the PortAudio audio thread.  Sync operations
    (interrupt, generation bump, state flags) run directly under
    ``_state_lock``.  Async operations (hooks, provider activity
    signals, idle-event updates) are scheduled to the event loop
    via ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.vad.base import VADEventType

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.vad.base import VADEvent

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeVADMixin:
    """Mixin providing manual VAD mode for RealtimeVoiceChannel.

    Expected attributes from the main class:

        _transport, _provider, _manual_vad, _event_loop,
        _user_speaking, _audio_generation, _audio_forward_count,
        _session_resamplers, _state_lock
    """

    _manual_vad: bool

    def _detect_vad_mode(self) -> bool:
        """Check if the transport has a local pipeline with VAD.

        Returns True if manual VAD mode should be used.
        Called from ``start_session`` after ``transport.accept()``.
        """
        has_vad = getattr(self._transport, "has_local_vad", False)
        if has_vad:
            logger.info(
                "Local VAD detected — using manual mode "
                "(local VAD drives speech events, server-side VAD disabled)"
            )
        return has_vad

    def _wire_local_vad(self) -> None:
        """Register VAD callbacks on the transport's pipeline.

        Called from ``start_session`` after ``transport.accept()``
        when manual mode is detected.
        """
        wire = getattr(self._transport, "on_pipeline_vad_event", None)
        if wire is not None:
            wire(self._on_local_vad_event)

    def _on_local_vad_event(self, session: VoiceSession, vad_event: VADEvent) -> None:
        """Handle VAD event from local pipeline (PortAudio thread).

        Only SPEECH_START and SPEECH_END are acted on; other events
        (SILENCE, AUDIO_LEVEL) are ignored.
        """
        if vad_event.type == VADEventType.SPEECH_START:
            self._on_local_speech_start(session)
        elif vad_event.type == VADEventType.SPEECH_END:
            self._on_local_speech_end(session)

    # ------------------------------------------------------------------
    # Speech start (PortAudio thread)
    # ------------------------------------------------------------------

    def _on_local_speech_start(self, session: VoiceSession) -> None:
        """Handle local VAD speech start.

        Sync operations (state flags, generation bump, interrupt) run
        on the PortAudio thread under ``_state_lock``.  Async operations
        (hooks, provider signals, idle-event) are scheduled to the
        event loop via ``call_soon_threadsafe``.
        """
        with self._state_lock:
            self._user_speaking[session.id] = True
            self._audio_generation[session.id] = self._audio_generation.get(session.id, 0) + 1
            resamplers = self._session_resamplers.get(session.id)
            is_barge_in = self._audio_forward_count.get(session.id, 0) > 0

        if resamplers:
            resamplers[1].reset()

        self._transport.interrupt(session)

        loop: asyncio.AbstractEventLoop | None = self._event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._schedule_local_speech_start, session, is_barge_in)

    def _schedule_local_speech_start(self, session: VoiceSession, is_barge_in: bool) -> None:
        """Schedule speech-start tasks on the event loop thread."""
        self._update_idle_event(session.id)

        loop = asyncio.get_running_loop()

        if is_barge_in:
            self._track_task(
                loop,
                self._fire_barge_in_hook(session),
                name=f"rt_barge_in:{session.id}",
            )

        self._track_task(
            loop,
            self._handle_speech_event(session, "start"),
            name=f"rt_speech_start:{session.id}",
        )

        self._track_task(
            loop,
            self._provider.send_activity_start(session),
            name=f"rt_activity_start:{session.id}",
        )

    # ------------------------------------------------------------------
    # Speech end (PortAudio thread)
    # ------------------------------------------------------------------

    def _on_local_speech_end(self, session: VoiceSession) -> None:
        """Handle local VAD speech end.

        Sync state update under ``_state_lock`` on PortAudio thread.
        Async operations scheduled to event loop.
        """
        with self._state_lock:
            self._user_speaking[session.id] = False

        loop: asyncio.AbstractEventLoop | None = self._event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._schedule_local_speech_end, session)

    def _schedule_local_speech_end(self, session: VoiceSession) -> None:
        """Schedule speech-end tasks on the event loop thread."""
        self._update_idle_event(session.id)

        loop = asyncio.get_running_loop()

        self._track_task(
            loop,
            self._handle_speech_event(session, "end"),
            name=f"rt_speech_end:{session.id}",
        )

        self._track_task(
            loop,
            self._provider.send_activity_end(session),
            name=f"rt_activity_end:{session.id}",
        )
