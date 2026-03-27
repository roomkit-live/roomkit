"""Speech events, barge-in detection, and audio level hooks for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.channels.realtime_voice")


@runtime_checkable
class RealtimeSpeechHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeSpeechMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _user_speaking: Whether the user is currently speaking per session.
        _audio_generation: Generation counter per session for stale audio detection.
        _audio_forward_count: Count of audio chunks forwarded per session.
        _barge_in_active: Session IDs with an active barge-in.
        _last_assistant_text: Last assistant utterance per session.
        _session_resamplers: Per-session (inbound, outbound) resampler pairs.
        _has_pipeline_vad: Whether local pipeline VAD is active.
        _last_input_level_at: Timestamp of last input audio level hook.
        _last_output_level_at: Timestamp of last output audio level hook.
        _event_loop: Cached event loop for cross-thread scheduling.
        _framework: The RoomKit framework instance (or None).
        _transport: The voice backend transport.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
        _send_client_message: Send a JSON message to the client UI.
        _update_idle_event: Update the idle signaling event for a session.
        _rt_span_ctx: Get the telemetry span context for a session.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _user_speaking: dict[str, bool]
    _audio_generation: dict[str, int]
    _audio_forward_count: dict[str, int]
    _barge_in_active: set[str]
    _last_assistant_text: dict[str, str]
    _session_resamplers: dict[str, Any]
    _has_pipeline_vad: bool
    _last_input_level_at: float
    _last_output_level_at: float
    _event_loop: asyncio.AbstractEventLoop | None
    _framework: RoomKit | None
    _transport: VoiceBackend

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    async def _send_client_message(self, session: Any, message: dict[str, Any]) -> None: ...

    def _update_idle_event(self, session_id: str) -> None: ...

    def _rt_span_ctx(self, session_id: str) -> tuple[Any, Any]: ...


class RealtimeSpeechMixin:
    """Speech event handling and audio level hooks for RealtimeVoiceChannel.

    Handles both provider-driven speech events (server-side VAD) and
    pipeline-driven speech events (local VAD).  Audio level hooks are
    colocated here because they are only fired from audio/speech paths.

    Host contract: :class:`RealtimeSpeechHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _user_speaking: dict[str, bool]
    _audio_generation: dict[str, int]
    _audio_forward_count: dict[str, int]
    _barge_in_active: set[str]
    _last_assistant_text: dict[str, str]
    _session_resamplers: dict[str, Any]
    _has_pipeline_vad: bool
    _last_input_level_at: float
    _last_output_level_at: float
    _event_loop: asyncio.AbstractEventLoop | None
    _framework: RoomKit | None
    _transport: VoiceBackend

    _track_task: Any  # see RealtimeSpeechHost — cross-mixin
    _send_client_message: Any  # see RealtimeSpeechHost — cross-mixin
    _update_idle_event: Any  # see RealtimeSpeechHost — cross-mixin
    _rt_span_ctx: Any  # see RealtimeSpeechHost — cross-mixin

    # -----------------------------------------------------------------
    # Provider speech events (server-side VAD)
    # -----------------------------------------------------------------

    def _on_provider_speech_start(self, session: VoiceSession) -> Any:
        """Handle speech start from provider's server-side VAD.

        When the channel has a local pipeline with VAD, speech events are
        driven locally — provider callbacks are ignored to prevent duplicates.

        Bumps the generation counter so pending send_audio tasks become
        stale, resets the outbound resampler to discard its buffered frame,
        and signals the transport to interrupt outbound audio.
        """
        if self._has_pipeline_vad:
            return
        self._user_speaking[session.id] = True
        self._update_idle_event(session.id)
        with self._state_lock:
            self._audio_generation[session.id] = self._audio_generation.get(session.id, 0) + 1
            resamplers = self._session_resamplers.get(session.id)
            is_barge_in = self._audio_forward_count.get(session.id, 0) > 0

        if is_barge_in:
            self._barge_in_active.add(session.id)

        if resamplers:
            resamplers[1].reset()

        if is_barge_in:
            logger.debug(
                "Barge-in detected for session %s (forwarded %d chunks) — "
                "keeping AEC filter intact",
                session.id,
                self._audio_forward_count.get(session.id, 0),
            )

        self._transport.interrupt(session)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

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

    def _on_provider_speech_end(self, session: VoiceSession) -> Any:
        """Handle speech end from provider's server-side VAD."""
        if self._has_pipeline_vad:
            return
        self._user_speaking[session.id] = False
        self._update_idle_event(session.id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_speech_event(session, "end"),
            name=f"rt_speech_end:{session.id}",
        )

    # -----------------------------------------------------------------
    # Barge-in + speech hooks
    # -----------------------------------------------------------------

    async def _fire_barge_in_hook(self, session: VoiceSession) -> None:
        """Fire ON_BARGE_IN when user interrupts AI playback."""
        if not self._framework:
            return
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return
        try:
            from roomkit.voice.events import BargeInEvent

            context = await self._framework._build_context(room_id)
            event = BargeInEvent(
                session=session,
                interrupted_text=self._last_assistant_text.get(session.id, ""),
                audio_position_ms=0,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_BARGE_IN,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_BARGE_IN hook")

    async def _handle_speech_event(self, session: VoiceSession, event_type: str) -> None:
        """Fire speech hooks and publish ephemeral indicator."""
        if not self._framework:
            return

        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return

        from roomkit.telemetry.context import reset_span

        _, _tok = self._rt_span_ctx(session.id)
        try:
            context = await self._framework._build_context(room_id)
            trigger = (
                HookTrigger.ON_SPEECH_START if event_type == "start" else HookTrigger.ON_SPEECH_END
            )

            await self._framework.hook_engine.run_async_hooks(
                room_id,
                trigger,
                session,
                context,
                skip_event_filter=True,
            )

            await self._send_client_message(
                session,
                {
                    "type": "speaking",
                    "speaking": event_type == "start",
                    "who": "user",
                },
            )

            if event_type == "start":
                await self._send_client_message(
                    session,
                    {"type": "clear_audio"},
                )

        except Exception:
            logger.exception("Error handling speech %s for session %s", event_type, session.id)
        finally:
            if _tok is not None:
                reset_span(_tok)

    # -----------------------------------------------------------------
    # Audio level hooks
    # -----------------------------------------------------------------

    def _fire_audio_level_task(
        self, session: VoiceSession, level_db: float, trigger: HookTrigger
    ) -> None:
        """Schedule a task to fire an audio level hook, throttled to ~10/sec.

        Works from both the event-loop thread and foreign threads (e.g.
        PortAudio speaker callback).
        """
        now = time.monotonic()
        if trigger == HookTrigger.ON_INPUT_AUDIO_LEVEL:
            if now - self._last_input_level_at < 0.1:
                return
            self._last_input_level_at = now
        else:
            if now - self._last_output_level_at < 0.1:
                return
            self._last_output_level_at = now
        if not self._framework:
            return
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            cached = self._event_loop
            if cached is not None and cached.is_running():
                cached.call_soon_threadsafe(
                    self._create_audio_level_task, session, level_db, room_id, trigger
                )
            return
        self._event_loop = loop
        self._create_audio_level_task(session, level_db, room_id, trigger)

    def _create_audio_level_task(
        self,
        session: VoiceSession,
        level_db: float,
        room_id: str,
        trigger: HookTrigger,
    ) -> None:
        """Create the audio level hook task (must be called on the event loop thread)."""
        self._track_task(
            asyncio.get_running_loop(),
            self._fire_audio_level(session, level_db, room_id, trigger),
            name=f"rt_audio_level:{session.id}",
        )

    async def _fire_audio_level(
        self,
        session: VoiceSession,
        level_db: float,
        room_id: str,
        trigger: HookTrigger,
    ) -> None:
        """Fire an ON_INPUT_AUDIO_LEVEL or ON_OUTPUT_AUDIO_LEVEL hook."""
        if not self._framework:
            return
        from roomkit.telemetry.context import reset_span

        _, _tok = self._rt_span_ctx(session.id)
        try:
            from roomkit.voice.events import AudioLevelEvent

            context = await self._framework._build_context(room_id)
            event = AudioLevelEvent(session=session, level_db=level_db)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                trigger,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing %s hook", trigger)
        finally:
            if _tok is not None:
                reset_span(_tok)
