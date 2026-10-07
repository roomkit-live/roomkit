"""VoiceChannel mixin — STT streaming and speech processing."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._voice_speakers import (
    PipelineSpeakerTally,
    SpeakerAttribution,
    SpeakerTracker,
    claimed_speaker,
)
from roomkit.core.task_utils import cancel_and_wait
from roomkit.models.enums import HookTrigger
from roomkit.telemetry.base import Attr, SpanKind, TelemetryProvider
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.telemetry.redaction import redact
from roomkit.voice.base import AudioChunk, SpeakerSegment, TranscriptionResult
from roomkit.voice.events import SpeakerChangeEvent, TranscriptionEvent
from roomkit.voice.interruption import InterruptionHandler
from roomkit.voice.stt.base import diarizes

_NOOP = NoopTelemetryProvider()

if TYPE_CHECKING:
    import threading
    from concurrent.futures import Future

    from roomkit.channels._voice_turn import DueTranscript
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.pipeline.turn.base import TurnEntry
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.stt.language import STTLanguageLock
    from roomkit.voice.tts.context import TTSContextStore

    from .voice import TTSPlaybackState, _STTStreamState

logger = logging.getLogger("roomkit.voice")


def _transcription_event(
    session: VoiceSession, text: str, language: str | None, speaker: SpeakerAttribution | None
) -> TranscriptionEvent:
    """The ON_TRANSCRIPTION event, with who said ``text`` when it is known."""
    return TranscriptionEvent(
        session=session,
        text=text,
        language=language,
        speaker=speaker.label if speaker else None,
        speaker_epoch=speaker.epoch if speaker else None,
        sender_name=speaker.sender_name if speaker else None,
    )


def _with_hook_sender_name(speaker: SpeakerAttribution, hook_event: Any) -> SpeakerAttribution:
    """The attribution with the name an ON_TRANSCRIPTION hook gave, if it gave one."""
    if isinstance(hook_event, TranscriptionEvent) and hook_event.sender_name:
        return dataclasses.replace(speaker, sender_name=hook_event.sender_name)
    return speaker


_CARRY_OVER_MAX_S = 5.0
"""Audio carried from one continuous STT stream into the next at most (RFC
§12.2): a reconnect that waits on a silent service gathers a backlog the next
stream's service may refuse (Meta: 7 s at once), and speech older than this is
past answering anyway."""


def _duration(chunk: AudioChunk) -> float:
    """Seconds of 16-bit PCM in ``chunk``."""
    return len(chunk.data) / (2 * max(chunk.channels, 1) * chunk.sample_rate)


def _carry_over(old: asyncio.Queue[Any], new: asyncio.Queue[Any]) -> float:
    """Move what *old* holds into *new*, keeping the most recent
    ``_CARRY_OVER_MAX_S`` of audio; the seconds dropped."""
    held: list[Any] = []
    while not old.empty():
        held.append(old.get_nowait())
    excess = sum(_duration(item) for item in held if item is not None) - _CARRY_OVER_MAX_S
    dropped = 0.0
    # The tolerance keeps a float sum of chunk durations from dropping one too many.
    while held and dropped < excess - 1e-6:
        item = held.pop(0)
        if item is not None:
            dropped += _duration(item)
    for item in held:
        new.put_nowait(item)
    return dropped


def _segments_of(result: TranscriptionResult) -> list[SpeakerSegment]:
    """A diarizing STT's final as segments; text it gave none for is nobody's.

    The provider contract says every final carries its segments (RFC §12.2.3);
    a provider that breaks it loses no words, they reach the room unattributed.
    """
    return result.segments or [SpeakerSegment(None, result.text)]


def _extract_transcription_text(hook_event: Any, original: str) -> str:
    """Extract text from an ON_TRANSCRIPTION hook result.

    Handles both ``TranscriptionEvent`` (preferred) and plain ``str``
    (backward compat) returns from hook handlers.
    """
    if isinstance(hook_event, TranscriptionEvent):
        return hook_event.text
    if isinstance(hook_event, str):
        return hook_event
    return original


async def _await_stream_end(task: asyncio.Task[Any], session_id: str) -> None:
    """Give an STT stream up to 5 s to end, then cancel it.

    ``asyncio.wait``, not ``wait_for``: the stream task ends quietly when it
    is cancelled, so under ``wait_for`` a cancellation of the caller read as
    a normal end and the turn went on.
    """
    try:
        _, pending = await asyncio.wait({task}, timeout=5.0)
    except asyncio.CancelledError:
        # The stream does not outlive its caller, and ends before it moves on
        await cancel_and_wait(task, log_errors_to=logger)
        raise
    if pending:
        logger.warning("STT stream timeout for %s, cancelling it", session_id)
    await cancel_and_wait(task)


@runtime_checkable
class STTHost(Protocol):
    """Contract: capabilities a host class must provide for VoiceSTTMixin.

    Every attribute listed here is initialized by ``VoiceChannel.__init__``.

    Attributes:
        channel_id: Unique identifier for this channel instance.
        _framework: Reference to the owning RoomKit framework (None before registration).
        _stt: Speech-to-text provider (None when not configured).
        _backend: The voice backend transport (None before configuration).
        _pipeline: The audio processing pipeline (None when no pipeline).
        _pipeline_config: Audio pipeline configuration (None when no pipeline).
        _session_bindings: Map of session ID to (room_id, binding) pairs.
        _playing_sessions: Active TTS playback states per session.
        _playback_done_events: Signals that send_audio() has returned for a session.
        _last_tts_ended_at: Monotonic timestamp of last TTS completion per session.
        _stt_streams: Active STT stream states per session.
        _stt_languages: STT language chosen per session (absent = provider default).
        _stt_language_lock: Policy choosing the session language, or None.
        _tts_context: Per-session dialogue for a context-aware TTS (None when unused).
        _continuous_stt: Whether continuous STT mode is enabled.
        _batch_mode: Whether batch STT mode is enabled.
        _batch_audio_buffers: Accumulated audio per session for batch transcription.
        _batch_audio_sample_rate: Sample rate of batch audio per session.
        _pending_turns: Accumulated turn entries per session, awaiting turn completion.
        _pending_audio: Accumulated raw audio per session for audio-native turn detectors.
        _debug_frame_count: Counter for RMS debug logging.
        _barge_in_energy_count: Consecutive high-energy frame count per session.
        _scheduled_tasks: Set of background tasks for lifecycle management.
        _state_lock: Threading lock protecting shared mutable state.
        _interruption_handler: Strategy for handling barge-in interruptions.
    """

    channel_id: str
    _framework: RoomKit | None
    _stt: STTProvider | None
    _backend: VoiceBackend | None
    _pipeline: AudioPipeline | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _playing_sessions: dict[str, TTSPlaybackState]
    _playback_done_events: dict[str, asyncio.Event]
    _last_tts_ended_at: dict[str, float]
    _stt_streams: dict[str, _STTStreamState]
    _stt_languages: dict[str, str]
    _stt_language_lock: STTLanguageLock | None
    _tts_context: TTSContextStore | None
    _continuous_stt: bool
    _batch_mode: bool
    _batch_audio_buffers: dict[str, bytearray]
    _batch_audio_sample_rate: dict[str, int]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_audio: dict[str, bytearray]
    _debug_frame_count: int
    _barge_in_energy_count: dict[str, int]
    _speech_started_at: dict[str, float]
    _burst_words: dict[str, str]
    _burst_backchannel: dict[str, str]
    _speaker_trackers: dict[str, SpeakerTracker]
    _transcript_locks: dict[str, asyncio.Lock]
    _pipeline_speaker_tally: PipelineSpeakerTally | None
    _scheduled_tasks: set[asyncio.Task[Any]]
    _state_lock: threading.Lock
    _interruption_handler: Any  # InterruptionHandler


class VoiceSTTMixin:
    """STT streaming and speech-processing helpers for VoiceChannel.

    Host contract: :class:`STTHost`.
    """

    # -- attributes provided by VoiceChannel.__init__ (see STTHost) --
    channel_id: str
    _framework: RoomKit | None
    _stt: STTProvider | None
    _backend: VoiceBackend | None
    _pipeline: AudioPipeline | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _playing_sessions: dict[str, TTSPlaybackState]
    _playback_done_events: dict[str, asyncio.Event]
    _last_tts_ended_at: dict[str, float]
    _stt_streams: dict[str, _STTStreamState]
    _stt_languages: dict[str, str]
    _stt_language_lock: STTLanguageLock | None
    _tts_context: TTSContextStore | None
    _continuous_stt: bool
    _batch_mode: bool
    _batch_audio_buffers: dict[str, bytearray]
    _batch_audio_sample_rate: dict[str, int]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_audio: dict[str, bytearray]
    _debug_frame_count: int
    _barge_in_energy_count: dict[str, int]
    _speech_started_at: dict[str, float]
    _burst_words: dict[str, str]
    _burst_backchannel: dict[str, str]
    _speaker_trackers: dict[str, SpeakerTracker]
    _transcript_locks: dict[str, asyncio.Lock]
    _pipeline_speaker_tally: PipelineSpeakerTally | None
    _scheduled_tasks: set[asyncio.Task[Any]]
    _state_lock: Any  # threading.Lock — see STTHost
    _interruption_handler: Any  # InterruptionHandler — see STTHost

    # -- cross-mixin methods (annotated as Any to avoid MRO shadowing) --
    _schedule: Any  # see STTHost — VoiceChannel._schedule
    _fire_partial_transcription_hook: Any  # see STTHost — VoiceHooksMixin
    _handle_barge_in: Any  # see STTHost — VoiceChannel._handle_barge_in
    _evaluate_turn: Any  # see STTHost — VoiceTurnMixin._evaluate_turn
    _route_text: Any  # see STTHost — VoiceTurnMixin._route_text
    _supersede_unheard_turn: Any  # VoiceTurnMixin._supersede_unheard_turn
    _release_unheard_turn: Any  # VoiceTurnMixin._release_unheard_turn
    _expect_transcript: Any  # VoiceTurnMixin._expect_transcript
    _fire_speech_start_hooks: Any  # see STTHost — VoiceHooksMixin
    _resolve_session_backend: Any  # see STTHost — VoiceChannel._resolve_session_backend
    _broadcast_bridge_transcription: Any  # see STTHost — VoiceChannel
    _task_done: Any  # see STTHost — VoiceChannel._task_done
    _pipeline_audio_rate: Any  # see STTHost — VoicePipelineMixin
    _on_held_transcript: Any  # see STTHost — VoiceChannel._on_held_transcript
    _is_held_for_transcript: Any  # see STTHost — VoiceChannel
    _fire_backchannel_hook: Any  # see STTHost — VoiceHooksMixin
    _fire_speaker_change_event: Any  # VoiceHooksMixin

    # -----------------------------------------------------------------
    # Per-session STT language
    # -----------------------------------------------------------------

    def _stt_call_kwargs(self, session_id: str) -> dict[str, Any]:
        """``language=`` for a provider call — only when one is chosen for the
        session and the provider honours it; otherwise the call is unchanged."""
        # getattr: a duck-typed provider on the older contract has no flag
        if self._stt is None or not getattr(self._stt, "supports_language_override", False):
            return {}
        with self._state_lock:
            language = self._stt_languages.get(session_id)
        return {"language": language} if language else {}

    def _store_stt_language(self, session_id: str, language: str | None) -> bool:
        """Record the session's language; return whether it changed."""
        with self._state_lock:
            current = self._stt_languages.get(session_id)
            if language is None:
                self._stt_languages.pop(session_id, None)
            else:
                self._stt_languages[session_id] = language
        if current == language:
            return False
        logger.info(
            "STT language for session %s: %s -> %s",
            session_id,
            current or "default",
            language or "default",
        )
        return True

    def _restart_continuous_stt_cycle(self, session_id: str) -> None:
        """End the current continuous cycle so the loop reconnects.

        The sentinel goes to the queue the loop is reading right now; the
        next cycle opens with whatever language is set by then, and the
        frames buffered in between are carried over as on every reconnect.
        """
        state = self._stt_streams.get(session_id)
        if state is None or state.cancelled:
            return
        with contextlib.suppress(asyncio.QueueFull):
            state.queue.put_nowait(None)

    def _observe_stt_language(
        self, session: VoiceSession, result: TranscriptionResult, *, restart: bool = True
    ) -> None:
        """Feed a final result to the language lock and apply its decision.

        ``restart=False`` is for the continuous loop, which reconnects on its
        own after a final — ending the cycle a second time would only add an
        empty one.
        """
        lock = self._stt_language_lock
        if lock is None:
            return
        target = lock.observe(session.id, result)
        if not self._store_stt_language(session.id, target):
            return
        if restart and self._continuous_stt:
            self._restart_continuous_stt_cycle(session.id)

    # -----------------------------------------------------------------
    # VAD-driven STT streaming
    # -----------------------------------------------------------------

    # Safety cap: ~2MB — prevents unbounded memory growth if flush fails
    _MAX_STT_BUFFER_BYTES = 10 * 6400  # 10x normal buffer (~2MB at 16kHz mono 16-bit)

    def _on_pipeline_speech_frame(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Handle processed audio frame during speech — feed to STT stream.

        Frames are buffered (~200ms) before being sent to the queue to
        avoid per-frame resampling artifacts in providers that resample
        (e.g. 16kHz -> 24kHz).
        """
        from .voice import _STT_STREAM_BUFFER_BYTES

        stream_state = self._stt_streams.get(session.id)
        if stream_state is None or stream_state.cancelled:
            return
        stream_state.frame_buffer.extend(frame.data)
        stream_state.frame_buffer_rate = frame.sample_rate
        if len(stream_state.frame_buffer) >= _STT_STREAM_BUFFER_BYTES:
            self._flush_stt_buffer(stream_state, session.id)
        # Safety cap: if buffer exceeds max, force flush + warn
        if len(stream_state.frame_buffer) >= self._MAX_STT_BUFFER_BYTES:
            logger.warning("STT buffer overflow for session %s, forcing flush", session.id)
            self._flush_stt_buffer(stream_state, session.id)

    def _flush_stt_buffer(self, state: _STTStreamState, session_id: str) -> None:
        """Flush buffered audio frames to the STT stream queue."""
        if not state.frame_buffer:
            return
        from roomkit.voice.base import AudioChunk as OutChunk

        chunk = OutChunk(
            data=bytes(state.frame_buffer),
            sample_rate=state.frame_buffer_rate,
        )
        state.frame_buffer.clear()
        try:
            state.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # Drop the oldest chunk to make room for fresher audio
            import contextlib

            with contextlib.suppress(asyncio.QueueEmpty):
                state.queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                state.queue.put_nowait(chunk)
            logger.warning("STT queue full for session %s, dropped oldest chunk", session_id)

    def _start_stt_stream(
        self,
        session: VoiceSession,
        room_id: str,
        pre_roll: bytes | None = None,
    ) -> None:
        """Start a streaming STT session.

        Args:
            pre_roll: Pre-speech audio from the VAD buffer.  Sent as the
                first chunk so the STT provider receives the full utterance
                including audio captured before SPEECH_START fired.
        """
        from .voice import _STTStreamState

        # Cancel any existing stream for this session
        self._cancel_stt_stream(session.id)
        logger.debug("Starting STT stream for session %s", session.id)

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=500)

        # Seed the queue with pre-roll audio so the first word isn't lost
        if pre_roll:
            from roomkit.voice.base import AudioChunk as OutChunk

            sample_rate = self._pipeline_audio_rate(session)
            logger.debug(
                "STT stream pre-roll: %d bytes, sample_rate=%d",
                len(pre_roll),
                sample_rate,
            )
            queue.put_nowait(OutChunk(data=pre_roll, sample_rate=sample_rate))

        async def audio_gen() -> AsyncIterator[AudioChunk]:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                yield chunk

        async def consume(state: _STTStreamState) -> None:
            try:
                if self._stt is None:
                    raise RuntimeError("STT provider not configured")
                async for result in self._stt.transcribe_stream(
                    audio_gen(), **self._stt_call_kwargs(session.id)
                ):
                    if state.cancelled:
                        return
                    if result.is_final and result.text:
                        state.final_text = result.text
                        state.final_result = result
                    elif not result.is_final and result.text:
                        state.partial_text = result.text
                        state.partial_result = result
                        # A held segment is not a turn yet: its words reach
                        # no hook unless they cut in.
                        if not self._is_held_for_transcript(session.id):
                            self._schedule(
                                self._fire_partial_transcription_hook(session, result, room_id),
                                name=f"partial_stt:{session.id}",
                            )
                    if result.text:
                        # A segment SEMANTIC holds during playback is judged
                        # on its words as they come (RFC §12.3.13).
                        self._on_held_transcript(session, room_id, result.text)
            except asyncio.CancelledError:
                state.cancelled = True
            except Exception:
                logger.exception("STT stream error for session %s", session.id)
                state.error = True

        state = _STTStreamState(queue=queue)
        self._stt_streams[session.id] = state
        try:
            loop = asyncio.get_running_loop()
            state.task = loop.create_task(consume(state), name=f"stt_stream:{session.id}")
            state.task.add_done_callback(self._task_done)
            self._scheduled_tasks.add(state.task)
        except RuntimeError:
            self._stt_streams.pop(session.id, None)

    def _end_stt_stream(self, session_id: str) -> _STTStreamState | None:
        """End a segment's STT stream: its last audio, then the end of input.

        The state leaves the table now, so a SPEECH_START right after starts a
        stream of its own; the caller collects the final transcript from it.
        """
        state = self._stt_streams.pop(session_id, None)
        if state is not None:
            self._flush_stt_buffer(state, session_id)
            try:
                state.queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning("STT stream queue full on sentinel for %s", session_id)
        return state

    def _cancel_stt_stream(self, session_id: str) -> None:
        """Cancel an active streaming STT session."""
        state = self._stt_streams.pop(session_id, None)
        if state is None:
            return
        state.cancelled = True
        import contextlib

        # Send sentinel to the current queue.  In continuous mode the queue
        # may be replaced on each reconnect cycle, so we must signal the
        # queue that ``run_continuous`` is actually reading from *right now*.
        with contextlib.suppress(asyncio.QueueFull):
            state.queue.put_nowait(None)
        if state.task is not None:
            state.task.cancel()

    # -----------------------------------------------------------------
    # Continuous STT (no local VAD — provider handles endpointing)
    # -----------------------------------------------------------------

    # Post-denoiser barge-in: detect user speaking during TTS playback.
    # After AEC + denoiser the audio contains only the user's voice —
    # echo and noise are stripped.  A simple absolute RMS threshold
    # reliably detects speech without false triggers from playback.
    _BARGE_IN_RMS_THRESHOLD = 200.0  # absolute RMS floor (post-denoiser)
    _BARGE_IN_FRAMES_REQUIRED = 5  # ~100ms at 20ms frames

    def _check_energy_barge_in(
        self,
        session: VoiceSession,
        frame: AudioFrame,
        playback: TTSPlaybackState,
    ) -> None:
        """Detect barge-in from post-denoiser audio during TTS playback.

        After AEC + denoiser the signal only contains the user's voice —
        echo and noise are stripped.  A simple absolute RMS threshold
        reliably detects speech without false triggers from playback.
        """
        import struct

        # A barge-in already owns this playback: nothing is left to decide.
        if playback.barge_in_claimed:
            return
        n_samples = len(frame.data) // 2
        if n_samples == 0:
            return

        samples = struct.unpack(f"<{n_samples}h", frame.data)
        rms = (sum(s * s for s in samples) / n_samples) ** 0.5

        with self._state_lock:
            if rms > self._BARGE_IN_RMS_THRESHOLD:
                self._barge_in_energy_count[session.id] = (
                    self._barge_in_energy_count.get(session.id, 0) + 1
                )
                # Onset of this run of above-threshold frames: what a
                # duration-based strategy measures against (RFC §12.6).
                self._speech_started_at.setdefault(session.id, time.monotonic())
            else:
                self._barge_in_energy_count[session.id] = 0
                self._speech_started_at.pop(session.id, None)

            triggered = (
                self._barge_in_energy_count.get(session.id, 0) >= self._BARGE_IN_FRAMES_REQUIRED
            )
            started_at = self._speech_started_at.get(session.id)
            binding_info = self._session_bindings.get(session.id)

        if not triggered:
            return
        if binding_info:
            room_id, _ = binding_info
            # Consult the interruption handler before firing barge-in
            # (respects InterruptionStrategy.DISABLED and other policies).
            # The energy run is itself the sustained-speech measurement, so
            # CONFIRMED gets a real duration here instead of a constant 0 —
            # and the counter is NOT reset while a strategy is still waiting
            # for the speech to sustain, or it could never reach its minimum.
            handler: InterruptionHandler = self._interruption_handler
            speech_duration_ms = (
                int((time.monotonic() - started_at) * 1000) if started_at is not None else 0
            )
            # The words the STT heard in this burst decide under SEMANTIC; an
            # acknowledgement is not cut for running long (RFC §12.3.13).
            with self._state_lock:
                words = self._burst_words.get(session.id, "")
                acknowledged = bool(words) and self._burst_backchannel.get(session.id) == words
            if acknowledged:
                return  # these words were judged already: nothing new to decide
            decision = handler.evaluate(
                playback_position_ms=playback.played_ms,
                speech_duration_ms=speech_duration_ms,
                speech_text=words,
                transcript_expected=True,
            )
            if decision.is_backchannel:
                self._report_burst_backchannel(session, words, room_id)
            if not decision.should_interrupt:
                return
            with self._state_lock:
                self._barge_in_energy_count[session.id] = 0
            logger.info(
                "Energy barge-in (post-denoiser): rms=%.0f, threshold=%.0f, pos=%dms",
                rms,
                self._BARGE_IN_RMS_THRESHOLD,
                playback.played_ms,
            )
            self._schedule(
                self._handle_barge_in(session, playback, room_id),
                name=f"energy_barge_in:{session.id}",
            )

    def _note_burst_words(self, session_id: str, text: str) -> None:
        """Keep the latest words of the utterance under way (continuous mode).

        They belong to the STT's utterance, not to a run of loud frames: the
        pauses between words end an energy run, never the utterance. They are
        forgotten at its final result and when the session is unbound.
        """
        with self._state_lock:
            self._burst_words[session_id] = text

    def _forget_burst_words(self, session_id: str) -> None:
        with self._state_lock:
            self._burst_words.pop(session_id, None)
            self._burst_backchannel.pop(session_id, None)

    def _report_burst_backchannel(self, session: VoiceSession, text: str, room_id: str) -> None:
        """Remember the words judged a backchannel; fire ON_BACKCHANNEL once per burst."""
        with self._state_lock:
            first = session.id not in self._burst_backchannel
            self._burst_backchannel[session.id] = text
        if not first:
            return
        self._schedule(
            self._fire_backchannel_hook(session, text, room_id),
            name=f"backchannel:{session.id}",
        )

    def _on_processed_frame_for_stt(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Feed every processed frame to the continuous STT stream.

        In continuous mode the STT provider handles endpointing
        server-side (e.g. Gradium ``end_text`` events), so we just
        buffer and forward audio without any local silence detection.

        Also performs energy-based barge-in on post-denoiser audio
        during TTS playback via :meth:`_check_energy_barge_in`.
        """
        from .voice import _STT_STREAM_BUFFER_BYTES

        # Energy barge-in on post-denoiser audio during TTS playback
        with self._state_lock:
            playback = self._playing_sessions.get(session.id)
        if playback:
            # Skip energy barge-in during drain period (send_audio returned,
            # waiting for echo decay) — nothing is actually playing.
            done_ev = self._playback_done_events.get(session.id)
            if done_ev is None or not done_ev.is_set():
                self._check_energy_barge_in(session, frame, playback)
        elif self._barge_in_energy_count.get(session.id, 0) > 0:
            with self._state_lock:
                self._barge_in_energy_count[session.id] = 0

        stream_state = self._stt_streams.get(session.id)
        if stream_state is None or stream_state.cancelled:
            return

        stream_state.frame_buffer.extend(frame.data)
        stream_state.frame_buffer_rate = frame.sample_rate
        if len(stream_state.frame_buffer) >= _STT_STREAM_BUFFER_BYTES:
            self._flush_stt_buffer(stream_state, session.id)

    async def _kept_stream_audio(
        self, state: _STTStreamState, queue: asyncio.Queue[Any], first_chunk: AudioChunk
    ) -> AsyncIterator[AudioChunk]:
        """The audio of a kept (diarizing) stream, never behind the clock.

        A provider closes a stream whose audio falls behind real time (Meta
        Muse: 1008 ~15 s after the audio stops, measured 2026-09-27), which
        would start a new label epoch. A muted microphone during playback
        sends nothing, and lost packets leave a stream short without any
        pause, over a whole call. So whenever the queue stays empty for a
        pacing period, the time the stream is behind is filled: the audio
        buffered so far, then silence. Only a pause is filled, never the
        middle of speech. The stream ends on the queue's sentinel or when the
        session's STT is cancelled.
        """
        from .voice import _KEPT_STREAM_PACE_S

        opened = time.monotonic()
        sent = _duration(first_chunk)
        yield first_chunk
        while not state.cancelled:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=_KEPT_STREAM_PACE_S)
            except TimeoutError:
                behind = time.monotonic() - opened - sent
                if behind >= _KEPT_STREAM_PACE_S:
                    for pad in self._padding(state, behind):
                        sent += _duration(pad)
                        yield pad
                continue
            if chunk is None:
                return
            sent += _duration(chunk)
            yield chunk

    def _padding(self, state: _STTStreamState, seconds: float) -> list[AudioChunk]:
        """``seconds`` of audio: what is buffered, then silence for the rest."""
        rate = state.frame_buffer_rate
        buffered = bytes(state.frame_buffer)
        state.frame_buffer.clear()
        silence = b"\x00" * max(int(rate * seconds) * 2 - len(buffered), 0)
        return [AudioChunk(data=data, sample_rate=rate) for data in (buffered, silence) if data]

    def _start_continuous_stt(self, session: VoiceSession) -> None:
        """Start continuous STT streaming for a session.

        Opens a long-lived stream to the STT provider and relies on the
        provider's server-side VAD / endpointing (e.g. Gradium ``end_text``
        events).  Individual segments are accumulated; when the provider
        yields ``is_final=True`` the accumulated text is treated as a
        complete utterance and routed to the AI.

        The stream auto-restarts on provider errors or session-duration
        limits (Gradium allows up to 300 s per stream).
        """
        from .voice import _STTStreamState

        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._stt:
            return
        room_id, _ = binding_info
        logger.info("Starting continuous STT for session %s", session.id)

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=500)
        state = _STTStreamState(queue=queue)
        self._stt_streams[session.id] = state

        # A diarizing STT's labels hold within one stream, so its stream is kept
        # across turns and through silence (RFC §12.2.3); other providers get
        # a fresh stream per turn.
        keep_stream = diarizes(self._stt)

        async def run_continuous(state: _STTStreamState) -> None:
            import time as _time

            backoff = 0.1
            while not state.cancelled:
                # Fresh queue + WebSocket per turn (avoids server-side overlap).
                # Keep frame_buffer intact — audio arriving during the
                # reconnection gap (sleep + WebSocket connect) is preserved
                # and will be flushed to the new queue on the next frame.
                old_queue = state.queue
                state.queue = asyncio.Queue[Any](maxsize=500)
                dropped = _carry_over(old_queue, state.queue)
                if dropped:
                    logger.warning(
                        "STT reconnect for %s: dropped %.1fs of buffered audio, "
                        "kept the last %.0fs",
                        session.id,
                        dropped,
                        _CARRY_OVER_MAX_S,
                    )
                cur_queue = state.queue

                async def audio_gen(
                    first_chunk: AudioChunk,
                    q: asyncio.Queue[Any] = cur_queue,
                ) -> AsyncIterator[AudioChunk]:
                    from .voice import _STT_INACTIVITY_TIMEOUT_S

                    # Yield the pre-fetched first chunk immediately so the
                    # STT provider gets audio right after connecting.
                    yield first_chunk
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                q.get(), timeout=_STT_INACTIVITY_TIMEOUT_S
                            )
                        except TimeoutError:
                            # No audio for timeout period — flush remaining
                            # buffer and close the stream so the provider
                            # can yield accumulated text as final.
                            if state.frame_buffer:
                                from roomkit.voice.base import AudioChunk as OutChunk

                                yield OutChunk(
                                    data=bytes(state.frame_buffer),
                                    sample_rate=state.frame_buffer_rate,
                                )
                                state.frame_buffer.clear()
                            logger.info(
                                "Audio inactivity timeout (%.1fs) for %s, closing STT stream",
                                _STT_INACTIVITY_TIMEOUT_S,
                                session.id,
                            )
                            return
                        if chunk is None:
                            return
                        yield chunk

                try:
                    if self._stt is None:
                        raise RuntimeError("STT provider not configured")
                    barge_in_fired = False
                    backoff = 0.1

                    # Wait for the first audio chunk before connecting the
                    # STT stream.  This avoids opening a WebSocket that sits
                    # idle — some providers lose the first audio after a
                    # long idle period.
                    first_chunk = await cur_queue.get()
                    if first_chunk is None or state.cancelled:
                        continue

                    last_tts = self._last_tts_ended_at.get(session.id, 0.0)
                    since_tts = _time.monotonic() - last_tts if last_tts else -1.0
                    logger.info(
                        "Continuous STT stream cycle starting for %s (since_tts=%.1fs)",
                        session.id,
                        since_tts,
                    )
                    epoch = state.streams_opened
                    state.streams_opened += 1
                    barge_in_playback: Any = None
                    stream_audio = (
                        self._kept_stream_audio(state, cur_queue, first_chunk)
                        if keep_stream
                        else audio_gen(first_chunk)
                    )
                    async for result in self._stt.transcribe_stream(
                        stream_audio, **self._stt_call_kwargs(session.id)
                    ):
                        if state.cancelled:
                            break
                        with self._state_lock:
                            playing = session.id in self._playing_sessions
                            current_playback = self._playing_sessions.get(session.id)
                        if keep_stream and current_playback is not barge_in_playback:
                            # A kept stream never starts over: a barge-in holds
                            # only for the playback it cut, even when no final
                            # with words followed it (a cough).
                            barge_in_fired = False

                        # Provider signals speech detected (server-side VAD)
                        if result.is_speech_start:
                            self._schedule(
                                self._fire_speech_start_hooks(session, room_id),
                                name=f"speech_start:{session.id}",
                            )

                        if result.is_final and result.text:
                            # The utterance is over: its words and verdict
                            # say nothing about the next one.
                            self._forget_burst_words(session.id)
                            last_tts = self._last_tts_ended_at.get(session.id, 0.0)
                            since_tts_now = _time.monotonic() - last_tts if last_tts else -1.0
                            logger.info(
                                "STT final: %r (playing=%s, barge_in=%s, since_tts=%.1fs)",
                                result.text,
                                playing,
                                barge_in_fired,
                                since_tts_now,
                            )
                            # During playback (and no barge-in), this is
                            # almost certainly TTS echo leaking through AEC.
                            # Discard it and reconnect for a fresh stream.
                            with self._state_lock:
                                playback = self._playing_sessions.get(session.id)
                            if playback and not barge_in_fired:
                                logger.info(
                                    "Discarding echo transcription during playback: %r",
                                    result.text,
                                )
                                if keep_stream:
                                    continue  # the stream, and its labels, stay
                                with contextlib.suppress(asyncio.QueueFull):
                                    cur_queue.put_nowait(None)
                                break

                            # The words that cut the playback: the person's
                            # turn, whatever playback the cut has not yet
                            # removed when this final is handled (RFC §12.3.13).
                            cut_by_it = barge_in_fired
                            barge_in_fired = False
                            if not keep_stream:
                                # Signal audio gen to stop so the SDK's
                                # sender task unblocks and the stream
                                # closes cleanly (no timeout needed).
                                with contextlib.suppress(asyncio.QueueFull):
                                    cur_queue.put_nowait(None)
                            # The lock sees the final before the next cycle
                            # opens, so a new language lands on that cycle. A
                            # kept stream is ended for it (a language holds
                            # for the life of a stream).
                            self._observe_stt_language(session, result, restart=keep_stream)
                            # The pipeline stage's speaker, taken as the final
                            # lands; an STT label outranks it.
                            stage_speaker = (
                                None if keep_stream else self._pipeline_speaker(session.id)
                            )
                            # Provider signals turn complete — route to AI
                            self._schedule(
                                self._handle_continuous_transcription(
                                    session,
                                    result.text,
                                    room_id,
                                    language=result.language,
                                    segments=_segments_of(result) if keep_stream else None,
                                    epoch=epoch,
                                    speaker=stage_speaker,
                                    cut_playback=cut_by_it,
                                ),
                                name=f"continuous_stt:{session.id}",
                            )
                        elif not result.is_final and result.text:
                            # Barge-in: user speaking during TTS playback
                            if not barge_in_fired:
                                with self._state_lock:
                                    playback = self._playing_sessions.get(session.id)
                                # Skip barge-in during drain period (send_audio
                                # returned, waiting for echo decay).
                                drain_ev = self._playback_done_events.get(session.id)
                                if playback and not (drain_ev is not None and drain_ev.is_set()):
                                    handler: InterruptionHandler = self._interruption_handler
                                    # The partial is what SEMANTIC classifies —
                                    # a detector given "" cannot tell "uh-huh"
                                    # from "wait, stop".
                                    self._note_burst_words(session.id, result.text)
                                    decision = handler.evaluate(
                                        playback_position_ms=playback.played_ms,
                                        speech_duration_ms=0,
                                        speech_text=result.text,
                                    )
                                    if decision.is_backchannel:
                                        self._report_burst_backchannel(
                                            session, result.text, room_id
                                        )
                                    logger.info(
                                        "Barge-in eval: partial=%r pos=%dms interrupt=%s",
                                        result.text,
                                        playback.played_ms,
                                        decision.should_interrupt,
                                    )
                                    if decision.should_interrupt:
                                        barge_in_fired = True
                                        barge_in_playback = playback
                                        self._schedule(
                                            self._handle_barge_in(session, playback, room_id),
                                            name=f"barge_in:{session.id}",
                                        )
                            self._schedule(
                                self._fire_partial_transcription_hook(session, result, room_id),
                                name=f"partial_stt:{session.id}",
                            )
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception(
                        "Continuous STT stream error for session %s",
                        session.id,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

                if not state.cancelled:
                    last_tts = self._last_tts_ended_at.get(session.id, 0.0)
                    since_tts_end = _time.monotonic() - last_tts if last_tts else -1.0
                    logger.info(
                        "STT stream cycle ended for %s, reconnecting (since_tts=%.1fs)",
                        session.id,
                        since_tts_end,
                    )
                    await asyncio.sleep(0.1)

        try:
            loop = asyncio.get_running_loop()
            state.task = loop.create_task(
                run_continuous(state), name=f"continuous_stt:{session.id}"
            )
            state.task.add_done_callback(self._task_done)
            self._scheduled_tasks.add(state.task)
        except RuntimeError:
            self._stt_streams.pop(session.id, None)

    def _stop_continuous_stt(self, session_id: str) -> None:
        """Stop continuous STT for a session."""
        self._cancel_stt_stream(session_id)

    def _record_user_turn(
        self,
        session: VoiceSession,
        transcript: str,
        final_text: str,
        *,
        audio: bytes | None = None,
        sample_rate: int = 16000,
        dtmf_seen: bool = False,
    ) -> None:
        """Add the utterance to the session's TTS context (RFC §12.2.2).

        Runs after ON_TRANSCRIPTION, with the text as the hooks left it; a
        changed text means a redaction, and the audio is then not kept.
        """
        if self._tts_context is None or not final_text.strip():
            return
        self._tts_context.add_user_turn(
            session.id,
            session.participant_id,
            final_text,
            audio=audio,
            sample_rate=sample_rate,
            text_changed=final_text != transcript,
            dtmf_seen=dtmf_seen,
        )

    async def _handle_continuous_transcription(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        *,
        language: str | None = None,
        segments: list[SpeakerSegment] | None = None,
        epoch: int = 0,
        speaker: SpeakerAttribution | None = None,
        cut_playback: bool = False,
    ) -> None:
        """Process a transcription result from continuous STT.

        ``segments`` is set for a diarizing STT: each segment is then its own
        transcript, with its speaker, in its own room message (RFC §12.2.3);
        ``epoch`` numbers the stream the labels come from. ``cut_playback``: the
        words of this final cut the playback, so they are the person's even while
        the cut, a task of its own, has not removed the playback yet.

        A diarized session's finals go through one at a time, in the order they
        were scheduled: a kept stream can deliver the next speaker's final
        while the previous one is still in its hooks, and the room must see
        them in the order they were said. The lock covers the hooks and the
        commit of each message, never the reply to it.
        """
        if not self._framework or not text.strip():
            return
        if segments is None:
            await self._process_continuous_final(
                session,
                text,
                room_id,
                language=language,
                segments=None,
                epoch=epoch,
                speaker=speaker,
                cut_playback=cut_playback,
            )
            return
        lock = self._transcript_locks.setdefault(session.id, asyncio.Lock())
        async with lock:
            await self._process_continuous_final(
                session,
                text,
                room_id,
                language=language,
                segments=segments,
                epoch=epoch,
                cut_playback=cut_playback,
            )

    async def _process_continuous_final(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        *,
        language: str | None,
        segments: list[SpeakerSegment] | None,
        epoch: int,
        speaker: SpeakerAttribution | None = None,
        cut_playback: bool = False,
    ) -> None:
        """One continuous-mode final, from the echo check to the room.

        ``speaker`` is the pipeline stage's, for a final the STT labelled not. A
        final that cut the playback (``cut_playback``) is never taken for echo.
        """
        if not self._framework:
            return
        _vs_token = None
        try:
            from roomkit.telemetry.context import reset_span, set_current_span

            _vs_parent = getattr(self, "_voice_session_spans", {}).get(session.id)
            _vs_token = set_current_span(_vs_parent) if _vs_parent else None
            import time as _time

            with self._state_lock:
                playback = self._playing_sessions.get(session.id)
            last_tts_end = self._last_tts_ended_at.get(session.id, 0.0)
            since_tts = _time.monotonic() - last_tts_end if last_tts_end else -1.0

            if playback and not cut_playback:
                logger.warning(
                    "Discarding echo during playback: %r (pos=%dms)",
                    text,
                    playback.played_ms,
                )
                return

            logger.info(
                "Transcription: %s (since_tts_end=%.1fs)",
                text,
                since_tts,
            )

            backend = self._resolve_session_backend(session)
            if backend:
                await backend.send_transcription(session, text, "user")
            # Broadcast to other bridged participants
            await self._broadcast_bridge_transcription(session, text, room_id)

            context = await self._framework._build_context(room_id)

            # Fire ON_SPEECH_END hooks (continuous mode synthesises speech events)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SPEECH_END,
                session,
                context,
                skip_event_filter=True,
            )

            if segments is None:
                await self._deliver_transcript(session, text, room_id, context, language, speaker)
                return
            for segment in segments:
                labelled = SpeakerAttribution.of(segment.speaker, epoch)
                await self._deliver_transcript(
                    session, segment.text, room_id, context, language, labelled, ordered=True
                )

        except Exception as exc:
            logger.exception("Error processing continuous STT transcription")
            # Emit stt_error like the VAD twin (_process_speech_end), so event
            # consumers see a continuous-mode routing failure too.
            if self._framework:
                try:
                    await self._framework._emit_framework_event(
                        "stt_error",
                        room_id=room_id,
                        data={
                            "session_id": session.id,
                            "provider": self._stt.name if self._stt else "unknown",
                            "error": str(exc),
                        },
                    )
                except Exception:
                    logger.exception("Error emitting stt_error")
        finally:
            if _vs_token is not None:
                reset_span(_vs_token)

    async def _deliver_transcript(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        context: RoomContext,
        language: str | None,
        speaker: SpeakerAttribution | None,
        *,
        ordered: bool = False,
    ) -> None:
        """ON_TRANSCRIPTION, the TTS context, then the turn detector or the room.

        ``speaker`` is who said ``text``, from a diarizing STT or the pipeline's
        diarization stage: the hook sees the label, its epoch and the name, and
        may rename the speaker (RFC §12.2.3). An STT label that changes speaker
        fires ON_SPEAKER_CHANGE; the pipeline stage fires its own. ``ordered``
        is set under a diarized session's ordering lock: the reply is then
        awaited outside it.
        """
        if not self._framework or not text.strip():
            return
        tx_event = _transcription_event(session, text, language, speaker)
        transcription_result = await self._framework.hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.ON_TRANSCRIPTION,
            tx_event,
            context,
            skip_event_filter=True,
        )
        if not transcription_result.allowed:
            logger.info("Transcription blocked by hook: %s", transcription_result.reason)
            return

        final_text = _extract_transcription_text(transcription_result.event, text)
        if speaker is not None:
            speaker = _with_hook_sender_name(speaker, transcription_result.event)
            if speaker.source == "stt":
                self._note_stt_speaker(session, speaker, room_id)
        # Continuous STT keeps no utterance audio: the turn is text only.
        self._record_user_turn(session, text, final_text)

        await_delivery = not ordered
        turn_detector = self._pipeline_config.turn_detector if self._pipeline_config else None
        if turn_detector is not None:
            await self._evaluate_turn(
                session,
                final_text,
                room_id,
                context,
                speaker=speaker,
                await_delivery=await_delivery,
            )
        else:
            await self._route_text(
                session, final_text, room_id, speaker=speaker, await_delivery=await_delivery
            )

    def _pipeline_speaker(self, session_id: str) -> SpeakerAttribution | None:
        """The pipeline stage's speaker for the transcript at hand, when asked for."""
        tally = self._pipeline_speaker_tally
        return tally.take(session_id) if tally is not None else None

    def _note_stt_speaker(
        self, session: VoiceSession, speaker: SpeakerAttribution, room_id: str
    ) -> None:
        """Fire ON_SPEAKER_CHANGE (source ``stt``) when this segment changes speaker."""
        tracker = self._speaker_trackers.setdefault(session.id, SpeakerTracker())
        is_new = tracker.observe(speaker.label, speaker.epoch)
        if is_new is None or speaker.label is None:
            return
        event = SpeakerChangeEvent(
            session=session,
            speaker_id=speaker.label,
            confidence=None,
            is_new_speaker=is_new,
            source="stt",
            speaker_epoch=speaker.epoch,
        )
        self._schedule(
            self._fire_speaker_change_event(session, event, room_id),
            name=f"speaker_change:{session.id}",
        )

    # -----------------------------------------------------------------
    # Speech-end processing (VAD mode)
    # -----------------------------------------------------------------

    async def _judge_held_speech(
        self,
        session: VoiceSession,
        audio: bytes,
        room_id: str,
        stream_state: _STTStreamState,
        *,
        speech_duration_ms: int,
        speaker_claim: Future[SpeakerAttribution | None] | None = None,
        dtmf_seen: bool = False,
        transcript: DueTranscript | None = None,
    ) -> None:
        """Process a held segment that ended before its first word, if its final words are a turn.

        SEMANTIC held it during playback for words that a streaming STT often
        releases only at the end (RFC §12.3.13). No words, or a backchannel,
        is discarded while the bot talks on; anything else cuts the bot off if
        it still speaks and becomes the user's turn.
        """
        transcript = transcript or self._expect_transcript(session.id)
        try:
            if stream_state.task is not None:
                await _await_stream_end(stream_state.task, session.id)
            words = stream_state.final_text or stream_state.partial_text or ""
            if not await self._held_words_are_a_turn(session, room_id, words, speech_duration_ms):
                logger.debug("Held speech of %s ended without a turn in it: discarded", session.id)
                self._release_unheard_turn(session.id)
                return
            logger.info(
                "Held speech judged on its final words %r (session %s)", redact(words), session.id
            )
            with self._state_lock:
                playback = self._playing_sessions.get(session.id)
            await self._fire_speech_start_hooks(session, room_id)
            if playback is not None:
                await self._handle_barge_in(session, playback, room_id)
            await self._process_speech_end(
                session,
                audio,
                room_id,
                stream_state,
                dtmf_seen=dtmf_seen,
                speaker_claim=speaker_claim,
                transcript=transcript,
            )
        finally:
            transcript.settle()

    async def _held_words_are_a_turn(
        self, session: VoiceSession, room_id: str, words: str, speech_duration_ms: int
    ) -> bool:
        """Whether a held segment's final *words* are the user's turn, a backchannel reported."""
        if not words:
            return False
        with self._state_lock:
            playback = self._playing_sessions.get(session.id)
        if playback is None:
            return True  # the bot finished meanwhile: plain speech is a turn
        decision = self._interruption_handler.evaluate(
            playback_position_ms=playback.played_ms,
            speech_duration_ms=speech_duration_ms,
            speech_text=words,
        )
        if decision.is_backchannel:
            await self._fire_backchannel_hook(session, words, room_id)
        return bool(decision.should_interrupt)

    async def _process_speech_end(
        self,
        session: VoiceSession,
        audio: bytes,
        room_id: str,
        stream_state: _STTStreamState | None = None,
        *,
        dtmf_seen: bool = False,
        speaker_claim: Future[SpeakerAttribution | None] | None = None,
        transcript: DueTranscript | None = None,
    ) -> None:
        """Process speech end: fire hooks, transcribe, route inbound.

        ON_SPEECH_END hooks are fired here (not in _on_pipeline_vad_event)
        to guarantee ordering: ON_SPEECH_END always fires before
        ON_TRANSCRIPTION and before routing to the AI.

        Args:
            stream_state: The STT stream state popped by the caller
                (_on_pipeline_speech_end) so it is immune to a rapid
                SPEECH_START overwriting _stt_streams[session.id].
            speaker_claim: With ``pipeline_speakers``, the utterance's
                speaker as the diarization stage will have heard it.
            transcript: The segment's transcript as the turn waits for it,
                expected when the speech ended; settled here at the latest.
        """
        transcript = transcript or self._expect_transcript(session.id)
        if not self._framework:
            transcript.settle()
            return

        from roomkit.telemetry.context import reset_span, set_current_span

        _vs_parent = getattr(self, "_voice_session_spans", {}).get(session.id)
        _vs_token = set_current_span(_vs_parent) if _vs_parent else None
        # Whether this segment settled the response it held (RFC §12.3.12).
        settled = False
        try:
            context = await self._framework._build_context(room_id)

            # Fire ON_SPEECH_END hooks
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SPEECH_END,
                session,
                context,
                skip_event_filter=True,
            )

            # Transcribe if STT is configured
            if not self._stt:
                logger.warning("Speech ended but no STT provider configured")
                return

            from roomkit.voice.audio_frame import AudioFrame

            # The segment left the pipeline after its inbound resampler.
            sample_rate = self._pipeline_audio_rate(session)

            # Try to collect streaming STT result; fall back to batch
            text: str | None = None
            result: TranscriptionResult | None = None
            if stream_state is not None and not stream_state.error and not stream_state.cancelled:
                try:
                    if stream_state.task is not None:
                        await _await_stream_end(stream_state.task, session.id)
                    if stream_state.final_text:
                        text, result = stream_state.final_text, stream_state.final_result
                    else:
                        text, result = stream_state.partial_text, stream_state.partial_result
                    if text:
                        logger.debug("STT stream result for %s: %s", session.id, redact(text))
                    else:
                        logger.debug(
                            "STT stream returned no text for %s, falling back to batch",
                            session.id,
                        )
                except Exception:
                    logger.exception("STT stream collection error for %s", session.id)
            elif stream_state is not None:
                logger.debug(
                    "STT stream unusable for %s (error=%s, cancelled=%s)",
                    session.id,
                    stream_state.error,
                    stream_state.cancelled,
                )

            # Batch fallback: no stream, stream error, or no final text
            if text is None:
                if stream_state is not None:
                    reason = "stream returned no text"
                    if stream_state.error:
                        reason = f"stream error: {stream_state.error}"
                    elif stream_state.cancelled:
                        reason = "stream was cancelled"
                    logger.info(
                        "STT stream→batch fallback for %s (%s) — "
                        "sending %d bytes to batch transcribe",
                        session.id,
                        reason,
                        len(audio),
                    )
                audio_frame = AudioFrame(
                    data=audio,
                    sample_rate=sample_rate,
                    channels=1,
                    sample_width=2,
                )
                _t = getattr(self._framework, "_telemetry", None)
                telemetry = _t if isinstance(_t, TelemetryProvider) else _NOOP
                t0 = time.monotonic()
                parent = getattr(self, "_voice_session_spans", {}).get(session.id)
                span_id = telemetry.start_span(
                    SpanKind.STT_TRANSCRIBE,
                    "stt.batch",
                    parent_id=parent,
                    room_id=room_id,
                    session_id=session.id,
                    channel_id=self.channel_id,
                    attributes={Attr.PROVIDER: self._stt.name, Attr.STT_MODE: "batch"},
                )
                try:
                    stt_result = await self._stt.transcribe(
                        audio_frame, **self._stt_call_kwargs(session.id)
                    )
                    text = stt_result.text
                    result = stt_result
                    ttfb_ms = (time.monotonic() - t0) * 1000
                    telemetry.end_span(
                        span_id,
                        attributes={
                            Attr.STT_TEXT_LENGTH: len(text) if text else 0,
                            Attr.TTFB_MS: round(ttfb_ms, 1),
                            Attr.DURATION_MS: round(ttfb_ms, 1),
                        },
                    )
                    telemetry.record_metric(
                        "roomkit.stt.duration_ms",
                        ttfb_ms,
                        unit="ms",
                        attributes={Attr.PROVIDER: self._stt.name},
                    )
                except Exception:
                    telemetry.end_span(span_id, status="error", error_message="batch STT failed")
                    raise

            if text is None:
                return

            elif stream_state is not None:
                # Record streaming STT metrics
                _t = getattr(self._framework, "_telemetry", None)
                telemetry = _t if isinstance(_t, TelemetryProvider) else _NOOP
                telemetry.record_metric(
                    "roomkit.stt.duration_ms",
                    0,
                    unit="ms",
                    attributes={
                        Attr.PROVIDER: self._stt.name,
                        Attr.STT_MODE: "stream",
                        Attr.STT_TEXT_LENGTH: len(text),
                    },
                )

            # The lock hears every final, an empty one included — a locked
            # language that stopped fitting shows up as empties and noise.
            if result is not None:
                self._observe_stt_language(session, result)

            if not text.strip():
                logger.debug("Empty transcription, skipping")
                return

            logger.debug("Transcription: %s", redact(text))

            # Send transcription to client UI (if backend supports it)
            backend = self._resolve_session_backend(session)
            if backend:
                await backend.send_transcription(session, text, "user")
            # Broadcast to other bridged participants
            await self._broadcast_bridge_transcription(session, text, room_id)

            # Fire ON_TRANSCRIPTION hooks (sync, can modify). With
            # pipeline_speakers the utterance carries the stage's speaker.
            speaker = await claimed_speaker(speaker_claim)
            tx_event = _transcription_event(
                session, text, result.language if result else None, speaker
            )
            transcription_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.ON_TRANSCRIPTION,
                tx_event,
                context,
                skip_event_filter=True,
            )

            if not transcription_result.allowed:
                logger.info("Transcription blocked by hook: %s", transcription_result.reason)
                return

            # Use potentially modified text
            final_text = _extract_transcription_text(transcription_result.event, text)
            if speaker is not None:
                speaker = _with_hook_sender_name(speaker, transcription_result.event)
            self._record_user_turn(
                session,
                text,
                final_text,
                audio=audio,
                sample_rate=sample_rate,
                dtmf_seen=dtmf_seen,
            )

            # The user continued a turn whose response they have not heard:
            # that response goes, and this transcript is routed on its own.
            settled = True
            if not await self._supersede_unheard_turn(session):
                self._release_unheard_turn(session.id)

            # Turn detection: if configured, evaluate before routing
            turn_detector = self._pipeline_config.turn_detector if self._pipeline_config else None
            if turn_detector is not None:
                await self._evaluate_turn(
                    session,
                    final_text,
                    room_id,
                    context,
                    audio_bytes=audio,
                    speaker=speaker,
                    transcript=transcript,
                )
            else:
                # No turn detector — route immediately
                await self._route_text(session, final_text, room_id, speaker=speaker)

        except Exception as exc:
            logger.exception("Error processing speech end")
            if self._framework:
                try:
                    await self._framework._emit_framework_event(
                        "stt_error",
                        room_id=room_id,
                        data={
                            "session_id": session.id,
                            "provider": self._stt.name if self._stt else "unknown",
                            "error": str(exc),
                        },
                    )
                except Exception:
                    logger.exception("Error emitting stt_error")
        finally:
            transcript.settle()
            if not settled:
                # No transcript, blocked or failed: the held response plays.
                self._release_unheard_turn(session.id)
            if _vs_token is not None:
                reset_span(_vs_token)

    # -----------------------------------------------------------------
    # Batch STT (no VAD — caller controls when to transcribe)
    # -----------------------------------------------------------------

    # ~5 minutes at 16 kHz mono 16-bit PCM
    _MAX_BATCH_BUFFER_BYTES = 10 * 1024 * 1024

    def _on_processed_frame_for_batch(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Accumulate processed frame into batch buffer."""
        buf = self._batch_audio_buffers.get(session.id)
        if buf is None:
            return
        if len(buf) + len(frame.data) > self._MAX_BATCH_BUFFER_BYTES:
            logger.warning("Batch buffer full for %s, dropping frame", session.id)
            return
        buf.extend(frame.data)
        if session.id not in self._batch_audio_sample_rate:
            self._batch_audio_sample_rate[session.id] = frame.sample_rate

    async def flush_stt(
        self,
        session: VoiceSession,
        *,
        route: bool = False,
    ) -> TranscriptionResult:
        """Transcribe accumulated batch audio and clear the buffer.

        Args:
            session: The voice session to flush.
            route: If ``True``, fire ON_SPEECH_END and ON_TRANSCRIPTION hooks
                and route the text through the inbound pipeline (same as VAD
                mode).  If ``False`` (default), return the result directly.

        Returns:
            The transcription result.

        Raises:
            RuntimeError: If the channel is not in batch mode.
        """
        if not self._batch_mode:
            raise RuntimeError("flush_stt() requires batch_mode=True")

        buf = self._batch_audio_buffers.get(session.id)
        sample_rate = self._batch_audio_sample_rate.get(session.id, 16000)

        # Empty or missing buffer → empty result
        if not buf:
            return TranscriptionResult(text="", is_final=True)

        # Snapshot and clear buffer
        audio_data = bytes(buf)
        buf.clear()
        dtmf_seen = self._tts_context is not None and self._tts_context.take_dtmf(session.id)

        if self._stt is None:
            raise RuntimeError("STT provider not configured")
        from roomkit.voice.audio_frame import AudioFrame

        audio_frame = AudioFrame(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
        )
        result = await self._stt.transcribe(audio_frame, **self._stt_call_kwargs(session.id))
        self._observe_stt_language(session, result)

        if route and result.text.strip() and self._framework:
            with self._state_lock:
                binding_info = self._session_bindings.get(session.id)
            if binding_info:
                room_id, _ = binding_info
                context = await self._framework._build_context(room_id)

                # Fire ON_SPEECH_END hooks
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_SPEECH_END,
                    session,
                    context,
                    skip_event_filter=True,
                )

                # Fire ON_TRANSCRIPTION hooks (sync, can modify/block)
                tx_event = TranscriptionEvent(
                    session=session, text=result.text, language=result.language
                )
                transcription_result = await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.ON_TRANSCRIPTION,
                    tx_event,
                    context,
                    skip_event_filter=True,
                )

                if transcription_result.allowed:
                    final_text = _extract_transcription_text(
                        transcription_result.event, result.text
                    )
                    self._record_user_turn(
                        session,
                        result.text,
                        final_text,
                        audio=audio_data,
                        sample_rate=sample_rate,
                        dtmf_seen=dtmf_seen,
                    )
                    await self._route_text(session, final_text, room_id)

        return result

    def clear_stt_buffer(self, session: VoiceSession) -> int:
        """Discard the accumulated batch audio buffer.

        Args:
            session: The voice session whose buffer to clear.

        Returns:
            Number of bytes discarded.

        Raises:
            RuntimeError: If the channel is not in batch mode.
        """
        if not self._batch_mode:
            raise RuntimeError("clear_stt_buffer() requires batch_mode=True")

        buf = self._batch_audio_buffers.get(session.id)
        if buf is None:
            return 0
        count = len(buf)
        buf.clear()
        return count

    def stt_buffer_size(self, session: VoiceSession) -> int:
        """Return the current batch buffer size in bytes.

        Returns 0 if the channel is not in batch mode or the session
        has no buffer.
        """
        buf = self._batch_audio_buffers.get(session.id)
        return len(buf) if buf is not None else 0
