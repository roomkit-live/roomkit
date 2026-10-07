"""VoiceChannel mixin — TTS delivery (streaming and non-streaming)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._stream_fanout import StreamBranch, StreamFanOut
from roomkit.channels._tts_sentence_gate import SentenceHookGate
from roomkit.channels._voice_unheard import UnheardTurns
from roomkit.models.enums import EventType, HookTrigger, Visibility
from roomkit.telemetry.base import Attr, SpanKind, TelemetryProvider
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.telemetry.redaction import redact
from roomkit.voice.base import VoiceCapability, require_pcm16

_NOOP = NoopTelemetryProvider()

if TYPE_CHECKING:
    import threading

    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.tts.base import TTSProvider
    from roomkit.voice.tts.context import AssistantTurnRecorder, TTSContextStore

    from .voice import TTSPlaybackState

logger = logging.getLogger("roomkit.voice")

# Time to keep _playing_sessions alive after send_audio() returns when nothing
# cancels echo.  Accounts for residual room echo/reverb after the speaker
# physically finishes playing.  During this period, speech is discarded as echo.
_PLAYBACK_DRAIN_S = 2.0

# Time the pipeline's AEC keeps cancelling after send_audio() returns: the
# room's echo tail, which a converged filter removes instead of the drain
# window discarding everything said over it.
_AEC_ECHO_TAIL_S = 0.5


@runtime_checkable
class TTSHost(Protocol):
    """Contract: capabilities a host class must provide for VoiceTTSMixin.

    Every attribute listed here is initialized by ``VoiceChannel.__init__``.

    Attributes:
        channel_id: Unique identifier for this channel instance.
        _framework: Reference to the owning RoomKit framework (None before registration).
        _tts: Text-to-speech provider (None when not configured).
        _backend: The voice backend transport (None before configuration).
        _pipeline: The audio processing pipeline (None when no pipeline).
        _pipeline_config: Audio pipeline configuration (None when no pipeline).
        _session_bindings: Map of session ID to (room_id, binding) pairs.
        _playing_sessions: Active TTS playback states per session.
        _playback_done_events: Signals that send_audio() has returned for a session.
        _delivered_tts_events: LRU set of event IDs already delivered via TTS.
        _last_tts_ended_at: Monotonic timestamp of last TTS completion per session.
        _last_output_level_at: Throttle timestamp for ON_OUTPUT_AUDIO_LEVEL per session.
        _debug_frame_count: Counter for RMS debug logging.
        _voice_map: Per-channel TTS voice overrides.
        _tts_filter: Optional callable to filter/transform TTS text (Callable[[str], str] | None).
        _tts_context: Per-session dialogue for a context-aware TTS (None when unused).
        _unheard_turns: The routed turn per session whose response is not heard yet.
        _state_lock: Threading lock protecting shared mutable state.

    Methods provided by VoiceChannel or sibling mixins:
        _schedule: Schedule a coroutine as a tracked background task.
        _fire_audio_level_hook: Fire ON_OUTPUT_AUDIO_LEVEL hook (VoiceHooksMixin).
        interrupt: Cancel in-progress TTS playback (VoiceChannel).
    """

    channel_id: str
    _framework: RoomKit | None
    _tts: TTSProvider | None
    _backend: VoiceBackend | None
    _pipeline: AudioPipeline | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _playing_sessions: dict[str, TTSPlaybackState]
    _playback_done_events: dict[str, asyncio.Event]
    _delivered_tts_events: OrderedDict[str, None]
    _last_tts_ended_at: dict[str, float]
    _last_output_level_at: dict[str, float]
    _debug_frame_count: int
    _voice_map: dict[str, str]
    _tts_filter: Any  # Callable[[str], str] | None
    _tts_context: TTSContextStore | None
    _unheard_turns: UnheardTurns
    _state_lock: threading.Lock
    _schedule: Any  # VoiceChannel._schedule
    _resolve_session_backend: Any  # VoiceChannel._resolve_session_backend
    _fire_audio_level_hook: Any  # VoiceHooksMixin._fire_audio_level_hook
    _fire_level_hook: Any  # VoiceChannel._fire_level_hook
    interrupt: Any  # VoiceChannel.interrupt


class VoiceTTSMixin:
    """TTS delivery helpers for VoiceChannel.

    Host contract: :class:`TTSHost`.
    """

    # -- attributes provided by VoiceChannel.__init__ (see TTSHost) --
    channel_id: str
    _framework: RoomKit | None
    _tts: TTSProvider | None
    _backend: VoiceBackend | None
    _pipeline: AudioPipeline | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _playing_sessions: dict[str, TTSPlaybackState]
    _playback_done_events: dict[str, asyncio.Event]
    _delivered_tts_events: OrderedDict[str, None]
    _last_tts_ended_at: dict[str, float]
    _last_output_level_at: dict[str, float]
    _debug_frame_count: int
    _voice_map: dict[str, str]
    _tts_filter: Any  # Callable[[str], str] | None
    _tts_context: TTSContextStore | None
    _unheard_turns: UnheardTurns
    _state_lock: Any  # threading.Lock — see TTSHost

    # -- cross-mixin methods (annotated as Any to avoid MRO shadowing) --
    _schedule: Any  # see TTSHost — VoiceChannel._schedule
    _resolve_session_backend: Any  # see TTSHost — VoiceChannel._resolve_session_backend
    _fire_audio_level_hook: Any  # see TTSHost — VoiceHooksMixin._fire_audio_level_hook
    _fire_level_hook: Any  # see TTSHost — VoiceChannel._fire_level_hook
    interrupt: Any  # see TTSHost — VoiceChannel.interrupt
    _flush_queued_speech: Any  # see TTSHost — VoiceChannel._flush_queued_speech

    def _session_output_backend(self, session: VoiceSession) -> VoiceBackend:
        """The transport serving *session*: one added with ``add_backend``, or the primary.

        Callers have already checked that the channel has a backend.
        """
        backend = self._resolve_session_backend(session)
        assert backend is not None
        return backend

    def _fire_output_level(self, session: VoiceSession, data: bytes) -> None:
        """Fire ON_OUTPUT_AUDIO_LEVEL hook from the outbound pipeline, throttled."""
        self._fire_level_hook(
            session, data, self._last_output_level_at, HookTrigger.ON_OUTPUT_AUDIO_LEVEL
        )

    def _resolve_voice(self, channel_id: str) -> str | None:
        """Look up TTS voice override for *channel_id* via voice_map."""
        return self._voice_map.get(channel_id) if self._voice_map else None

    def _begin_tts_turn(
        self, session: VoiceSession, telemetry: TelemetryProvider | None
    ) -> tuple[dict[str, Any], AssistantTurnRecorder | None]:
        """The ``context`` kwarg for one synthesis call, and its turn recorder.

        Empty for a provider that consumes no context: a NONE provider is
        called without ``context`` (RFC §12.2.2 passing rules).
        """
        store = self._tts_context
        if store is None:
            return {}, None
        context, recorder = store.begin_assistant_turn(session.id)
        if telemetry is not None:
            attrs = {Attr.PROVIDER: self._tts.name if self._tts else "unknown"}
            telemetry.record_metric(
                "pipeline.tts_context_turns", float(len(context.turns)), attributes=attrs
            )
            telemetry.record_metric(
                "pipeline.tts_context_audio_s",
                store.audio_seconds(session.id),
                unit="s",
                attributes=attrs,
            )
        return {"context": context}, recorder

    def _end_tts_turn(self, playback: TTSPlaybackState) -> None:
        """Record what the user heard of a synthesis call in its context.

        Runs once per call: at the interruption, so the call that replaces it
        already sees the cut turn, or when the call ends.
        """
        pending, playback.context_turn = playback.context_turn, None
        if pending is None or self._tts_context is None:
            return
        recorder, speaker_id = pending
        interrupted = playback.stopped_at is not None
        if playback.first_audio_at is None:
            # No chunk reached the transport: the user heard nothing of it.
            played_ms = 0
        elif interrupted or playback.audio_ms is None:
            played_ms = playback.played_ms
        else:
            # Not cut off: all of it plays, even when the transport returned
            # before the last sample left the speaker.
            played_ms = int(playback.audio_ms)
        self._tts_context.commit_assistant_turn(
            recorder, speaker_id, playback.text, played_ms=played_ms, interrupted=interrupted
        )

    def _release_tts_context(self, session_id: str) -> None:
        """Drop a session's TTS context, here and in the provider."""
        if self._tts_context is None:
            return
        self._tts_context.release(session_id)
        if self._tts is None:
            return
        try:
            self._tts.release_context(session_id)
        except Exception:
            logger.exception("TTS provider failed to release context %s", session_id)

    async def _wrap_outbound(
        self, session: VoiceSession, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Wrap a TTS stream through the pipeline outbound path.

        Each chunk is converted to an AudioFrame, processed through
        ``pipeline.process_outbound()`` (which feeds AEC reference,
        runs postprocessors, recorder taps, and outbound resampler),
        then converted back to an AudioChunk for the backend.
        """
        from roomkit.voice.audio_frame import AudioFrame
        from roomkit.voice.base import AudioChunk as OutChunk

        chunk_idx = 0
        total_in_bytes = 0
        total_out_bytes = 0
        async for chunk in chunks:
            if not chunk.data or self._pipeline is None:
                for cb in getattr(self, "_outbound_audio_taps", []):
                    try:
                        cb(session, chunk.data, chunk.sample_rate)
                    except Exception:
                        logger.debug("Outbound audio tap error", exc_info=True)
                yield chunk
                continue
            frame = AudioFrame(
                data=chunk.data,
                sample_rate=chunk.sample_rate,
                channels=chunk.channels,
                sample_width=2,
                timestamp_ms=chunk.timestamp_ms,
            )
            processed = self._pipeline.process_outbound(session, frame)
            total_in_bytes += len(chunk.data)
            total_out_bytes += len(processed.data)
            if chunk_idx < 3 or chunk_idx % 50 == 0:
                logger.debug(
                    "outbound[%d] in=%dB@%dHz out=%dB@%dHz",
                    chunk_idx,
                    len(chunk.data),
                    chunk.sample_rate,
                    len(processed.data),
                    processed.sample_rate,
                )
            chunk_idx += 1
            # Fire outbound audio taps (e.g. avatar lip-sync)
            for cb in getattr(self, "_outbound_audio_taps", []):
                try:
                    cb(session, processed.data, processed.sample_rate)
                except Exception:
                    logger.debug("Outbound audio tap error", exc_info=True)
            # Fire ON_OUTPUT_AUDIO_LEVEL from the outbound pipeline path.
            # This works regardless of backend playback-callback support.
            self._fire_output_level(session, processed.data)
            yield OutChunk(
                data=processed.data,
                sample_rate=processed.sample_rate,
                channels=processed.channels,
                format=chunk.format,
                timestamp_ms=(
                    int(processed.timestamp_ms) if processed.timestamp_ms is not None else None
                ),
                is_final=chunk.is_final,
            )
        logger.debug(
            "outbound done: %d chunks, in=%dB out=%dB (ratio=%.2f)",
            chunk_idx,
            total_in_bytes,
            total_out_bytes,
            total_out_bytes / total_in_bytes if total_in_bytes else 0,
        )

    def _playback_sent(self, session_id: str) -> None:
        """End a playback whose audio ``send_audio()`` has delivered.

        The room may still carry residual echo.  How it is kept out of the STT
        depends on whether something cancels it:

        - **An AEC** — the pipeline's, or the backend's own (``NATIVE_AEC``,
          e.g. ``LocalAudioBackend(aec=...)``): the playback ends here, at
          once — speech from now on is the user's, even a reply started the
          moment the agent stops.  The pipeline's AEC stays active for
          ``_AEC_ECHO_TAIL_S`` to cancel the echo tail, then is bypassed so
          user audio passes unchanged; its converged filter is preserved for
          the next playback turn.  A backend's AEC is the backend's to manage.
        - **No AEC**: :meth:`_finish_playback` keeps ``_playing_sessions``
          alive for ``_PLAYBACK_DRAIN_S`` so the echo transcribed in that
          window is discarded — along with any speech, which cannot be told
          apart from it.
        """
        pipeline_aec = self._pipeline is not None and self._pipeline.runs_aec
        backend_aec = (
            self._backend is not None and VoiceCapability.NATIVE_AEC in self._backend.capabilities
        )
        if not pipeline_aec and not backend_aec:
            self._schedule(
                self._finish_playback(session_id),
                name=f"finish_playback:{session_id}",
            )
            return
        self._end_playback(session_id, drain_s=0.0)
        if pipeline_aec:
            self._schedule(
                self._bypass_aec_after_echo_tail(session_id),
                name=f"aec_echo_tail:{session_id}",
            )
        # The bot has finished — anything the DISABLED strategy queued while it
        # spoke gets its turn now (RFC §12.6).
        self._schedule(
            self._flush_queued_speech(session_id),
            name=f"flush_queued_speech:{session_id}",
        )

    async def _finish_playback(self, session_id: str) -> None:
        """Clear playback state after the echo-decay window (no AEC).

        If ``interrupt()`` fires during the delay, it pops
        ``_playing_sessions`` immediately — the delayed pop becomes a no-op.
        """
        await asyncio.sleep(_PLAYBACK_DRAIN_S)
        self._end_playback(session_id, drain_s=_PLAYBACK_DRAIN_S)
        # The bot has finished — anything the DISABLED strategy queued while it
        # spoke gets its turn now (RFC §12.6).
        await self._flush_queued_speech(session_id)

    def _end_playback(self, session_id: str, *, drain_s: float) -> None:
        """Drop the playback state and stamp when the agent stopped speaking."""
        with self._state_lock:
            playback = self._playing_sessions.pop(session_id, None)
        if playback:
            self._last_tts_ended_at[session_id] = time.monotonic()
            logger.debug(
                "Playback drain complete for session %s (delay=%.1fs)",
                session_id,
                drain_s,
            )

    async def _bypass_aec_after_echo_tail(self, session_id: str) -> None:
        """Bypass the AEC once the echo tail has decayed, unless a new playback started."""
        await asyncio.sleep(_AEC_ECHO_TAIL_S)
        if self._pipeline is None:
            return
        with self._state_lock:
            # A playback started meanwhile owns the AEC again: leave it on.
            if session_id in self._playing_sessions:
                return
            self._pipeline.set_aec_active(session_id, False)

    def _find_sessions(self, room_id: str, binding: ChannelBinding) -> list[VoiceSession]:
        """Find voice sessions for a room/binding pair."""
        if not self._backend:
            return []

        with self._state_lock:
            bindings_snapshot = list(self._session_bindings.items())
        target_sessions: list[VoiceSession] = []
        for session_id, (bound_room_id, bound_binding) in bindings_snapshot:
            if bound_room_id == room_id and bound_binding.channel_id == binding.channel_id:
                session = self._backend.get_session(session_id)
                if session:
                    target_sessions.append(session)

        if not target_sessions:
            target_sessions = self._backend.list_sessions(room_id)
        return target_sessions

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Deliver a streaming AI response via TTS."""
        from roomkit.models.channel import ChannelOutput as ChannelOutputModel

        if not self._tts or not self._backend:
            return ChannelOutputModel.empty()

        # Defense-in-depth: skip system/internal events (primary guard is in deliver())
        if event.type == EventType.SYSTEM or event.visibility == Visibility.INTERNAL:
            return ChannelOutputModel.empty()

        # Track event to prevent duplicate TTS delivery (streaming + broadcast)
        self._delivered_tts_events[event.id] = None
        if len(self._delivered_tts_events) > 200:
            # Evict oldest 10%
            for _ in range(20):
                if self._delivered_tts_events:
                    self._delivered_tts_events.popitem(last=False)

        tts_name = self._tts.name  # capture before async yields (may become None)
        room_id = event.room_id
        target_sessions = self._find_sessions(room_id, binding)

        text = _ResponseText()

        _t = getattr(self._framework, "_telemetry", None) if self._framework else None
        telemetry: TelemetryProvider | None = _t if isinstance(_t, TelemetryProvider) else None

        # Capture parent span BEFORE playback — session may be unbound during TTS.
        _vs_parent = getattr(self, "_voice_session_spans", {}).get(
            target_sessions[0].id if target_sessions else ""
        )

        # The stream is read, filtered and split once, then every session gets
        # its own copy of the sentences: reading it drives persistence upstream,
        # and each session must hear the whole response.
        sentence_source, gate = self._sentences(text.read(text_stream), room_id, context)
        fan_out = StreamFanOut(sentence_source, len(target_sessions))
        producer = asyncio.create_task(fan_out.run(), name=f"tts_fan_out:{event.id}")
        voice = self._resolve_voice(event.source.channel_id)
        try:
            results = await asyncio.gather(
                *(
                    self._stream_to_session(
                        session,
                        branch,
                        voice=voice,
                        room_id=room_id,
                        tts_name=tts_name,
                        telemetry=telemetry,
                        accumulated=text.accumulated,
                        speaker_id=event.source.channel_id,
                        responds_to=event.responds_to,
                    )
                    for session, branch in zip(target_sessions, fan_out.branches, strict=True)
                ),
                return_exceptions=True,
            )
        except BaseException:
            # Cancelled from outside (the turn was interrupted on purpose): the
            # source goes down with the turn, as it did with a single reader.
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            raise
        if not producer.done():
            # Every session stopped early (barge-in) while the next item was
            # being pulled.  The user said stop, so that pull is cancelled
            # rather than left to finish: a token or a tool call arriving now
            # would be generated for nobody (RFC §12.2 step 13s).  The caller
            # sees the stream was not read to its end and closes it.
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        # A failure of the source the sessions share (a filter) is the
        # response's, not one session's: it takes the caller's error path.
        results = await self._stop_failed_sessions(
            room_id, tts_name, target_sessions, results, source_error=fan_out.error
        )
        if fan_out.error is not None:
            raise fan_out.error
        delivered = _served_sessions(target_sessions, results)

        full_text = self._streamed_text(text.accumulated, gate)
        await self._close_streamed_response(delivered, full_text, room_id, context, _vs_parent)
        if text.failure is not None:
            # The response failed: what it produced was spoken, the failure is the turn's.
            raise text.failure
        return ChannelOutputModel.empty()

    def _sentences(
        self, tokens: AsyncIterator[str], room_id: str, context: RoomContext
    ) -> tuple[AsyncIterator[str], SentenceHookGate | None]:
        """The sentences the sessions read from *tokens*: filtered, split, then gated."""
        from roomkit.voice.tts.sentence_splitter import split_sentences

        token_source = tokens
        if self._tts_filter is not None:
            from roomkit.voice.tts.filters import TTSStreamFilter, filtered_stream

            if isinstance(self._tts_filter, TTSStreamFilter):
                token_source = filtered_stream(token_source, self._tts_filter)
            else:
                token_source = _filter_sentences_plain(token_source, self._tts_filter)
        sentence_source = split_sentences(token_source)
        gate = self._sentence_gate(room_id, context)
        if gate is not None:
            sentence_source = gate.run(sentence_source)
        return sentence_source, gate

    async def _stop_failed_sessions(
        self,
        room_id: str,
        provider: str,
        sessions: list[VoiceSession],
        results: list[Any],
        *,
        source_error: Exception | None,
    ) -> list[Any]:
        """Count a session whose synthesis failed as stopped early (RFC §12.2 step 12s.d).

        A TTS failing on one session is that session's early stop, as a
        barge-in is, not the response's failure: it is reported as
        ``tts_error``, and once every session has stopped the response is
        stored as it stood, cancelled (step 13s), never replayed.
        """
        stopped: list[Any] = []
        for session, result in zip(sessions, results, strict=True):
            # A failure of the shared source reaches every branch: it is the response's.
            if isinstance(result, Exception) and result is not source_error:
                logger.error(
                    "Streaming TTS failed for session %s; it stops here",
                    session.id,
                    exc_info=result,
                )
                await self._report_tts_failure(room_id, provider, result, session.id)
                result = False
            stopped.append(result)
        return stopped

    async def _close_streamed_response(
        self,
        delivered: list[VoiceSession],
        full_text: str,
        room_id: str,
        context: RoomContext,
        parent_span: str | None,
    ) -> None:
        """Close a streamed response on the sessions that heard it (RFC §12.2 step 13s).

        Each gets the whole text as its playback and final transcript, and
        AFTER_TTS reports what was sent; with no such session, nothing was.
        """
        from .voice import TTSPlaybackState

        if not delivered:
            return

        # Replace the relayed prefix with the whole streamed text
        for session in delivered:
            with self._state_lock:
                previous = self._playing_sessions.get(session.id)
                if previous is not None:
                    # A barge-in that claimed the prefix owns the whole text.
                    self._playing_sessions[session.id] = TTSPlaybackState(
                        session_id=session.id,
                        text=full_text or "(empty)",
                        barge_in_claimed=previous.barge_in_claimed,
                        answer_channel_id=previous.answer_channel_id,
                        answer_responds_to=previous.answer_responds_to,
                    )
        # Only a session that was served gets the final transcript: showing a
        # response the user never heard would contradict the audio.
        if full_text:
            for session in delivered:
                await self._session_output_backend(session).send_transcription(
                    session, full_text, "assistant"
                )

        # BEFORE_TTS ran on each sentence (12s.b); AFTER_TTS reports what was sent
        if full_text:
            await self._run_after_tts(room_id, full_text, context, parent_span)

    async def _run_after_tts(
        self, room_id: str, text: str, context: RoomContext, parent_span: str | None
    ) -> None:
        """Run AFTER_TTS on *text*, under the voice session's span when there is one."""
        if self._framework is None:
            return
        from roomkit.telemetry.context import reset_span, set_current_span

        _tok = set_current_span(parent_span) if parent_span else None
        try:
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.AFTER_TTS,
                text,
                context,
                skip_event_filter=True,
            )
        finally:
            if _tok is not None:
                reset_span(_tok)

    def _sentence_gate(self, room_id: str, context: RoomContext) -> SentenceHookGate | None:
        """BEFORE_TTS on each sentence of a streamed response (RFC §12.2 step 12s.b).

        ``None`` when no BEFORE_TTS hook is registered: the sentences then go to
        the TTS untouched, as they always did.
        """
        if self._framework is None:
            return None
        hooks = self._framework.hook_engine
        if not hooks.has_hooks(HookTrigger.BEFORE_TTS):
            return None
        return SentenceHookGate(hooks, room_id, context)

    def _streamed_text(self, accumulated: list[str], gate: SentenceHookGate | None) -> str:
        """The text a streamed response leaves for its transcript and AFTER_TTS.

        What the sessions were sent: the sentences as BEFORE_TTS left them when
        a hook changed or dropped one, the whole filtered stream otherwise.
        """
        if gate is not None and gate.changed:
            return gate.text()
        full_text = "".join(accumulated)
        if self._tts_filter is not None and full_text:
            full_text = self._tts_filter(full_text)
        return full_text

    async def _stream_to_session(
        self,
        session: VoiceSession,
        sentences: StreamBranch[str],
        *,
        voice: str | None,
        room_id: str,
        tts_name: str,
        telemetry: TelemetryProvider | None,
        accumulated: list[str],
        speaker_id: str,
        responds_to: str | None = None,
    ) -> bool:
        """Play one session's copy of a streamed response through TTS.

        Returns whether the session was served.  Its branch is closed on every
        exit, so the producer stops once no session reads any more.
        """
        if self._tts is None or self._backend is None:
            sentences.close()
            return False
        tts, backend = self._tts, self._backend
        try:
            await self._play_branch(
                session,
                sentences,
                tts=tts,
                backend=backend,
                voice=voice,
                room_id=room_id,
                tts_name=tts_name,
                telemetry=telemetry,
                accumulated=accumulated,
                speaker_id=speaker_id,
                responds_to=responds_to,
            )
        finally:
            sentences.close()
        return True

    async def _play_branch(
        self,
        session: VoiceSession,
        sentences: StreamBranch[str],
        *,
        tts: TTSProvider,
        backend: VoiceBackend,
        voice: str | None,
        room_id: str,
        tts_name: str,
        telemetry: TelemetryProvider | None,
        accumulated: list[str],
        speaker_id: str,
        responds_to: str | None = None,
    ) -> None:
        """Interrupt, play and drain one session's streamed response."""
        from .voice import TTSPlaybackState

        # Cancel any existing TTS to prevent overlapping audio
        with self._state_lock:
            existing = self._playing_sessions.get(session.id)
        if existing:
            logger.info(
                "Cancelling previous TTS for session %s before starting new one",
                session.id,
            )
            await self.interrupt(session, reason="new_tts")

        playback = TTSPlaybackState(
            session_id=session.id,
            text="",
            answer_channel_id=speaker_id,
            answer_responds_to=responds_to,
        )
        with self._state_lock:
            self._playing_sessions[session.id] = playback
            # Clear done event so wait_playback_done() blocks until send_audio returns
            done_ev = self._playback_done_events.get(session.id)
            if done_ev is None:
                done_ev = asyncio.Event()
                self._playback_done_events[session.id] = done_ev
            else:
                done_ev.clear()
        # Activate AEC so echo cancellation runs during playback
        if self._pipeline is not None and self._pipeline._config.aec is not None:
            self._pipeline.set_aec_active(session.id, True)
        t0 = time.monotonic()
        logger.info("Streaming TTS playback started for session %s", session.id)

        span_id = None
        if telemetry is not None:
            parent = getattr(self, "_voice_session_spans", {}).get(session.id)
            span_id = telemetry.start_span(
                SpanKind.TTS_SYNTHESIZE,
                "tts.stream",
                parent_id=parent,
                room_id=room_id,
                session_id=session.id,
                channel_id=self.channel_id,
                attributes={Attr.PROVIDER: tts_name},
            )

        # Relay each sentence to the client before TTS synthesis, and keep
        # the text handed to TTS so far on the playback state: a barge-in
        # records it as the interrupted utterance (RFC §12.3.13 step 2).
        relayed: list[str] = []

        async def relay_sentences() -> AsyncIterator[str]:
            async for sentence in sentences:
                await backend.send_transcription(session, sentence, "assistant_interim")
                relayed.append(sentence.strip())
                with self._state_lock:
                    playback.text = " ".join(relayed)
                yield sentence

        context_kwargs, recorder = self._begin_tts_turn(session, telemetry)
        if recorder is not None:
            playback.context_turn = (recorder, speaker_id)
        tts_stream: AsyncIterator[AudioChunk] | None = None
        try:
            tts_stream = tts.synthesize_stream_input(
                relay_sentences(), voice=voice, **context_kwargs
            )
            # A response the user resumes speaking over waits for them (RFC §12.3.12).
            audio = _observe_audio(
                playback,
                recorder,
                tts_stream,
                gate=partial(self._unheard_turns.wait_to_play, session.id),
            )
            if self._pipeline is not None or getattr(self, "_outbound_audio_taps", []):
                audio = self._wrap_outbound(session, audio)
            await backend.send_audio(session, audio)
        except Exception:
            if telemetry is not None and span_id is not None:
                telemetry.end_span(span_id, status="error", error_message="stream TTS failed")
                span_id = None
            raise
        finally:
            self._end_tts_turn(playback)
            await _close_stream(tts_stream)
            duration_ms = (time.monotonic() - t0) * 1000
            if telemetry is not None and span_id is not None:
                telemetry.end_span(
                    span_id,
                    attributes={
                        Attr.DURATION_MS: round(duration_ms, 1),
                        Attr.TTS_CHAR_COUNT: len("".join(accumulated)),
                    },
                )
                telemetry.record_metric(
                    "roomkit.tts.duration_ms",
                    duration_ms,
                    unit="ms",
                    attributes={Attr.PROVIDER: tts_name},
                )
            logger.debug(
                "Streaming TTS send_audio returned for session %s (%.1fs), draining",
                session.id,
                time.monotonic() - t0,
            )
            self._debug_frame_count = 0  # reset RMS debug counter
            # Signal that send_audio() has returned so
            # wait_playback_done() can unblock immediately.
            done_ev = self._playback_done_events.get(session.id)
            if done_ev is not None:
                done_ev.set()
            self._playback_sent(session.id)

    async def _send_tts(
        self,
        session: VoiceSession,
        text: str,
        *,
        voice: str | None = None,
        speaker_id: str | None = None,
        response: bool = False,
        responds_to: str | None = None,
    ) -> None:
        """Speak *text* on *session*, reporting a failure once the playback has ended.

        What ``say()`` and a delivery call. The report comes after the
        playback is released, so a ``tts_error`` handler that waits for it
        does not wait on this session.
        """
        provider = self._tts.name if self._tts else "unknown"
        try:
            await self._play_tts(
                session,
                text,
                voice=voice,
                speaker_id=speaker_id,
                response=response,
                responds_to=responds_to,
            )
        except Exception as exc:
            await self._report_tts_failure(self._room_of(session), provider, exc, session.id)
            raise

    def _room_of(self, session: VoiceSession) -> str:
        """The room *session* speaks in: its binding's, else the one it joined."""
        with self._state_lock:
            binding = self._session_bindings.get(session.id)
        return binding[0] if binding else session.room_id

    async def _play_tts(
        self,
        session: VoiceSession,
        text: str,
        *,
        voice: str | None = None,
        speaker_id: str | None = None,
        response: bool = False,
        responds_to: str | None = None,
    ) -> None:
        """Synthesize *text* and send audio to *session*.

        Handles transcription, playback state tracking, streaming synthesis
        with pipeline wrapping, and fallback to batch synthesis. A *response*
        to a routed turn waits while the user resumes speaking (RFC §12.3.12);
        ``say()`` does not.
        """
        import time as _time

        from .voice import TTSPlaybackState

        if self._tts is None:
            raise RuntimeError("TTS provider not configured")
        if self._backend is None:
            raise RuntimeError("Voice backend not configured")

        tts_name = self._tts.name  # capture before async yields (may become None)

        # Cancel any existing TTS to prevent overlapping audio
        with self._state_lock:
            existing = self._playing_sessions.get(session.id)
        if existing:
            logger.info(
                "Cancelling previous TTS for session %s before starting new one",
                session.id,
            )
            await self.interrupt(session, reason="new_tts")

        await self._session_output_backend(session).send_transcription(session, text, "assistant")

        playback = TTSPlaybackState(
            session_id=session.id,
            text=text,
            answer_channel_id=speaker_id if response else None,
            answer_responds_to=responds_to if response else None,
        )
        with self._state_lock:
            self._playing_sessions[session.id] = playback
            # Clear done event so wait_playback_done() blocks until send_audio returns
            done_ev = self._playback_done_events.get(session.id)
            if done_ev is None:
                done_ev = asyncio.Event()
                self._playback_done_events[session.id] = done_ev
            else:
                done_ev.clear()
        # Activate AEC so echo cancellation runs during playback
        if self._pipeline is not None and self._pipeline._config.aec is not None:
            self._pipeline.set_aec_active(session.id, True)

        # Resolve telemetry provider
        _t = getattr(self._framework, "_telemetry", None) if self._framework else None
        telemetry: TelemetryProvider | None = _t if isinstance(_t, TelemetryProvider) else None

        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        room_id = binding_info[0] if binding_info else None

        span_id = None
        if telemetry is not None:
            parent = getattr(self, "_voice_session_spans", {}).get(session.id)
            span_id = telemetry.start_span(
                SpanKind.TTS_SYNTHESIZE,
                "tts.synthesize",
                parent_id=parent,
                room_id=room_id,
                session_id=session.id,
                channel_id=self.channel_id,
                attributes={
                    Attr.PROVIDER: tts_name,
                    Attr.TTS_CHAR_COUNT: len(text),
                    Attr.TTS_TEXT_LENGTH: len(text),
                },
            )
            if voice:
                telemetry.set_attribute(span_id, Attr.TTS_VOICE, voice)

        context_kwargs, recorder = self._begin_tts_turn(session, telemetry)
        if recorder is not None:
            playback.context_turn = (recorder, speaker_id or self.channel_id)
        t0 = _time.monotonic()
        tts_stream: AsyncIterator[AudioChunk] | None = None
        try:
            tts_stream = self._tts.synthesize_stream(text, voice=voice, **context_kwargs)
            gate = partial(self._unheard_turns.wait_to_play, session.id) if response else None
            audio_stream = _observe_audio(playback, recorder, tts_stream, gate=gate)
            if self._pipeline is not None or getattr(self, "_outbound_audio_taps", []):
                audio_stream = self._wrap_outbound(session, audio_stream)
            await self._session_output_backend(session).send_audio(session, audio_stream)
        except Exception:
            if telemetry is not None and span_id is not None:
                telemetry.end_span(span_id, status="error", error_message="TTS failed")
                span_id = None  # prevent double-end
            raise
        finally:
            self._end_tts_turn(playback)
            await _close_stream(tts_stream)
            duration_ms = (_time.monotonic() - t0) * 1000
            if telemetry is not None and span_id is not None:
                telemetry.end_span(
                    span_id,
                    attributes={Attr.DURATION_MS: round(duration_ms, 1)},
                )
                telemetry.record_metric(
                    "roomkit.tts.duration_ms",
                    duration_ms,
                    unit="ms",
                    attributes={Attr.PROVIDER: tts_name},
                )
            # Signal that send_audio() has returned so
            # wait_playback_done() can unblock immediately.
            done_ev = self._playback_done_events.get(session.id)
            if done_ev is not None:
                done_ev.set()
            self._playback_sent(session.id)

    async def _deliver_voice(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> None:
        if not self._tts or not self._backend or not self._framework:
            return

        from roomkit.models.event import TextContent

        if not isinstance(event.content, TextContent):
            return

        text = event.content.body
        room_id = event.room_id

        # Skip if already delivered via deliver_stream() or a prior deliver() call
        if event.id in self._delivered_tts_events:
            logger.debug("Skipping duplicate TTS for event %s", event.id)
            return
        self._delivered_tts_events[event.id] = None
        if len(self._delivered_tts_events) > 200:
            # Evict oldest 10%
            for _ in range(20):
                if self._delivered_tts_events:
                    self._delivered_tts_events.popitem(last=False)

        try:
            before_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.BEFORE_TTS,
                text,
                context,
                skip_event_filter=True,
            )

            if not before_result.allowed:
                logger.info("TTS blocked by hook: %s", before_result.reason)
                return

            final_text = before_result.event if isinstance(before_result.event, str) else text

            if self._tts_filter is not None:
                final_text = self._tts_filter(final_text)
                if not final_text:
                    logger.debug("TTS text empty after filter — skipping")
                    return

            logger.debug("AI response: %s", redact(final_text))

            target_sessions = self._find_sessions(room_id, binding)

            # Capture parent span BEFORE _send_tts — session may be unbound
            # during playback, removing it from _voice_session_spans.
            _parent = getattr(self, "_voice_session_spans", {}).get(
                target_sessions[0].id if target_sessions else ""
            )

            voice = self._resolve_voice(event.source.channel_id)
            # Sessions play side by side; a failed one does not hold the others
            # back, and only a delivery nobody received takes the error path.
            speaker_id = event.source.channel_id
            results = await asyncio.gather(
                *(
                    self._send_tts(
                        s,
                        final_text,
                        voice=voice,
                        speaker_id=speaker_id,
                        response=True,
                        responds_to=event.responds_to,
                    )
                    for s in target_sessions
                ),
                return_exceptions=True,
            )
            if _served_sessions(target_sessions, results):
                await self._run_after_tts(room_id, final_text, context, _parent)

        except Exception:
            # Each session's failed synthesis was reported as it failed (_send_tts).
            logger.exception("Error delivering voice audio")

    async def _report_tts_failure(
        self, room_id: str | None, provider: str, error: BaseException, session_id: str
    ) -> None:
        """Report one session's failed synthesis, once, as the ``tts_error`` event.

        RFC section 8.2. The failure itself is logged by whoever caught it; a
        provider without streaming synthesis is named here as the
        misconfiguration it is. A failure to emit is logged, never raised.
        """
        if isinstance(error, NotImplementedError):
            logger.error(
                "TTS provider %s does not support streaming synthesis; "
                "voice channels require synthesize_stream(). No audio sent.",
                provider,
            )
        if self._framework is None or room_id is None:
            return
        data: dict[str, Any] = {
            "provider": provider,
            "error": str(error),
            "session_id": session_id,
        }
        try:
            await self._framework._emit_framework_event("tts_error", room_id=room_id, data=data)
        except Exception:
            logger.exception("Error emitting tts_error")

    # -------------------------------------------------------------------------
    # Public API: say() and play()
    # -------------------------------------------------------------------------

    async def say(self, session: VoiceSession, text: str, *, voice: str | None = None) -> None:
        """Synthesize *text* and play it to the participant.

        Args:
            session: The voice session to speak into.
            text: The text to synthesize.
            voice: Optional voice override for this utterance.

        Raises:
            VoiceNotConfiguredError: If no TTS provider is configured.
            VoiceBackendNotConfiguredError: If no voice backend is configured.
        """
        from roomkit.core.framework import VoiceBackendNotConfiguredError, VoiceNotConfiguredError

        if not self._tts:
            raise VoiceNotConfiguredError("No TTS provider configured")
        if not self._backend:
            raise VoiceBackendNotConfiguredError("No voice backend configured")

        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        room_id = binding_info[0] if binding_info else None

        from roomkit.telemetry.context import reset_span, set_current_span

        _parent = getattr(self, "_voice_session_spans", {}).get(session.id)
        _tok = set_current_span(_parent) if _parent else None
        try:
            final_text = text

            # Run BEFORE_TTS sync hook (can block or modify text)
            if self._framework and room_id:
                context = await self._framework._build_context(room_id)
                before_result = await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.BEFORE_TTS,
                    text,
                    context,
                    skip_event_filter=True,
                )
                if not before_result.allowed:
                    logger.info("say() blocked by hook: %s", before_result.reason)
                    return
                final_text = before_result.event if isinstance(before_result.event, str) else text

            if self._tts_filter is not None:
                final_text = self._tts_filter(final_text)
                if not final_text:
                    logger.debug("say() text empty after filter — skipping")
                    return

            await self._send_tts(session, final_text, voice=voice)

            # Run AFTER_TTS async hook
            if self._framework and room_id:
                context = await self._framework._build_context(room_id)
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.AFTER_TTS,
                    final_text,
                    context,
                    skip_event_filter=True,
                )

        except (VoiceNotConfiguredError, VoiceBackendNotConfiguredError):
            raise
        except Exception:
            # A failed synthesis was reported as it failed (_send_tts).
            logger.exception("Error in say()")
        finally:
            if _tok is not None:
                reset_span(_tok)

    async def play(
        self,
        session: VoiceSession,
        audio: str | bytes,
        *,
        text: str | None = None,
    ) -> None:
        """Play a WAV file to the participant.

        Args:
            session: The voice session to play into.
            audio: Path to a WAV file (str / pathlib.Path) or raw WAV
                bytes already loaded into memory.
            text: Optional transcript text to send for UI display.

        Raises:
            VoiceBackendNotConfiguredError: If no voice backend is configured.
            ValueError: If the WAV file is not 16-bit PCM mono.
        """
        import io
        import wave

        from roomkit.core.framework import VoiceBackendNotConfiguredError

        from .voice import TTSPlaybackState

        if not self._backend:
            raise VoiceBackendNotConfiguredError("No voice backend configured")

        # Read and validate WAV
        if isinstance(audio, (str, os.PathLike)):
            src: str | io.BytesIO = str(audio)
        else:
            src = io.BytesIO(audio)

        with wave.open(src, "rb") as wf:
            if wf.getsampwidth() != 2:
                raise ValueError(f"WAV must be 16-bit PCM (got {wf.getsampwidth() * 8}-bit)")
            if wf.getnchannels() != 1:
                raise ValueError(f"WAV must be mono (got {wf.getnchannels()} channels)")
            if wf.getcomptype() != "NONE":
                raise ValueError(f"WAV must be uncompressed PCM (got {wf.getcompname()})")
            sample_rate = wf.getframerate()
            pcm_data = wf.readframes(wf.getnframes())

        if text is not None:
            await self._session_output_backend(session).send_transcription(
                session, text, "assistant"
            )

        with self._state_lock:
            self._playing_sessions[session.id] = TTSPlaybackState(
                session_id=session.id,
                text=text or "(audio)",
            )

        try:
            # Wrap PCM as an AudioChunk stream so the outbound pipeline can
            # resample from the WAV sample rate to the backend's codec rate.
            from roomkit.voice.base import AudioChunk as OutChunk

            async def _pcm_stream() -> AsyncIterator[OutChunk]:
                yield OutChunk(
                    data=pcm_data,
                    sample_rate=sample_rate,
                    channels=1,
                    is_final=True,
                )

            audio_stream: AsyncIterator[OutChunk] = _pcm_stream()
            if self._pipeline is not None or getattr(self, "_outbound_audio_taps", []):
                audio_stream = self._wrap_outbound(session, audio_stream)
            await self._session_output_backend(session).send_audio(session, audio_stream)
        finally:
            self._playback_sent(session.id)


def _observe_audio(
    playback: TTSPlaybackState,
    recorder: AssistantTurnRecorder | None,
    chunks: AsyncIterator[AudioChunk],
    *,
    gate: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[AudioChunk]:
    """Measure the audio a synthesis call hands to the transport.

    ``playback`` learns how much audio went out (its ``played_ms``), and the
    turn recorder keeps a copy when the context keeps audio. The measure
    starts here, not at the first chunk: until one goes out, the user has
    heard nothing, however long synthesis takes. ``gate``, when given, is
    awaited before the first chunk goes out; False ends the stream unsaid.
    A chunk that is not 16-bit PCM is refused before any of that.
    """
    playback.start_measuring()
    return _observed_chunks(playback, recorder, _pcm16_only(chunks), gate)


async def _pcm16_only(chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
    """Hand on a TTS stream's chunks, refusing the first one that is not 16-bit PCM.

    The outbound pipeline and every voice backend read a chunk as 16-bit signed
    PCM whatever its ``format``: an MP3 or G.711 chunk would play as noise (RFC
    section 12.2). The refusal comes before a byte reaches either.
    """
    async for chunk in chunks:
        require_pcm16(chunk, "VoiceChannel")
        yield chunk


async def _observed_chunks(
    playback: TTSPlaybackState,
    recorder: AssistantTurnRecorder | None,
    chunks: AsyncIterator[AudioChunk],
    gate: Callable[[], Awaitable[bool]] | None,
) -> AsyncIterator[AudioChunk]:
    async for chunk in chunks:
        if gate is not None:
            if not await gate():
                return  # superseded before a word of it was heard
            gate = None
        playback.note_audio(chunk)
        if recorder is not None:
            recorder.add(chunk)
        yield chunk


async def _close_stream(stream: AsyncIterator[AudioChunk] | None) -> None:
    """Close a provider's audio stream once the transport stops reading it.

    A backend leaves ``send_audio()`` on a cancel without closing the iterator
    it was given; the provider's cleanup (an HTTP response, a GPU thread)
    would otherwise wait for the stream to be garbage collected.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        logger.debug("Closing the TTS stream failed", exc_info=True)


class _ResponseText:
    """The text of a streamed response, as its sessions read it, and how it ended."""

    def __init__(self) -> None:
        self.accumulated: list[str] = []
        self.failure: Exception | None = None

    async def read(self, text_stream: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Each text delta of *text_stream*, kept as it passes.

        A response that fails ends here, its failure kept: the splitter then
        hands on its last partial sentence, so the sessions speak all the
        text the response produced (RFC §12.2 step 15s).
        """
        try:
            async for delta in text_stream:
                if not isinstance(delta, str):
                    continue
                self.accumulated.append(delta)
                yield delta
        except Exception as exc:
            self.failure = exc


def _served_sessions(sessions: list[VoiceSession], results: list[Any]) -> list[VoiceSession]:
    """Sessions a per-session delivery reached; raise when it reached none.

    *results* are ``asyncio.gather(..., return_exceptions=True)`` outcomes:
    an exception is a failed session, ``False`` a session skipped, anything
    else a session served.  One session's transport failure must not take
    the response away from the others, so failures are logged; when every
    session failed, the first error takes the caller's error path, as a
    single session's did.
    """
    served = [
        s
        for s, r in zip(sessions, results, strict=True)
        if r is not False and not isinstance(r, BaseException)
    ]
    failures = [r for r in results if isinstance(r, BaseException)]
    raised = failures[0] if failures and not served else None
    for session, result in zip(sessions, results, strict=True):
        if isinstance(result, BaseException) and result is not raised:
            logger.error("Voice delivery failed for session %s", session.id, exc_info=result)
    if raised is not None:
        raise raised
    return served


async def _filter_sentences_plain(
    source: AsyncIterator[str],
    fn: Any,
) -> AsyncGenerator[str, None]:
    """Wrap an async token stream through a plain ``Callable[[str], str]``."""
    async for chunk in source:
        cleaned = fn(chunk)
        if cleaned:
            yield cleaned
