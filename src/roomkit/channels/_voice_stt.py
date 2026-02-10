"""VoiceChannel mixin — STT streaming and speech processing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.pipeline.turn.base import TurnEntry
    from roomkit.voice.stt.base import STTProvider

    from .voice import TTSPlaybackState, _STTStreamState

logger = logging.getLogger("roomkit.voice")


class VoiceSTTMixin:
    """STT streaming and speech-processing helpers for VoiceChannel."""

    # -- attributes provided by VoiceChannel.__init__ --
    channel_id: str
    _framework: RoomKit | None
    _stt: STTProvider | None
    _backend: VoiceBackend | None
    _pipeline: AudioPipeline | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _playing_sessions: dict[str, TTSPlaybackState]
    _last_tts_ended_at: dict[str, float]
    _stt_streams: dict[str, _STTStreamState]
    _continuous_stt: bool
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_audio: dict[str, bytearray]
    _debug_frame_count: int

    # -- methods provided by other mixins / main class (TYPE_CHECKING only) --
    if TYPE_CHECKING:

        def _schedule(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None: ...
        async def _fire_partial_transcription_hook(
            self, session: VoiceSession, result: Any, room_id: str
        ) -> None: ...
        async def _handle_barge_in(
            self, session: VoiceSession, playback: TTSPlaybackState, room_id: str
        ) -> None: ...
        async def _evaluate_turn(
            self,
            session: VoiceSession,
            text: str,
            room_id: str,
            context: RoomContext,
            *,
            audio_bytes: bytes | None = None,
        ) -> None: ...
        async def _route_text(self, session: VoiceSession, text: str, room_id: str) -> None: ...

    # -----------------------------------------------------------------
    # VAD-driven STT streaming
    # -----------------------------------------------------------------

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
            logger.warning("STT stream queue full for session %s, dropping chunk", session_id)

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

            sample_rate = session.metadata.get("input_sample_rate", 16000)
            queue.put_nowait(OutChunk(data=pre_roll, sample_rate=sample_rate))

        async def audio_gen() -> AsyncIterator[AudioChunk]:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                yield chunk

        async def consume(state: _STTStreamState) -> None:
            try:
                assert self._stt is not None
                async for result in self._stt.transcribe_stream(audio_gen()):
                    if state.cancelled:
                        return
                    if result.is_final and result.text:
                        state.final_text = result.text
                    elif not result.is_final and result.text:
                        state.partial_text = result.text
                        self._schedule(
                            self._fire_partial_transcription_hook(session, result, room_id),
                            name=f"partial_stt:{session.id}",
                        )
            except Exception:
                logger.exception("STT stream error for session %s", session.id)
                state.error = True

        state = _STTStreamState(queue=queue)
        self._stt_streams[session.id] = state
        try:
            loop = asyncio.get_running_loop()
            state.task = loop.create_task(consume(state), name=f"stt_stream:{session.id}")
        except RuntimeError:
            self._stt_streams.pop(session.id, None)

    def _cancel_stt_stream(self, session_id: str) -> None:
        """Cancel an active streaming STT session."""
        state = self._stt_streams.pop(session_id, None)
        if state is None:
            return
        state.cancelled = True
        import contextlib

        with contextlib.suppress(asyncio.QueueFull):
            state.queue.put_nowait(None)
        if state.task is not None:
            state.task.cancel()

    # -----------------------------------------------------------------
    # Continuous STT (no local VAD — provider handles endpointing)
    # -----------------------------------------------------------------

    def _on_processed_frame_for_stt(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Feed every processed frame to the continuous STT stream.

        In continuous mode the STT provider handles endpointing
        server-side (e.g. Gradium ``end_text`` events), so we just
        buffer and forward audio without any local silence detection.
        """
        from .voice import _STT_STREAM_BUFFER_BYTES

        stream_state = self._stt_streams.get(session.id)
        if stream_state is None or stream_state.cancelled:
            return

        # Debug: log RMS energy during playback (sampled every ~1s)
        if logger.isEnabledFor(logging.DEBUG) and self._playing_sessions.get(session.id):
            import struct

            samples = struct.unpack(f"<{len(frame.data) // 2}h", frame.data)
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            self._debug_frame_count = getattr(self, "_debug_frame_count", 0) + 1
            if self._debug_frame_count % 50 == 0:  # ~1s at 20ms frames
                logger.debug(
                    "STT feed during playback: rms=%.0f, frames=%d",
                    rms,
                    self._debug_frame_count,
                )

        stream_state.frame_buffer.extend(frame.data)
        stream_state.frame_buffer_rate = frame.sample_rate
        if len(stream_state.frame_buffer) >= _STT_STREAM_BUFFER_BYTES:
            self._flush_stt_buffer(stream_state, session.id)

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

        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._stt:
            return
        room_id, _ = binding_info
        logger.info("Starting continuous STT for session %s", session.id)

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=500)
        state = _STTStreamState(queue=queue)
        self._stt_streams[session.id] = state

        async def run_continuous(state: _STTStreamState) -> None:
            import time as _time

            backoff = 1.0
            while not state.cancelled:
                # Fresh queue + WebSocket per turn (avoids server-side overlap).
                # Keep frame_buffer intact — audio arriving during the
                # reconnection gap (sleep + WebSocket connect) is preserved
                # and will be flushed to the new queue on the next frame.
                state.queue = asyncio.Queue[Any](maxsize=500)
                cur_queue = state.queue

                async def audio_gen(
                    q: asyncio.Queue[Any] = cur_queue,
                ) -> AsyncIterator[AudioChunk]:
                    from .voice import _STT_INACTIVITY_TIMEOUT_S

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
                    assert self._stt is not None
                    barge_in_fired = False
                    backoff = 1.0
                    last_tts = self._last_tts_ended_at.get(session.id, 0.0)
                    since_tts = _time.monotonic() - last_tts if last_tts else -1.0
                    logger.info(
                        "Continuous STT stream cycle starting for %s (since_tts=%.1fs)",
                        session.id,
                        since_tts,
                    )
                    async for result in self._stt.transcribe_stream(audio_gen()):
                        if state.cancelled:
                            break
                        playing = session.id in self._playing_sessions
                        if result.is_final and result.text:
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
                            playback = self._playing_sessions.get(session.id)
                            if playback and not barge_in_fired:
                                logger.info(
                                    "Discarding echo transcription during playback: %r",
                                    result.text,
                                )
                                import contextlib

                                with contextlib.suppress(asyncio.QueueFull):
                                    cur_queue.put_nowait(None)
                                break

                            barge_in_fired = False
                            # Signal audio gen to stop so the SDK's
                            # sender task unblocks and the stream
                            # closes cleanly (no timeout needed).
                            import contextlib

                            with contextlib.suppress(asyncio.QueueFull):
                                cur_queue.put_nowait(None)
                            # Provider signals turn complete — route to AI
                            self._schedule(
                                self._handle_continuous_transcription(
                                    session, result.text, room_id
                                ),
                                name=f"continuous_stt:{session.id}",
                            )
                        elif not result.is_final and result.text:
                            # Barge-in: user speaking during TTS playback
                            if not barge_in_fired:
                                playback = self._playing_sessions.get(session.id)
                                if playback:
                                    from roomkit.voice.interruption import InterruptionHandler

                                    handler: InterruptionHandler = self._interruption_handler  # type: ignore[attr-defined]
                                    decision = handler.evaluate(
                                        playback_position_ms=(playback.position_ms),
                                        speech_duration_ms=0,
                                    )
                                    logger.info(
                                        "Barge-in eval: partial=%r pos=%dms interrupt=%s",
                                        result.text,
                                        playback.position_ms,
                                        decision.should_interrupt,
                                    )
                                    if decision.should_interrupt:
                                        barge_in_fired = True
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
        except RuntimeError:
            self._stt_streams.pop(session.id, None)

    def _stop_continuous_stt(self, session_id: str) -> None:
        """Stop continuous STT for a session."""
        self._cancel_stt_stream(session_id)

    async def _handle_continuous_transcription(
        self, session: VoiceSession, text: str, room_id: str
    ) -> None:
        """Process a transcription result from continuous STT."""
        if not self._framework or not text.strip():
            return
        try:
            import time as _time

            playback = self._playing_sessions.get(session.id)
            last_tts_end = self._last_tts_ended_at.get(session.id, 0.0)
            since_tts = _time.monotonic() - last_tts_end if last_tts_end else -1.0

            if playback:
                logger.warning(
                    "Discarding echo during playback: %r (pos=%dms)",
                    text,
                    playback.position_ms,
                )
                return

            logger.info(
                "Transcription: %s (since_tts_end=%.1fs)",
                text,
                since_tts,
            )

            if self._backend:
                await self._backend.send_transcription(session, text, "user")

            context = await self._framework._build_context(room_id)

            # Fire ON_SPEECH_END hooks (continuous mode synthesises speech events)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SPEECH_END,
                session,
                context,
                skip_event_filter=True,
            )

            # Fire ON_TRANSCRIPTION hooks
            transcription_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.ON_TRANSCRIPTION,
                text,
                context,
                skip_event_filter=True,
            )

            if not transcription_result.allowed:
                logger.info("Transcription blocked by hook: %s", transcription_result.reason)
                return

            final_text = (
                transcription_result.event if isinstance(transcription_result.event, str) else text
            )

            turn_detector = self._pipeline_config.turn_detector if self._pipeline_config else None
            if turn_detector is not None:
                await self._evaluate_turn(session, final_text, room_id, context)
            else:
                await self._route_text(session, final_text, room_id)

        except Exception:
            logger.exception("Error processing continuous STT transcription")

    # -----------------------------------------------------------------
    # Speech-end processing (VAD mode)
    # -----------------------------------------------------------------

    async def _process_speech_end(
        self,
        session: VoiceSession,
        audio: bytes,
        room_id: str,
        stream_state: _STTStreamState | None = None,
    ) -> None:
        """Process speech end: fire hooks, transcribe, route inbound.

        ON_SPEECH_END hooks are fired here (not in _on_pipeline_vad_event)
        to guarantee ordering: ON_SPEECH_END always fires before
        ON_TRANSCRIPTION and before routing to the AI.

        Args:
            stream_state: The STT stream state popped by the caller
                (_on_pipeline_speech_end) so it is immune to a rapid
                SPEECH_START overwriting _stt_streams[session.id].
        """
        if not self._framework:
            return

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

            # Get audio parameters from session metadata (set by backend)
            sample_rate = session.metadata.get("input_sample_rate", 16000)

            # Try to collect streaming STT result; fall back to batch
            text: str | None = None
            if stream_state is not None and not stream_state.error and not stream_state.cancelled:
                try:
                    if stream_state.task is not None:
                        await asyncio.wait_for(stream_state.task, timeout=5.0)
                    text = stream_state.final_text or stream_state.partial_text
                    if text:
                        logger.debug("STT stream result for %s: %s", session.id, text)
                    else:
                        logger.debug(
                            "STT stream returned no text for %s, falling back to batch",
                            session.id,
                        )
                except (TimeoutError, asyncio.CancelledError):
                    logger.warning(
                        "STT stream timeout/cancelled for %s, falling back to batch",
                        session.id,
                    )
                    if stream_state.task is not None:
                        stream_state.task.cancel()
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
                    logger.info("Falling back to batch STT for %s", session.id)
                audio_frame = AudioFrame(
                    data=audio,
                    sample_rate=sample_rate,
                    channels=1,
                    sample_width=2,
                )
                stt_result = await self._stt.transcribe(audio_frame)
                text = stt_result.text

            if not text.strip():
                logger.debug("Empty transcription, skipping")
                return

            logger.info("Transcription: %s", text)

            # Send transcription to client UI (if backend supports it)
            if self._backend:
                await self._backend.send_transcription(session, text, "user")

            # Fire ON_TRANSCRIPTION hooks (sync, can modify)
            transcription_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.ON_TRANSCRIPTION,
                text,
                context,
                skip_event_filter=True,
            )

            if not transcription_result.allowed:
                logger.info("Transcription blocked by hook: %s", transcription_result.reason)
                return

            # Use potentially modified text
            final_text = (
                transcription_result.event if isinstance(transcription_result.event, str) else text
            )

            # Turn detection: if configured, evaluate before routing
            turn_detector = self._pipeline_config.turn_detector if self._pipeline_config else None
            if turn_detector is not None:
                await self._evaluate_turn(session, final_text, room_id, context, audio_bytes=audio)
            else:
                # No turn detector — route immediately
                await self._route_text(session, final_text, room_id)

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
