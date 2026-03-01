"""VoiceChannel mixin — TTS delivery (streaming and non-streaming)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookTrigger
from roomkit.telemetry.base import Attr, SpanKind, TelemetryProvider
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.voice.utils import rms_db

_NOOP = NoopTelemetryProvider()

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.tts.base import TTSProvider

    from .voice import TTSPlaybackState

logger = logging.getLogger("roomkit.voice")

# Time to keep _playing_sessions alive after send_audio() returns.
# Accounts for residual room echo/reverb after the speaker physically
# finishes playing.  During this period, continuous STT discards any
# transcription as echo.
_PLAYBACK_DRAIN_S = 2.0


class VoiceTTSMixin:
    """TTS delivery helpers for VoiceChannel."""

    # -- attributes provided by VoiceChannel.__init__ --
    channel_id: str
    _framework: RoomKit | None
    _tts: TTSProvider | None
    _backend: VoiceBackend | None
    _pipeline: AudioPipeline | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _playing_sessions: dict[str, TTSPlaybackState]
    _playback_done_events: dict[str, asyncio.Event]
    _last_tts_ended_at: dict[str, float]
    _last_output_level_at: dict[str, float]
    _debug_frame_count: int
    _voice_map: dict[str, str]

    # -- methods provided by other mixins / main class (TYPE_CHECKING only) --
    if TYPE_CHECKING:

        def _schedule(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None: ...

        async def _fire_audio_level_hook(
            self,
            session: VoiceSession,
            level_db: float,
            room_id: str,
            trigger: HookTrigger,
        ) -> None: ...

    def _fire_output_level(self, session: VoiceSession, data: bytes) -> None:
        """Fire ON_OUTPUT_AUDIO_LEVEL hook from the outbound pipeline, throttled."""
        now = time.monotonic()
        if now - self._last_output_level_at.get(session.id, 0.0) < 0.1:
            return
        self._last_output_level_at[session.id] = now
        with self._state_lock:  # type: ignore[attr-defined]
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return
        room_id, _ = binding_info
        self._schedule(
            self._fire_audio_level_hook(
                session,
                rms_db(data),
                room_id,
                HookTrigger.ON_OUTPUT_AUDIO_LEVEL,
            ),
            name=f"output_audio_level:{session.id}",
        )

    def _resolve_voice(self, channel_id: str) -> str | None:
        """Look up TTS voice override for *channel_id* via voice_map."""
        return self._voice_map.get(channel_id) if self._voice_map else None

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

    async def _finish_playback(self, session_id: str) -> None:
        """Clear playback state after a post-drain delay for echo decay.

        After ``send_audio()`` returns (speaker buffer drained), the room
        may still have residual echo for 1-2 seconds.  This delay keeps
        ``_playing_sessions`` alive so continuous STT discards any echo
        transcribed during that window.

        AEC is deactivated immediately (before the drain delay) so the
        stale adaptive filter doesn't suppress user speech.  The
        ``_playing_sessions`` flag stays alive during the delay to gate
        echo transcriptions on the STT side.

        If ``interrupt()`` fires during the delay, it pops
        ``_playing_sessions`` immediately — the delayed pop becomes a no-op.
        """
        import time as _time

        # Deactivate AEC immediately — don't wait for drain delay.
        # The adaptive filter is stale and will suppress user speech.
        if self._pipeline is not None and self._pipeline._config.aec is not None:
            aec = self._pipeline._config.aec
            aec.reset()
            aec.set_active(False)

        await asyncio.sleep(_PLAYBACK_DRAIN_S)
        with self._state_lock:  # type: ignore[attr-defined]
            playback = self._playing_sessions.pop(session_id, None)
        if playback:
            self._last_tts_ended_at[session_id] = _time.monotonic()
            logger.debug(
                "Playback drain complete for session %s (delay=%.1fs)",
                session_id,
                _PLAYBACK_DRAIN_S,
            )

    def _find_sessions(self, room_id: str, binding: ChannelBinding) -> list[VoiceSession]:
        """Find voice sessions for a room/binding pair."""
        if not self._backend:
            return []

        with self._state_lock:  # type: ignore[attr-defined]
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
        from roomkit.voice.tts.sentence_splitter import split_sentences

        from .voice import TTSPlaybackState

        if not self._tts or not self._backend:
            return ChannelOutputModel.empty()

        # Defense-in-depth: skip system/internal events (primary guard is in deliver())
        from roomkit.models.enums import EventType

        if event.type == EventType.SYSTEM or event.visibility == "internal":
            return ChannelOutputModel.empty()

        tts_name = self._tts.name  # capture before async yields (may become None)
        room_id = event.room_id
        target_sessions = self._find_sessions(room_id, binding)

        accumulated: list[str] = []

        async def tracking_stream() -> AsyncIterator[str]:
            async for delta in text_stream:
                accumulated.append(delta)
                yield delta

        import time as _time

        _t = getattr(self._framework, "_telemetry", None) if self._framework else None
        telemetry: TelemetryProvider | None = _t if isinstance(_t, TelemetryProvider) else None

        # Capture parent span BEFORE playback — session may be unbound during TTS.
        _vs_parent = getattr(self, "_voice_session_spans", {}).get(
            target_sessions[0].id if target_sessions else ""
        )

        for session in target_sessions:
            # Cancel any existing TTS to prevent overlapping audio
            with self._state_lock:  # type: ignore[attr-defined]
                existing = self._playing_sessions.get(session.id)
            if existing:
                logger.info(
                    "Cancelling previous TTS for session %s before starting new one",
                    session.id,
                )
                await self.interrupt(session, reason="new_tts")  # type: ignore[attr-defined]

            with self._state_lock:  # type: ignore[attr-defined]
                self._playing_sessions[session.id] = TTSPlaybackState(
                    session_id=session.id, text="(streaming)"
                )
                # Clear done event so wait_playback_done() blocks until send_audio returns
                done_ev = self._playback_done_events.get(session.id)
                if done_ev is None:
                    done_ev = asyncio.Event()
                    self._playback_done_events[session.id] = done_ev
                else:
                    done_ev.clear()
            # Activate AEC so echo cancellation runs during playback
            if self._pipeline is not None and self._pipeline._config.aec is not None:
                aec = self._pipeline._config.aec
                aec.set_active(True)
            t0 = _time.monotonic()
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

            try:
                voice = self._resolve_voice(event.source.channel_id)
                sentences = split_sentences(tracking_stream())

                # Relay each sentence to the client before TTS synthesis.
                async def relay_sentences(
                    source: AsyncIterator[str],
                    _session: VoiceSession = session,
                ) -> AsyncIterator[str]:
                    async for sentence in source:
                        await self._backend.send_transcription(  # type: ignore[union-attr]
                            _session, sentence, "assistant_interim"
                        )
                        yield sentence

                audio = self._tts.synthesize_stream_input(relay_sentences(sentences), voice=voice)
                if self._pipeline is not None:
                    audio = self._wrap_outbound(session, audio)
                await self._backend.send_audio(session, audio)
            except Exception:
                if telemetry is not None and span_id is not None:
                    telemetry.end_span(span_id, status="error", error_message="stream TTS failed")
                    span_id = None
                raise
            finally:
                duration_ms = (_time.monotonic() - t0) * 1000
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
                    _time.monotonic() - t0,
                )
                self._debug_frame_count = 0  # reset RMS debug counter
                # Signal that send_audio() has returned so
                # wait_playback_done() can unblock immediately.
                done_ev = self._playback_done_events.get(session.id)
                if done_ev is not None:
                    done_ev.set()
                # Keep _playing_sessions alive during post-drain echo decay.
                # _finish_playback pops it after the delay (interrupt() pops
                # immediately if barge-in fires first).
                self._schedule(
                    self._finish_playback(session.id),
                    name=f"finish_playback:{session.id}",
                )

        full_text = "".join(accumulated)
        # Update playback state with actual streamed text (was "(streaming)")
        for session in target_sessions:
            with self._state_lock:  # type: ignore[attr-defined]
                if session.id in self._playing_sessions:
                    self._playing_sessions[session.id] = TTSPlaybackState(
                        session_id=session.id, text=full_text or "(empty)"
                    )
        if full_text:
            for session in target_sessions:
                await self._backend.send_transcription(session, full_text, "assistant")

        # Fire AFTER_TTS hooks (BEFORE_TTS skipped — can't block mid-stream)
        if self._framework and full_text:
            from roomkit.telemetry.context import reset_span, set_current_span

            _tok = set_current_span(_vs_parent) if _vs_parent else None
            try:
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.AFTER_TTS,
                    full_text,
                    context,
                    skip_event_filter=True,
                )
            finally:
                if _tok is not None:
                    reset_span(_tok)

        return ChannelOutputModel.empty()

    async def _send_tts(
        self, session: VoiceSession, text: str, *, voice: str | None = None
    ) -> None:
        """Synthesize *text* and send audio to *session*.

        Handles transcription, playback state tracking, streaming synthesis
        with pipeline wrapping, and fallback to batch synthesis.
        """
        import time as _time

        from .voice import TTSPlaybackState

        assert self._tts is not None  # caller must guard  # noqa: S101
        assert self._backend is not None  # noqa: S101

        tts_name = self._tts.name  # capture before async yields (may become None)

        # Cancel any existing TTS to prevent overlapping audio
        with self._state_lock:  # type: ignore[attr-defined]
            existing = self._playing_sessions.get(session.id)
        if existing:
            logger.info(
                "Cancelling previous TTS for session %s before starting new one",
                session.id,
            )
            await self.interrupt(session, reason="new_tts")  # type: ignore[attr-defined]

        await self._backend.send_transcription(session, text, "assistant")

        with self._state_lock:  # type: ignore[attr-defined]
            self._playing_sessions[session.id] = TTSPlaybackState(
                session_id=session.id,
                text=text,
            )
            # Clear done event so wait_playback_done() blocks until send_audio returns
            done_ev = self._playback_done_events.get(session.id)
            if done_ev is None:
                done_ev = asyncio.Event()
                self._playback_done_events[session.id] = done_ev
            else:
                done_ev.clear()
        # Activate AEC so echo cancellation runs during playback
        if self._pipeline is not None and self._pipeline._config.aec is not None:
            aec = self._pipeline._config.aec
            aec.set_active(True)

        # Resolve telemetry provider
        _t = getattr(self._framework, "_telemetry", None) if self._framework else None
        telemetry: TelemetryProvider | None = _t if isinstance(_t, TelemetryProvider) else None

        with self._state_lock:  # type: ignore[attr-defined]
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

        t0 = _time.monotonic()
        try:
            audio_stream = self._tts.synthesize_stream(text, voice=voice)
            if self._pipeline is not None:
                audio_stream = self._wrap_outbound(session, audio_stream)
            await self._backend.send_audio(session, audio_stream)
        except NotImplementedError:
            logger.error(
                "TTS provider %s does not support streaming synthesis; "
                "voice channels require synthesize_stream(). No audio sent.",
                tts_name,
            )
        except Exception:
            if telemetry is not None and span_id is not None:
                telemetry.end_span(span_id, status="error", error_message="TTS failed")
                span_id = None  # prevent double-end
            raise
        finally:
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
            self._schedule(
                self._finish_playback(session.id),
                name=f"finish_playback:{session.id}",
            )

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

            logger.info("AI response: %s", final_text)

            target_sessions = self._find_sessions(room_id, binding)

            # Capture parent span BEFORE _send_tts — session may be unbound
            # during playback, removing it from _voice_session_spans.
            from roomkit.telemetry.context import reset_span, set_current_span

            _parent = getattr(self, "_voice_session_spans", {}).get(
                target_sessions[0].id if target_sessions else ""
            )

            voice = self._resolve_voice(event.source.channel_id)
            for session in target_sessions:
                await self._send_tts(session, final_text, voice=voice)

            _tok = set_current_span(_parent) if _parent else None
            try:
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.AFTER_TTS,
                    final_text,
                    context,
                    skip_event_filter=True,
                )
            finally:
                if _tok is not None:
                    reset_span(_tok)

        except Exception as exc:
            logger.exception("Error delivering voice audio")
            try:
                await self._framework._emit_framework_event(
                    "tts_error",
                    room_id=room_id,
                    data={
                        "provider": self._tts.name if self._tts else "unknown",
                        "error": str(exc),
                    },
                )
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

        with self._state_lock:  # type: ignore[attr-defined]
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
        except Exception as exc:
            logger.exception("Error in say()")
            if self._framework and room_id:
                try:
                    await self._framework._emit_framework_event(
                        "tts_error",
                        room_id=room_id,
                        data={
                            "provider": self._tts.name if self._tts else "unknown",
                            "error": str(exc),
                        },
                    )
                except Exception:
                    logger.exception("Error emitting tts_error")
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
            await self._backend.send_transcription(session, text, "assistant")

        with self._state_lock:  # type: ignore[attr-defined]
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
            if self._pipeline is not None:
                audio_stream = self._wrap_outbound(session, audio_stream)
            await self._backend.send_audio(session, audio_stream)
        finally:
            self._schedule(
                self._finish_playback(session.id),
                name=f"finish_playback:{session.id}",
            )
