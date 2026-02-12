"""VoiceChannel mixin — TTS delivery (streaming and non-streaming)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookTrigger
from roomkit.voice.utils import rms_db

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
    _last_tts_ended_at: dict[str, float]
    _last_output_level_at: float
    _debug_frame_count: int

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
        if now - self._last_output_level_at < 0.1:
            return
        self._last_output_level_at = now
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

        If ``interrupt()`` fires during the delay, it pops
        ``_playing_sessions`` immediately — the delayed pop becomes a no-op.
        """
        import time as _time

        await asyncio.sleep(_PLAYBACK_DRAIN_S)
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

        target_sessions: list[VoiceSession] = []
        for session_id, (bound_room_id, bound_binding) in self._session_bindings.items():
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

        room_id = event.room_id
        target_sessions = self._find_sessions(room_id, binding)

        accumulated: list[str] = []

        async def tracking_stream() -> AsyncIterator[str]:
            async for delta in text_stream:
                accumulated.append(delta)
                yield delta

        import time as _time

        for session in target_sessions:
            self._playing_sessions[session.id] = TTSPlaybackState(
                session_id=session.id, text="(streaming)"
            )
            t0 = _time.monotonic()
            logger.info("Streaming TTS playback started for session %s", session.id)
            try:
                sentences = split_sentences(tracking_stream())
                audio = self._tts.synthesize_stream_input(sentences)
                if self._pipeline is not None:
                    audio = self._wrap_outbound(session, audio)
                await self._backend.send_audio(session, audio)
            finally:
                logger.debug(
                    "Streaming TTS send_audio returned for session %s (%.1fs), draining",
                    session.id,
                    _time.monotonic() - t0,
                )
                self._debug_frame_count = 0  # reset RMS debug counter
                # Keep _playing_sessions alive during post-drain echo decay.
                # _finish_playback pops it after the delay (interrupt() pops
                # immediately if barge-in fires first).
                self._schedule(
                    self._finish_playback(session.id),
                    name=f"finish_playback:{session.id}",
                )

        full_text = "".join(accumulated)
        if full_text:
            for session in target_sessions:
                await self._backend.send_transcription(session, full_text, "assistant")

        # Fire AFTER_TTS hooks (BEFORE_TTS skipped — can't block mid-stream)
        if self._framework and full_text:
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.AFTER_TTS,
                full_text,
                context,
                skip_event_filter=True,
            )

        return ChannelOutputModel.empty()

    async def _send_tts(
        self, session: VoiceSession, text: str, *, voice: str | None = None
    ) -> None:
        """Synthesize *text* and send audio to *session*.

        Handles transcription, playback state tracking, streaming synthesis
        with pipeline wrapping, and fallback to batch synthesis.
        """
        from .voice import TTSPlaybackState

        assert self._tts is not None  # caller must guard  # noqa: S101
        assert self._backend is not None  # noqa: S101

        await self._backend.send_transcription(session, text, "assistant")

        self._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text=text,
        )

        try:
            audio_stream = self._tts.synthesize_stream(text, voice=voice)
            if self._pipeline is not None:
                audio_stream = self._wrap_outbound(session, audio_stream)
            await self._backend.send_audio(session, audio_stream)
        except NotImplementedError:
            await self._tts.synthesize(text, voice=voice)
            logger.warning("TTS provider %s doesn't support streaming", self._tts.name)
        finally:
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

            for session in target_sessions:
                await self._send_tts(session, final_text)

            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.AFTER_TTS,
                final_text,
                context,
                skip_event_filter=True,
            )

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

        binding_info = self._session_bindings.get(session.id)
        room_id = binding_info[0] if binding_info else None

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

    async def play(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
        *,
        text: str | None = None,
    ) -> None:
        """Play pre-rendered audio to the participant.

        Args:
            session: The voice session to play into.
            audio: Raw bytes or an async iterator of AudioChunk.
            text: Optional transcript text to send for UI display.

        Raises:
            VoiceBackendNotConfiguredError: If no voice backend is configured.
        """
        from roomkit.core.framework import VoiceBackendNotConfiguredError

        from .voice import TTSPlaybackState

        if not self._backend:
            raise VoiceBackendNotConfiguredError("No voice backend configured")

        if text is not None:
            await self._backend.send_transcription(session, text, "assistant")

        self._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text=text or "(audio)",
        )

        try:
            if not isinstance(audio, bytes) and self._pipeline is not None:
                audio = self._wrap_outbound(session, audio)
            await self._backend.send_audio(session, audio)
        finally:
            self._schedule(
                self._finish_playback(session.id),
                name=f"finish_playback:{session.id}",
            )
