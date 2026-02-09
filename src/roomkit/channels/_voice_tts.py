"""VoiceChannel mixin — TTS delivery (streaming and non-streaming)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger

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
    _last_interrupt_at: dict[str, float]
    _debug_frame_count: int

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
            logger.debug("Streaming TTS playback started for session %s", session.id)
            try:
                sentences = split_sentences(tracking_stream())
                audio = self._tts.synthesize_stream_input(sentences)
                if self._pipeline is not None:
                    audio = self._wrap_outbound(session, audio)
                await self._backend.send_audio(session, audio)
            finally:
                self._playing_sessions.pop(session.id, None)
                self._last_interrupt_at[session.id] = _time.monotonic()
                logger.debug(
                    "Streaming TTS playback ended for session %s (%.1fs)",
                    session.id,
                    _time.monotonic() - t0,
                )
                self._debug_frame_count = 0  # reset RMS debug counter

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

    async def _deliver_voice(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> None:
        if not self._tts or not self._backend or not self._framework:
            return

        from roomkit.models.event import TextContent

        from .voice import TTSPlaybackState

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
                await self._backend.send_transcription(session, final_text, "assistant")

                self._playing_sessions[session.id] = TTSPlaybackState(
                    session_id=session.id,
                    text=final_text,
                )

                try:
                    audio_stream = self._tts.synthesize_stream(final_text)
                    if self._pipeline is not None:
                        audio_stream = self._wrap_outbound(session, audio_stream)
                    await self._backend.send_audio(session, audio_stream)
                except NotImplementedError:
                    await self._tts.synthesize(final_text)
                    logger.warning("TTS provider %s doesn't support streaming", self._tts.name)
                finally:
                    import time as _time

                    self._playing_sessions.pop(session.id, None)
                    self._last_interrupt_at[session.id] = _time.monotonic()

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
