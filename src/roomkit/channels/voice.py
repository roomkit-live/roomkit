"""Voice channel for real-time audio communication."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    HookTrigger,
)
from roomkit.voice.base import VoiceCapability

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.delivery import InboundMessage
    from roomkit.models.event import RoomEvent
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.diarization_provider import DiarizationResult
    from roomkit.voice.pipeline.vad_provider import VADEvent
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.voice")


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass
class TTSPlaybackState:
    """Track ongoing TTS playback for barge-in detection."""

    session_id: str
    text: str
    started_at: datetime = field(default_factory=_utcnow)
    total_duration_ms: int | None = None

    @property
    def position_ms(self) -> int:
        """Estimate current playback position based on elapsed time."""
        elapsed = datetime.now(UTC) - self.started_at
        return int(elapsed.total_seconds() * 1000)


class VoiceChannel(Channel):
    """Real-time voice communication channel.

    Supports two modes:
    - **Streaming mode** (default): Audio is streamed directly via VoiceBackend.
      When a backend is configured, deliver() streams TTS audio to the session.
    - **Store-and-forward mode**: Audio is synthesized and stored for later retrieval.
      Requires a MediaStore for URL generation (not yet implemented).

    When a VoiceBackend and AudioPipelineConfig are configured, the channel:
    - Registers for raw audio frames from the backend via on_audio_received
    - Routes frames through the AudioPipeline (denoiser -> VAD -> diarization)
    - Fires hooks based on pipeline events (speech start/end, silence, speaker change)
    - Transcribes speech using the STT provider
    - Synthesizes AI responses using TTS and streams to the client

    When no pipeline is configured, the channel operates without VAD — the backend
    must handle speech detection externally.
    """

    channel_type = ChannelType.VOICE
    category = ChannelCategory.TRANSPORT
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        *,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        backend: VoiceBackend | None = None,
        pipeline: AudioPipelineConfig | None = None,
        streaming: bool = True,
        enable_barge_in: bool = True,
        barge_in_threshold_ms: int = 200,
    ) -> None:
        super().__init__(channel_id)
        self._stt = stt
        self._tts = tts
        self._backend = backend
        self._pipeline_config = pipeline
        self._streaming = streaming
        self._enable_barge_in = enable_barge_in
        self._barge_in_threshold_ms = barge_in_threshold_ms
        self._framework: RoomKit | None = None
        # Map session_id -> (room_id, binding) for routing
        self._session_bindings: dict[str, tuple[str, ChannelBinding]] = {}
        # Track TTS playback for barge-in detection
        self._playing_sessions: dict[str, TTSPlaybackState] = {}
        # The instantiated pipeline engine (if config provided)
        self._pipeline: Any = None  # AudioPipeline | None

        # Wire up pipeline if both backend and pipeline config are provided
        if backend and pipeline:
            self._setup_pipeline(backend, pipeline)
        elif backend and VoiceCapability.BARGE_IN in backend.capabilities:
            backend.on_barge_in(self._on_backend_barge_in)

    def _setup_pipeline(self, backend: VoiceBackend, config: AudioPipelineConfig) -> None:
        """Create AudioPipeline and wire backend -> pipeline -> callbacks."""
        from roomkit.voice.pipeline.engine import AudioPipeline

        self._pipeline = AudioPipeline(config)

        # Backend delivers raw audio -> pipeline processes it
        backend.on_audio_received(self._on_audio_received)

        # Pipeline events -> VoiceChannel hooks
        self._pipeline.on_speech_end(self._on_pipeline_speech_end)
        self._pipeline.on_vad_event(self._on_pipeline_vad_event)
        if config.diarization is not None:
            self._pipeline.on_speaker_change(self._on_pipeline_speaker_change)

        # Barge-in from backend (transport-level)
        if VoiceCapability.BARGE_IN in backend.capabilities:
            backend.on_barge_in(self._on_backend_barge_in)

    def _on_audio_received(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Handle raw audio frame from backend — feed into pipeline."""
        if self._pipeline is not None:
            self._pipeline.process_frame(session, frame)

    def _on_pipeline_speech_end(self, session: VoiceSession, audio: bytes) -> None:
        """Handle speech end from pipeline — fire hooks and transcribe."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        self._schedule(
            self._process_speech_end(session, audio, room_id),
            name=f"speech_end:{session.id}",
        )

    def _on_pipeline_vad_event(self, session: VoiceSession, vad_event: VADEvent) -> None:
        """Handle VAD events from pipeline — fire corresponding hooks."""
        from roomkit.voice.pipeline.vad_provider import VADEventType

        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        if vad_event.type == VADEventType.SPEECH_START:
            # Check for barge-in
            playback = self._playing_sessions.get(session.id)
            if (
                self._enable_barge_in
                and playback
                and playback.position_ms >= self._barge_in_threshold_ms
            ):
                self._schedule(
                    self._handle_barge_in(session, playback, room_id),
                    name=f"barge_in:{session.id}",
                )

            self._schedule(
                self._fire_speech_start_hooks(session, room_id),
                name=f"speech_start:{session.id}",
            )
        elif vad_event.type == VADEventType.SPEECH_END:
            self._schedule(
                self._fire_speech_end_hooks(session, room_id),
                name=f"speech_end_hook:{session.id}",
            )
        elif vad_event.type == VADEventType.SILENCE:
            self._schedule(
                self._fire_vad_silence_hook(
                    session, int(vad_event.duration_ms or 0), room_id
                ),
                name=f"vad_silence:{session.id}",
            )
        elif vad_event.type == VADEventType.AUDIO_LEVEL:
            self._schedule(
                self._fire_vad_audio_level_hook(
                    session,
                    vad_event.level_db or 0.0,
                    vad_event.confidence is not None and vad_event.confidence > 0.5,
                    room_id,
                ),
                name=f"vad_audio_level:{session.id}",
            )

    def _on_pipeline_speaker_change(
        self, session: VoiceSession, result: DiarizationResult
    ) -> None:
        """Handle speaker change from pipeline — fire ON_SPEAKER_CHANGE hook."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        self._schedule(
            self._fire_speaker_change_hook(session, result, room_id),
            name=f"speaker_change:{session.id}",
        )

    def _schedule(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        """Schedule *coro* as a fire-and-forget task if an event loop is running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(coro, name=name)

    def set_framework(self, framework: RoomKit) -> None:
        """Set the framework reference for inbound routing.

        Called automatically when the channel is registered with RoomKit.
        """
        self._framework = framework

    def bind_session(self, session: VoiceSession, room_id: str, binding: ChannelBinding) -> None:
        """Bind a voice session to a room for message routing."""
        self._session_bindings[session.id] = (room_id, binding)

    def unbind_session(self, session: VoiceSession) -> None:
        """Remove session binding."""
        self._session_bindings.pop(session.id, None)

    @property
    def backend(self) -> VoiceBackend | None:
        """The voice backend (if configured)."""
        return self._backend

    @property
    def info(self) -> dict[str, Any]:
        return {
            "stt": self._stt.name if self._stt else None,
            "tts": self._tts.name if self._tts else None,
            "backend": self._backend.name if self._backend else None,
            "streaming": self._streaming,
            "pipeline": self._pipeline_config is not None,
        }

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.TEXT],
            supports_audio=True,
            supported_audio_formats=["wav", "mp3", "ogg", "webm"],
            max_audio_duration_seconds=3600,
        )

    # -------------------------------------------------------------------------
    # Hook firing helpers
    # -------------------------------------------------------------------------

    async def _fire_speech_start_hooks(self, session: VoiceSession, room_id: str) -> None:
        if not self._framework:
            return
        try:
            context = await self._framework._build_context(room_id)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SPEECH_START,
                session,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_SPEECH_START hooks")

    async def _fire_speech_end_hooks(self, session: VoiceSession, room_id: str) -> None:
        if not self._framework:
            return
        try:
            context = await self._framework._build_context(room_id)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SPEECH_END,
                session,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_SPEECH_END hooks")

    async def _fire_vad_silence_hook(
        self, session: VoiceSession, silence_duration_ms: int, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import VADSilenceEvent

            context = await self._framework._build_context(room_id)
            event = VADSilenceEvent(
                session=session,
                silence_duration_ms=silence_duration_ms,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_VAD_SILENCE,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_VAD_SILENCE hook")

    async def _fire_vad_audio_level_hook(
        self, session: VoiceSession, level_db: float, is_speech: bool, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import VADAudioLevelEvent

            context = await self._framework._build_context(room_id)
            event = VADAudioLevelEvent(
                session=session,
                level_db=level_db,
                is_speech=is_speech,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_VAD_AUDIO_LEVEL,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_VAD_AUDIO_LEVEL hook")

    async def _fire_speaker_change_hook(
        self, session: VoiceSession, result: DiarizationResult, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import SpeakerChangeEvent

            context = await self._framework._build_context(room_id)
            event = SpeakerChangeEvent(
                session=session,
                speaker_id=result.speaker_id,
                confidence=result.confidence,
                is_new_speaker=result.is_new_speaker,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SPEAKER_CHANGE,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_SPEAKER_CHANGE hook")

    # -------------------------------------------------------------------------
    # Barge-in handling
    # -------------------------------------------------------------------------

    async def _handle_barge_in(
        self, session: VoiceSession, playback: TTSPlaybackState, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import BargeInEvent

            context = await self._framework._build_context(room_id)
            event = BargeInEvent(
                session=session,
                interrupted_text=playback.text,
                audio_position_ms=playback.position_ms,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_BARGE_IN,
                event,
                context,
                skip_event_filter=True,
            )
            await self.interrupt(session, reason="barge_in")
        except Exception:
            logger.exception("Error handling barge-in for session %s", session.id)

    async def interrupt(self, session: VoiceSession, *, reason: str = "explicit") -> bool:
        """Interrupt ongoing TTS playback for a session."""
        playback = self._playing_sessions.pop(session.id, None)
        if not playback:
            return False

        if self._backend and VoiceCapability.INTERRUPTION in self._backend.capabilities:
            await self._backend.cancel_audio(session)

        if self._framework:
            binding_info = self._session_bindings.get(session.id)
            if binding_info:
                room_id, _ = binding_info
                try:
                    from roomkit.voice.events import TTSCancelledEvent

                    context = await self._framework._build_context(room_id)
                    event = TTSCancelledEvent(
                        session=session,
                        reason=reason,  # type: ignore[arg-type]
                        text=playback.text,
                        audio_position_ms=playback.position_ms,
                    )
                    await self._framework.hook_engine.run_async_hooks(
                        room_id,
                        HookTrigger.ON_TTS_CANCELLED,
                        event,
                        context,
                        skip_event_filter=True,
                    )
                except Exception:
                    logger.exception("Error firing ON_TTS_CANCELLED hook")

        logger.info(
            "TTS interrupted for session %s: reason=%s, position=%dms",
            session.id,
            reason,
            playback.position_ms,
        )
        return True

    def _on_backend_barge_in(self, session: VoiceSession) -> None:
        """Handle barge-in detected by backend."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        playback = self._playing_sessions.get(session.id)
        if not playback:
            return

        self._schedule(
            self._handle_barge_in(session, playback, room_id),
            name=f"backend_barge_in:{session.id}",
        )

    # -------------------------------------------------------------------------
    # Speech processing pipeline
    # -------------------------------------------------------------------------

    async def _process_speech_end(self, session: VoiceSession, audio: bytes, room_id: str) -> None:
        """Process speech end: fire hooks, transcribe, route inbound."""
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
            audio_frame = AudioFrame(
                data=audio,
                sample_rate=sample_rate,
                channels=1,
                sample_width=2,
            )
            text = await self._stt.transcribe(audio_frame)

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

            # Route through inbound pipeline
            from roomkit.models.delivery import InboundMessage
            from roomkit.models.event import TextContent

            inbound = InboundMessage(
                channel_id=self.channel_id,
                sender_id=session.participant_id,
                content=TextContent(body=final_text),
                metadata={"voice_session_id": session.id, "source": "voice"},
            )

            await self._framework.process_inbound(inbound)

        except Exception:
            logger.exception("Error processing speech end")

    # -------------------------------------------------------------------------
    # Channel interface
    # -------------------------------------------------------------------------

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        from roomkit.models.event import EventSource
        from roomkit.models.event import RoomEvent as RoomEventModel

        return RoomEventModel(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
                external_id=message.external_id,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        from roomkit.models.channel import ChannelOutput as ChannelOutputModel
        from roomkit.models.event import TextContent

        if self._streaming and not self._backend:
            return ChannelOutputModel.empty()

        if self._streaming and self._backend and isinstance(event.content, TextContent):
            await self._deliver_voice(event, binding, context)
            return ChannelOutputModel.empty()

        if not self._streaming and isinstance(event.content, TextContent) and self._tts:
            raise NotImplementedError(
                "VoiceChannel store-and-forward mode requires MediaStore support. "
                "Use streaming=True (default) for real-time voice, or implement "
                "MediaStore for async audio delivery."
            )

        return ChannelOutputModel.empty()

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

            target_sessions: list[VoiceSession] = []
            for session_id, (bound_room_id, bound_binding) in self._session_bindings.items():
                if bound_room_id == room_id and bound_binding.channel_id == binding.channel_id:
                    session = self._backend.get_session(session_id)
                    if session:
                        target_sessions.append(session)

            if not target_sessions:
                target_sessions = self._backend.list_sessions(room_id)

            for session in target_sessions:
                await self._backend.send_transcription(session, final_text, "assistant")

                self._playing_sessions[session.id] = TTSPlaybackState(
                    session_id=session.id,
                    text=final_text,
                )

                try:
                    audio_stream = self._tts.synthesize_stream(final_text)
                    await self._backend.send_audio(session, audio_stream)
                except NotImplementedError:
                    await self._tts.synthesize(final_text)
                    logger.warning("TTS provider %s doesn't support streaming", self._tts.name)
                finally:
                    self._playing_sessions.pop(session.id, None)

            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.AFTER_TTS,
                final_text,
                context,
                skip_event_filter=True,
            )

        except Exception:
            logger.exception("Error delivering voice audio")

    async def close(self) -> None:
        if self._stt:
            await self._stt.close()
        if self._tts:
            await self._tts.close()
        if self._pipeline is not None:
            self._pipeline.close()
        if self._backend:
            await self._backend.close()
        self._session_bindings.clear()
        self._playing_sessions.clear()
