"""Voice channel for real-time audio communication."""

from __future__ import annotations

import asyncio
import logging
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
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
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

    When a VoiceBackend is configured, the channel automatically:
    - Registers for VAD (Voice Activity Detection) callbacks
    - Transcribes speech using the STT provider
    - Routes transcriptions through the standard inbound pipeline
    - Synthesizes AI responses using TTS and streams to the client
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
        streaming: bool = True,
        enable_barge_in: bool = True,
        barge_in_threshold_ms: int = 200,
    ) -> None:
        """Initialize voice channel.

        Args:
            channel_id: Unique channel identifier.
            stt: Speech-to-text provider for transcription.
            tts: Text-to-speech provider for synthesis.
            backend: Voice transport backend for real-time audio.
            streaming: If True (default), deliver() streams audio via backend.
                If False, deliver() requires MediaStore support (not implemented).
            enable_barge_in: If True (default), detect when user speaks during TTS
                and fire ON_BARGE_IN hook. Requires backend with INTERRUPTION capability.
            barge_in_threshold_ms: Minimum TTS playback time before barge-in is
                detected. Helps avoid false triggers from very short interruptions.
        """
        super().__init__(channel_id)
        self._stt = stt
        self._tts = tts
        self._backend = backend
        self._streaming = streaming
        self._enable_barge_in = enable_barge_in
        self._barge_in_threshold_ms = barge_in_threshold_ms
        self._framework: RoomKit | None = None
        # Map session_id -> (room_id, binding) for routing
        self._session_bindings: dict[str, tuple[str, ChannelBinding]] = {}
        # Track TTS playback for barge-in detection
        self._playing_sessions: dict[str, TTSPlaybackState] = {}

        # Register VAD callbacks if backend is provided
        if backend:
            backend.on_speech_start(self._on_speech_start)
            backend.on_speech_end(self._on_speech_end)

            # Register for enhanced callbacks based on capabilities
            if VoiceCapability.PARTIAL_STT in backend.capabilities:
                backend.on_partial_transcription(self._on_partial_transcription)
            if VoiceCapability.VAD_SILENCE in backend.capabilities:
                backend.on_vad_silence(self._on_vad_silence)
            if VoiceCapability.VAD_AUDIO_LEVEL in backend.capabilities:
                backend.on_vad_audio_level(self._on_vad_audio_level)
            if VoiceCapability.BARGE_IN in backend.capabilities:
                backend.on_barge_in(self._on_backend_barge_in)

    def set_framework(self, framework: RoomKit) -> None:
        """Set the framework reference for inbound routing.

        Called automatically when the channel is registered with RoomKit.
        """
        self._framework = framework

    def bind_session(self, session: VoiceSession, room_id: str, binding: ChannelBinding) -> None:
        """Bind a voice session to a room for message routing.

        Args:
            session: The voice session to bind.
            room_id: The room to route messages to.
            binding: The channel binding for delivery.
        """
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
        }

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.TEXT],
            supports_audio=True,
            supported_audio_formats=["wav", "mp3", "ogg", "webm"],
            max_audio_duration_seconds=3600,
        )

    def _on_speech_start(self, session: VoiceSession) -> None:
        """Handle VAD speech start event.

        Fires ON_SPEECH_START hooks for the bound room.
        Also checks for barge-in if TTS is playing.
        """
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        # Fire hook asynchronously (don't block VAD processing)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        # Check for barge-in (user speaking while TTS is playing)
        playback = self._playing_sessions.get(session.id)
        if (
            self._enable_barge_in
            and playback
            and playback.position_ms >= self._barge_in_threshold_ms
        ):
            loop.create_task(
                self._handle_barge_in(session, playback, room_id),
                name=f"barge_in:{session.id}",
            )

        loop.create_task(
            self._fire_speech_start_hooks(session, room_id),
            name=f"speech_start:{session.id}",
        )

    async def _fire_speech_start_hooks(self, session: VoiceSession, room_id: str) -> None:
        """Fire ON_SPEECH_START hooks."""
        if not self._framework:
            return

        try:
            context = await self._framework._build_context(room_id)
            # Skip event filtering for voice hooks - they receive VoiceSession instead of RoomEvent
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SPEECH_START,
                session,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_SPEECH_START hooks")

    async def _handle_barge_in(
        self, session: VoiceSession, playback: TTSPlaybackState, room_id: str
    ) -> None:
        """Handle barge-in: fire hook and interrupt TTS."""
        if not self._framework:
            return

        try:
            from roomkit.voice.events import BargeInEvent

            context = await self._framework._build_context(room_id)

            # Fire ON_BARGE_IN hook
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

            # Interrupt TTS playback
            await self.interrupt(session, reason="barge_in")

        except Exception:
            logger.exception("Error handling barge-in for session %s", session.id)

    async def interrupt(self, session: VoiceSession, *, reason: str = "explicit") -> bool:
        """Interrupt ongoing TTS playback for a session.

        Args:
            session: The voice session to interrupt.
            reason: Why the TTS was cancelled ('barge_in', 'explicit', etc.)

        Returns:
            True if TTS was cancelled, False if nothing was playing.
        """
        playback = self._playing_sessions.pop(session.id, None)
        if not playback:
            return False

        # Cancel audio in backend if supported
        if self._backend and VoiceCapability.INTERRUPTION in self._backend.capabilities:
            await self._backend.cancel_audio(session)

        # Fire ON_TTS_CANCELLED hook
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

    def _on_partial_transcription(
        self, session: VoiceSession, text: str, confidence: float, is_stable: bool
    ) -> None:
        """Handle partial transcription from streaming STT."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        loop.create_task(
            self._fire_partial_transcription_hook(session, text, confidence, is_stable, room_id),
            name=f"partial_transcription:{session.id}",
        )

    async def _fire_partial_transcription_hook(
        self,
        session: VoiceSession,
        text: str,
        confidence: float,
        is_stable: bool,
        room_id: str,
    ) -> None:
        """Fire ON_PARTIAL_TRANSCRIPTION hook."""
        if not self._framework:
            return

        try:
            from roomkit.voice.events import PartialTranscriptionEvent

            context = await self._framework._build_context(room_id)
            event = PartialTranscriptionEvent(
                session=session,
                text=text,
                confidence=confidence,
                is_stable=is_stable,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_PARTIAL_TRANSCRIPTION,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_PARTIAL_TRANSCRIPTION hook")

    def _on_vad_silence(self, session: VoiceSession, silence_duration_ms: int) -> None:
        """Handle VAD silence detection."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        loop.create_task(
            self._fire_vad_silence_hook(session, silence_duration_ms, room_id),
            name=f"vad_silence:{session.id}",
        )

    async def _fire_vad_silence_hook(
        self, session: VoiceSession, silence_duration_ms: int, room_id: str
    ) -> None:
        """Fire ON_VAD_SILENCE hook."""
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

    def _on_vad_audio_level(self, session: VoiceSession, level_db: float, is_speech: bool) -> None:
        """Handle VAD audio level update."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        loop.create_task(
            self._fire_vad_audio_level_hook(session, level_db, is_speech, room_id),
            name=f"vad_audio_level:{session.id}",
        )

    async def _fire_vad_audio_level_hook(
        self, session: VoiceSession, level_db: float, is_speech: bool, room_id: str
    ) -> None:
        """Fire ON_VAD_AUDIO_LEVEL hook."""
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

    def _on_backend_barge_in(self, session: VoiceSession) -> None:
        """Handle barge-in detected by backend."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        playback = self._playing_sessions.get(session.id)

        if not playback:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        loop.create_task(
            self._handle_barge_in(session, playback, room_id),
            name=f"backend_barge_in:{session.id}",
        )

    def _on_speech_end(self, session: VoiceSession, audio: bytes) -> None:
        """Handle VAD speech end event.

        Fires ON_SPEECH_END hooks, transcribes audio, and routes to framework.
        """
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        # Process asynchronously
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        loop.create_task(
            self._process_speech_end(session, audio, room_id),
            name=f"speech_end:{session.id}",
        )

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

            from roomkit.voice.base import AudioChunk

            # Get audio parameters from session metadata (set by backend)
            sample_rate = session.metadata.get("input_sample_rate", 16000)
            audio_chunk = AudioChunk(
                data=audio,
                sample_rate=sample_rate,
                channels=1,
                format="pcm_s16le",
            )
            text = await self._stt.transcribe(audio_chunk)

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

        # In streaming mode without backend, delivery is handled externally
        if self._streaming and not self._backend:
            return ChannelOutputModel.empty()

        # In streaming mode with backend, stream TTS audio to the session
        if self._streaming and self._backend and isinstance(event.content, TextContent):
            await self._deliver_voice(event, binding, context)
            return ChannelOutputModel.empty()

        # Store-and-forward mode: synthesize audio for later retrieval.
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
        """Deliver text event as voice audio via the backend."""
        if not self._tts or not self._backend or not self._framework:
            return

        from roomkit.models.event import TextContent

        if not isinstance(event.content, TextContent):
            return

        text = event.content.body
        room_id = event.room_id

        try:
            # Fire BEFORE_TTS hooks (sync, can modify)
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

            # Use potentially modified text
            final_text = before_result.event if isinstance(before_result.event, str) else text

            logger.info("AI response: %s", final_text)

            # Find the session(s) to send audio to
            # Look for sessions bound to this room
            target_sessions: list[VoiceSession] = []
            for session_id, (bound_room_id, bound_binding) in self._session_bindings.items():
                if bound_room_id == room_id and bound_binding.channel_id == binding.channel_id:
                    session = self._backend.get_session(session_id)
                    if session:
                        target_sessions.append(session)

            if not target_sessions:
                # Fallback: get all sessions in the room from backend
                target_sessions = self._backend.list_sessions(room_id)

            # Stream TTS audio to each session
            for session in target_sessions:
                # Send AI response text to client UI
                await self._backend.send_transcription(session, final_text, "assistant")

                # Track playback state for barge-in detection
                self._playing_sessions[session.id] = TTSPlaybackState(
                    session_id=session.id,
                    text=final_text,
                )

                try:
                    # Use streaming if available
                    audio_stream = self._tts.synthesize_stream(final_text)
                    await self._backend.send_audio(session, audio_stream)
                except NotImplementedError:
                    # Fallback to non-streaming
                    await self._tts.synthesize(final_text)
                    # For non-streaming, we'd need to fetch the audio from URL
                    # This is a limitation - mock backends handle it
                    logger.warning("TTS provider %s doesn't support streaming", self._tts.name)
                finally:
                    # Clear playback state (TTS finished or failed)
                    self._playing_sessions.pop(session.id, None)

            # Fire AFTER_TTS hooks (async)
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
        if self._backend:
            await self._backend.close()
        self._session_bindings.clear()
        self._playing_sessions.clear()
