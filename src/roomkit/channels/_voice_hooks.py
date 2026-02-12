"""VoiceChannel mixin — hook firing and framework event emission."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.diarization.base import DiarizationResult

logger = logging.getLogger("roomkit.voice")


class VoiceHooksMixin:
    """Hook-firing and framework-event helpers for VoiceChannel."""

    # -- attributes provided by VoiceChannel.__init__ --
    channel_id: str
    _framework: RoomKit | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]

    # -------------------------------------------------------------------------
    # Framework event emitters (session lifecycle, errors)
    # -------------------------------------------------------------------------

    async def _emit_session_started(self, session: VoiceSession, room_id: str) -> None:
        if not self._framework:
            return
        try:
            await self._framework._emit_framework_event(
                "voice_session_started",
                room_id=room_id,
                data={
                    "session_id": session.id,
                    "channel_id": self.channel_id,
                },
            )
        except Exception:
            logger.exception("Error emitting voice_session_started")

    async def _emit_session_ended(self, session: VoiceSession, room_id: str) -> None:
        if not self._framework:
            return
        try:
            await self._framework._emit_framework_event(
                "voice_session_ended",
                room_id=room_id,
                data={
                    "session_id": session.id,
                    "channel_id": self.channel_id,
                },
            )
        except Exception:
            logger.exception("Error emitting voice_session_ended")

    async def _emit_recording_started(
        self, session: VoiceSession, recording_id: str, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            await self._framework._emit_framework_event(
                "recording_started",
                room_id=room_id,
                data={"session_id": session.id, "id": recording_id},
            )
        except Exception:
            logger.exception("Error emitting recording_started")

    async def _emit_recording_stopped(
        self,
        session: VoiceSession,
        recording_id: str,
        room_id: str,
        *,
        duration_seconds: float = 0.0,
    ) -> None:
        if not self._framework:
            return
        try:
            await self._framework._emit_framework_event(
                "recording_stopped",
                room_id=room_id,
                data={
                    "session_id": session.id,
                    "id": recording_id,
                    "duration_seconds": duration_seconds,
                },
            )
        except Exception:
            logger.exception("Error emitting recording_stopped")

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

    async def _fire_partial_transcription_hook(
        self, session: VoiceSession, result: Any, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import PartialTranscriptionEvent

            context = await self._framework._build_context(room_id)
            event = PartialTranscriptionEvent(
                session=session,
                text=result.text,
                confidence=result.confidence or 0.0,
                is_stable=result.is_final,
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

    async def _fire_audio_level_hook(
        self,
        session: VoiceSession,
        level_db: float,
        room_id: str,
        trigger: HookTrigger,
    ) -> None:
        if not self._framework:
            return
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

    async def _fire_backchannel_hook(self, session: VoiceSession, text: str, room_id: str) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import BackchannelEvent

            context = await self._framework._build_context(room_id)
            event = BackchannelEvent(
                session=session,
                text=text,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_BACKCHANNEL,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_BACKCHANNEL hook")

    async def _fire_dtmf_hook(self, session: VoiceSession, dtmf_event: Any, room_id: str) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import DTMFDetectedEvent

            context = await self._framework._build_context(room_id)
            event = DTMFDetectedEvent(
                session=session,
                digit=dtmf_event.digit,
                duration_ms=dtmf_event.duration_ms,
                confidence=dtmf_event.confidence,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_DTMF,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_DTMF hook")

    async def _fire_recording_started_hook(
        self, session: VoiceSession, handle: Any, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import RecordingStartedEvent

            context = await self._framework._build_context(room_id)
            event = RecordingStartedEvent(
                session=session,
                id=handle.id,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_RECORDING_STARTED,
                event,
                context,
                skip_event_filter=True,
            )
            await self._emit_recording_started(session, handle.id, room_id)
        except Exception:
            logger.exception("Error firing ON_RECORDING_STARTED hook")

    async def _fire_recording_stopped_hook(
        self, session: VoiceSession, result: Any, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import RecordingStoppedEvent

            context = await self._framework._build_context(room_id)
            event = RecordingStoppedEvent(
                session=session,
                id=result.id,
                urls=tuple(result.urls),
                duration_seconds=result.duration_seconds,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_RECORDING_STOPPED,
                event,
                context,
                skip_event_filter=True,
            )
            await self._emit_recording_stopped(
                session, result.id, room_id, duration_seconds=result.duration_seconds
            )
        except Exception:
            logger.exception("Error firing ON_RECORDING_STOPPED hook")
