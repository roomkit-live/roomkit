"""VoiceChannel mixin — hook firing and framework event emission."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._room_hooks import RoomHooks
from roomkit.models.enums import HookTrigger
from roomkit.telemetry.context import reset_span, set_current_span

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.events import SpeakerChangeEvent
    from roomkit.voice.pipeline.diarization.base import DiarizationResult

logger = logging.getLogger("roomkit.voice")


@runtime_checkable
class VoiceHookHost(Protocol):
    """Contract: capabilities a host class must provide for VoiceHooksMixin.

    Attributes provided by the host's ``__init__``:
        channel_id: Unique identifier for this channel instance.
        _framework: Reference to the RoomKit orchestrator (``None`` until
            the channel is registered).  The mixin accesses
            ``_framework._build_context``, ``_framework.hook_engine``,
            and ``_framework._emit_framework_event``.

    Optional attributes accessed via ``getattr`` with fallbacks:
        _voice_session_spans: Per-session telemetry span IDs
            (``dict[str, str | None]``).  Accessed with
            ``getattr(self, "_voice_session_spans", {})``.
    """

    channel_id: str
    _framework: RoomKit | None


class VoiceHooksMixin:
    """Hook-firing and framework-event helpers for VoiceChannel.

    Host contract: :class:`VoiceHookHost`.
    """

    channel_id: str
    _framework: RoomKit | None
    _pipeline_config: Any  # AudioPipelineConfig | None — see VoiceHookHost

    @contextmanager
    def _voice_span_ctx(self, session: VoiceSession) -> Generator[None, None, None]:
        """Set the voice session span as current for child spans."""
        parent = getattr(self, "_voice_session_spans", {}).get(session.id)
        token = (
            set_current_span(parent, telemetry=getattr(self, "_telemetry", None))
            if parent
            else None
        )
        try:
            yield
        finally:
            if token is not None:
                reset_span(token)

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

    # -------------------------------------------------------------------------
    # Hook firing helpers
    # -------------------------------------------------------------------------

    async def _fire_session_started_hook(self, session: VoiceSession, room_id: str) -> None:
        if not self._framework:
            return
        try:
            from roomkit.models.enums import ChannelType
            from roomkit.models.session_event import SessionStartedEvent

            with self._voice_span_ctx(session):
                event = SessionStartedEvent(
                    room_id=room_id,
                    channel_id=self.channel_id,
                    channel_type=ChannelType.VOICE,
                    participant_id=session.participant_id,
                    session=session,
                )
                await RoomHooks(self._framework, room_id).run_async(
                    HookTrigger.ON_SESSION_STARTED, event
                )
                await self._framework._emit_framework_event(
                    "session_started",
                    room_id=room_id,
                    data={
                        "session_id": session.id,
                        "channel_id": self.channel_id,
                    },
                )
                # RFC §8.2 ``voice_session_ready`` — the audio path is live.
                # Distinct from ``session_started``, which the RFC does not
                # define and which text channels raise too.
                await self._framework._emit_framework_event(
                    "voice_session_ready",
                    room_id=room_id,
                    channel_id=self.channel_id,
                    data={
                        "session_id": session.id,
                        "participant_id": session.participant_id,
                    },
                )
        except Exception:
            logger.exception("Error firing ON_SESSION_STARTED hook")

    async def _fire_speech_start_hooks(self, session: VoiceSession, room_id: str) -> None:
        if not self._framework:
            return
        try:
            with self._voice_span_ctx(session):
                await RoomHooks(self._framework, room_id).run_async(
                    HookTrigger.ON_SPEECH_START, session
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

            with self._voice_span_ctx(session):
                event = PartialTranscriptionEvent(
                    session=session,
                    text=result.text,
                    confidence=result.confidence or 0.0,
                    is_stable=result.is_final,
                    language=result.language,
                )
                await RoomHooks(self._framework, room_id).run_async(
                    HookTrigger.ON_PARTIAL_TRANSCRIPTION, event
                )
        except Exception:
            logger.exception("Error firing ON_PARTIAL_TRANSCRIPTION hook")

    async def _fire_vad_silence_hook(
        self, session: VoiceSession, silence_duration_ms: int, room_id: str
    ) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import VADSilenceEvent

            with self._voice_span_ctx(session):
                event = VADSilenceEvent(
                    session=session,
                    silence_duration_ms=silence_duration_ms,
                )
                await RoomHooks(self._framework, room_id).run_async(
                    HookTrigger.ON_VAD_SILENCE, event
                )
        except Exception:
            logger.exception("Error firing ON_VAD_SILENCE hook")

    async def _fire_vad_audio_level_hook(
        self, session: VoiceSession, level_db: float, is_speech: bool, room_id: str
    ) -> None:
        if not self._framework:
            return
        if not self._framework.hook_engine.has_hooks(HookTrigger.ON_VAD_AUDIO_LEVEL):
            return
        try:
            from roomkit.voice.events import VADAudioLevelEvent

            with self._voice_span_ctx(session):
                event = VADAudioLevelEvent(
                    session=session,
                    level_db=level_db,
                    is_speech=is_speech,
                )
                await RoomHooks(self._framework, room_id).run_async(
                    HookTrigger.ON_VAD_AUDIO_LEVEL, event
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
        if not self._framework.hook_engine.has_hooks(trigger):
            return
        try:
            from roomkit.voice.events import AudioLevelEvent

            with self._voice_span_ctx(session):
                event = AudioLevelEvent(session=session, level_db=level_db)
                await RoomHooks(self._framework, room_id).run_async(trigger, event)
        except Exception:
            logger.exception("Error firing %s hook", trigger)

    async def _fire_speaker_change_hook(
        self, session: VoiceSession, result: DiarizationResult, room_id: str
    ) -> None:
        """ON_SPEAKER_CHANGE from the pipeline's diarization stage."""
        from roomkit.voice.events import SpeakerChangeEvent

        if result.speaker_id is None:  # nobody: the pipeline reports no change for it
            return
        event = SpeakerChangeEvent(
            session=session,
            speaker_id=result.speaker_id,
            confidence=result.confidence,
            is_new_speaker=result.is_new_speaker,
        )
        await self._fire_speaker_change_event(session, event, room_id)

    async def _fire_speaker_change_event(
        self, session: VoiceSession, event: SpeakerChangeEvent, room_id: str
    ) -> None:
        """Run the ON_SPEAKER_CHANGE hooks, whatever the source of the change."""
        if not self._framework:
            return
        try:
            with self._voice_span_ctx(session):
                await RoomHooks(self._framework, room_id).run_async(
                    HookTrigger.ON_SPEAKER_CHANGE, event
                )
        except Exception:
            logger.exception("Error firing ON_SPEAKER_CHANGE hook")

    async def _fire_backchannel_hook(self, session: VoiceSession, text: str, room_id: str) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import BackchannelEvent

            with self._voice_span_ctx(session):
                event = BackchannelEvent(
                    session=session,
                    text=text,
                )
                await RoomHooks(self._framework, room_id).run_async(
                    HookTrigger.ON_BACKCHANNEL, event
                )
        except Exception:
            logger.exception("Error firing ON_BACKCHANNEL hook")

    async def _fire_dtmf_hook(self, session: VoiceSession, dtmf_event: Any, room_id: str) -> None:
        if not self._framework:
            return
        try:
            from roomkit.voice.events import DTMFDetectedEvent

            with self._voice_span_ctx(session):
                redaction = (
                    self._pipeline_config.dtmf_redaction
                    if self._pipeline_config is not None
                    else None
                )
                event = DTMFDetectedEvent(
                    session=session,
                    digit=dtmf_event.digit,
                    duration_ms=dtmf_event.duration_ms,
                    confidence=dtmf_event.confidence,
                    redacted_digit=(
                        redaction.mask(dtmf_event.digit)
                        if redaction is not None
                        else dtmf_event.digit
                    ),
                )
                await RoomHooks(self._framework, room_id).run_async(HookTrigger.ON_DTMF, event)
        except Exception:
            logger.exception("Error firing ON_DTMF hook")
