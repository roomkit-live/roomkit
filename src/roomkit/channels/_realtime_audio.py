"""Audio inbound/outbound paths for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.enums import Access, HookTrigger
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.utils import rms_db

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")


@runtime_checkable
class RealtimeAudioHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeAudioMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_bindings: Per-session channel binding info.
        _session_resamplers: Per-session (inbound, outbound) resampler pairs.
        _session_transport_rates: Negotiated transport sample rate per session.
        _recording_tracks: Per-session recording track and room ID.
        _audio_forward_count: Count of audio chunks forwarded per session.
        _audio_generation: Generation counter per session for stale audio detection.
        _input_sample_rate: Provider input sample rate.
        _output_sample_rate: Provider output sample rate.
        _pipeline: The active audio pipeline instance (or None).
        _provider: The realtime voice provider.
        _transport: The voice backend transport.
        _framework: The RoomKit framework instance (or None).
        _recording: Recording configuration.
        _user_speaking: Whether the user is currently speaking per session.
        _barge_in_active: Session IDs with an active barge-in.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
        _fire_audio_level_task: Fire audio level hooks (from RealtimeSpeechMixin).
        _update_idle_event: Update the idle signaling event for a session.
        _fire_barge_in_hook: Fire barge-in hook (from RealtimeSpeechMixin).
        _handle_speech_event: Handle speech start/end events (from RealtimeSpeechMixin).
        _send_client_message: Send a JSON message to the client UI.
    """

    _state_lock: threading.Lock
    _session_bindings: dict[str, Any]
    _session_resamplers: dict[str, Any]
    _session_transport_rates: dict[str, int]
    _recording_tracks: dict[str, Any]
    _audio_forward_count: dict[str, int]
    _audio_generation: dict[str, int]
    _input_sample_rate: int
    _output_sample_rate: int
    _pipeline: AudioPipeline | None
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _recording: Any
    _user_speaking: dict[str, bool]
    _barge_in_active: set[str]

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _fire_audio_level_task(self, session: Any, level_db: float, trigger: Any) -> None: ...

    def _update_idle_event(self, session_id: str) -> None: ...

    async def _fire_barge_in_hook(self, session: Any) -> None: ...

    async def _handle_speech_event(self, session: Any, event_type: str) -> None: ...

    async def _send_client_message(self, session: Any, message: dict[str, Any]) -> None: ...


class RealtimeAudioMixin:
    """Inbound/outbound audio paths, recording, and resampling.

    Handles both the pipeline path (when ``pipeline=`` is configured)
    and the direct path (when no pipeline — backward compatibility).

    Host contract: :class:`RealtimeAudioHost`.
    """

    _state_lock: threading.Lock
    _session_bindings: dict[str, Any]
    _session_resamplers: dict[str, Any]
    _session_transport_rates: dict[str, int]
    _recording_tracks: dict[str, Any]
    _audio_forward_count: dict[str, int]
    _audio_generation: dict[str, int]
    _input_sample_rate: int
    _output_sample_rate: int
    _pipeline: AudioPipeline | None
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _recording: Any
    _user_speaking: dict[str, bool]
    _barge_in_active: set[str]

    _track_task: Any  # see RealtimeAudioHost — cross-mixin
    _fire_audio_level_task: Any  # see RealtimeAudioHost — cross-mixin
    _update_idle_event: Any  # see RealtimeAudioHost — cross-mixin
    _fire_barge_in_hook: Any  # see RealtimeAudioHost — cross-mixin
    _handle_speech_event: Any  # see RealtimeAudioHost — cross-mixin
    _send_client_message: Any  # see RealtimeAudioHost — cross-mixin

    # -----------------------------------------------------------------
    # Recording setup
    # -----------------------------------------------------------------

    def _wire_realtime_recording(self, room_id: str, session: VoiceSession) -> None:
        """Wire room-level audio recording for a realtime voice session.

        Registers a single audio track that receives both mic input and
        AI output.  The recorder injects silence to fill gaps between
        speech segments, keeping the audio stream continuous and in sync
        with video.
        """
        if self._recording is None or not self._recording.audio:
            return
        if not self._framework:
            return

        mgr = self._framework._room_recorder_mgr
        if not mgr.has_recorders(room_id):
            return

        from roomkit.recorder.base import RecordingTrack

        audio_track = RecordingTrack(
            id=f"audio:{session.id}",
            kind="audio",
            channel_id=self.channel_id,  # ty: ignore[unresolved-attribute]
            codec="pcm_s16le",
            sample_rate=self._output_sample_rate,
        )
        mgr.on_track_added(room_id, audio_track)

        with self._state_lock:
            self._recording_tracks[session.id] = (audio_track, room_id)

        logger.info(
            "Realtime recording wired for session %s (rate=%dHz)",
            session.id,
            self._output_sample_rate,
        )

    # -----------------------------------------------------------------
    # Pipeline audio path (active when pipeline= is configured)
    # -----------------------------------------------------------------

    def _pipeline_on_audio_received(
        self,
        session: VoiceSession,
        frame: AudioFrame,
    ) -> None:
        """Handle raw audio from transport — gate by binding, feed pipeline.

        Overrides the mixin's default to use RealtimeVoiceChannel's
        binding format (``_session_bindings[sid]`` -> ChannelBinding, not
        a tuple).
        """
        with self._state_lock:
            binding = self._session_bindings.get(session.id)
        if binding is not None and (
            binding.access in (Access.READ_ONLY, Access.NONE) or binding.muted
        ):
            return

        if self._pipeline is not None:
            self._pipeline.process_inbound(session, frame)

    def _on_pipeline_processed_frame(
        self,
        session: VoiceSession,
        frame: AudioFrame,
    ) -> None:
        """Forward processed audio from pipeline to provider.

        Called for every frame after pipeline processing (AEC, denoiser,
        etc.).  Applies inbound resampling if needed, taps recording,
        then sends to the realtime provider.
        """
        if session.state != VoiceSessionState.ACTIVE:
            return

        audio = frame.data

        # Inbound resampling: transport rate -> provider rate (e.g. SIP 8kHz -> 16kHz)
        with self._state_lock:
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
            rec = self._recording_tracks.get(session.id)
        if resamplers and transport_rate and transport_rate != self._input_sample_rate:
            from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

            f = _AudioFrame(data=audio, sample_rate=transport_rate, channels=1, sample_width=2)
            f = resamplers[0].resample(f, self._input_sample_rate, 1, 2)
            audio = f.data

        # Recording tap: send processed mic audio to room recorder
        if rec is not None and self._framework is not None:
            audio_track, rec_room_id = rec
            self._framework._room_recorder_mgr.on_data(
                rec_room_id, audio_track, audio, time.monotonic() * 1000
            )

        self._fire_audio_level_task(
            session,
            rms_db(audio),
            HookTrigger.ON_INPUT_AUDIO_LEVEL,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._provider.send_audio(session, audio),
            name=f"rt_send_audio:{session.id}",
        )

    # -----------------------------------------------------------------
    # Pipeline VAD callbacks
    # -----------------------------------------------------------------

    def _on_pipeline_vad_event(
        self,
        session: VoiceSession,
        vad_event: Any,
    ) -> None:
        """Handle VAD events from the local pipeline.

        SPEECH_START triggers barge-in detection + activityStart to provider.
        SPEECH_END triggers activityEnd to provider.
        """
        from roomkit.voice.pipeline.vad.base import VADEventType

        logger.info(
            "[VAD] %s (session %s, confidence=%.2f)",
            vad_event.type.name,
            session.id,
            getattr(vad_event, "confidence", 0.0),
        )
        if vad_event.type == VADEventType.SPEECH_START:
            self._on_pipeline_speech_start(session)
        elif vad_event.type == VADEventType.SPEECH_END:
            self._on_pipeline_speech_end(session)

    def _on_pipeline_speech_start(self, session: VoiceSession) -> None:
        """Handle speech start from local pipeline VAD."""
        with self._state_lock:
            self._user_speaking[session.id] = True
            self._audio_generation[session.id] = self._audio_generation.get(session.id, 0) + 1
            resamplers = self._session_resamplers.get(session.id)
            fwd_count = self._audio_forward_count.get(session.id, 0)
            is_barge_in = fwd_count > 0
            if is_barge_in:
                self._barge_in_active.add(session.id)
            # Reset outbound resampler inside lock to prevent race with
            # concurrent _resample_outbound_with calls.
            if resamplers:
                resamplers[1].reset()
        self._update_idle_event(session.id)

        logger.info(
            "[BARGE-IN] speech_start → interrupt (session %s, "
            "is_barge_in=%s, forwarded=%d chunks)",
            session.id,
            is_barge_in,
            fwd_count,
        )
        self._transport.interrupt(session)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        # Send clear_audio IMMEDIATELY — before hooks or context building.
        self._track_task(
            loop,
            self._send_client_message(session, {"type": "clear_audio"}),
            name=f"rt_clear_audio:{session.id}",
        )

        if is_barge_in:
            self._track_task(
                loop,
                self._fire_barge_in_hook(session),
                name=f"rt_barge_in:{session.id}",
            )

        self._track_task(
            loop,
            self._handle_speech_event(session, "start"),
            name=f"rt_speech_start:{session.id}",
        )

        self._track_task(
            loop,
            self._provider.send_activity_start(session),
            name=f"rt_activity_start:{session.id}",
        )

    def _on_pipeline_speech_end(self, session: VoiceSession) -> None:
        """Handle speech end from local pipeline VAD."""
        with self._state_lock:
            self._user_speaking[session.id] = False
        self._update_idle_event(session.id)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._track_task(
            loop,
            self._handle_speech_event(session, "end"),
            name=f"rt_speech_end:{session.id}",
        )

        self._track_task(
            loop,
            self._provider.send_activity_end(session),
            name=f"rt_activity_end:{session.id}",
        )

    # -----------------------------------------------------------------
    # Direct audio path (active when no pipeline is configured)
    # -----------------------------------------------------------------

    def _on_client_audio(self, session: VoiceSession, audio: AudioFrame | bytes) -> Any:
        """Forward client audio to provider."""
        if not isinstance(audio, bytes):
            audio = audio.data

        # Recording tap: send mic audio to room recorder
        with self._state_lock:
            rec = self._recording_tracks.get(session.id)
        if rec is not None and self._framework is not None:
            audio_track, rec_room_id = rec
            self._framework._room_recorder_mgr.on_data(
                rec_room_id,
                audio_track,
                audio,
                time.monotonic() * 1000,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._forward_client_audio(session, audio, time.monotonic()),
            name=f"rt_client_audio:{session.id}",
        )

    async def _forward_client_audio(
        self,
        session: VoiceSession,
        audio: bytes,
        enqueued_at: float = 0.0,
    ) -> None:
        if session.state != VoiceSessionState.ACTIVE:
            return
        with self._state_lock:
            binding = self._session_bindings.get(session.id)
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
        if binding is not None and (
            binding.access in (Access.READ_ONLY, Access.NONE) or binding.muted
        ):
            return
        try:
            if resamplers and transport_rate and transport_rate != self._input_sample_rate:
                from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

                frame = _AudioFrame(
                    data=audio,
                    sample_rate=transport_rate,
                    channels=1,
                    sample_width=2,
                )
                frame = resamplers[0].resample(frame, self._input_sample_rate, 1, 2)
                audio = frame.data

            self._fire_audio_level_task(
                session,
                rms_db(audio),
                HookTrigger.ON_INPUT_AUDIO_LEVEL,
            )
            await self._provider.send_audio(session, audio)
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.exception("Error forwarding client audio for session %s", session.id)

    # -----------------------------------------------------------------
    # Outbound audio (provider -> transport)
    # -----------------------------------------------------------------

    def _on_provider_audio(self, session: VoiceSession, audio: bytes) -> None:
        """Resample + forward provider audio to transport.

        Returns ``None`` so the provider's ``_fire_audio_callbacks`` does not
        await anything — the receive loop is never blocked.

        Each task captures the current generation counter so that tasks
        created before an interrupt are silently discarded.
        """
        with self._state_lock:
            # Drop outbound audio while user is speaking — prevents new
            # provider chunks from refilling the transport buffer after
            # interrupt() flushed it during barge-in.
            if self._user_speaking.get(session.id, False):
                logger.debug("[BARGE-IN] dropping outbound audio (user speaking)")
                return
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
            gen = self._audio_generation.get(session.id, 0)
            binding = self._session_bindings.get(session.id)
            if binding is not None and binding.output_muted:
                return
        audio = self._resample_outbound_with(audio, resamplers, transport_rate)
        if not audio:
            return

        # Route through pipeline outbound path for AEC reference feeding,
        # recorder taps, and post-processors.
        if self._pipeline is not None:
            from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

            rate = transport_rate or self._output_sample_rate
            out_frame = _AudioFrame(data=audio, sample_rate=rate, channels=1, sample_width=2)
            out_frame = self._pipeline.process_outbound(session, out_frame)
            audio = out_frame.data

        # Fire outbound audio taps (e.g. avatar lip-sync)
        rate = transport_rate or self._output_sample_rate
        for cb in getattr(self, "_outbound_audio_taps", []):
            try:
                cb(session, audio, rate)
            except Exception:
                logger.debug("Outbound audio tap error", exc_info=True)

        # Recording tap: send AI audio to room recorder (non-pipeline path)
        if self._pipeline is None:
            with self._state_lock:
                rec = self._recording_tracks.get(session.id)
            if rec is not None and self._framework is not None:
                audio_track, rec_room_id = rec
                self._framework._room_recorder_mgr.on_data(
                    rec_room_id,
                    audio_track,
                    audio,
                    time.monotonic() * 1000,
                )

        with self._state_lock:
            self._audio_forward_count[session.id] = (
                self._audio_forward_count.get(session.id, 0) + 1
            )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._send_outbound_audio(session, audio, gen),
            name=f"rt_send_audio:{session.id}",
        )

    async def _send_outbound_audio(self, session: VoiceSession, audio: bytes, gen: int) -> None:
        """Send audio to transport, skipping if the generation is stale."""
        with self._state_lock:
            current_gen = self._audio_generation.get(session.id, 0)
        if current_gen != gen:
            return

        binding = self._session_bindings.get(session.id)
        if binding is not None and binding.output_muted:
            return

        await self._transport.send_audio(session, audio)
        if not self._transport.supports_playback_callback:
            self._fire_audio_level_task(
                session,
                rms_db(audio),
                HookTrigger.ON_OUTPUT_AUDIO_LEVEL,
            )

    def _on_transport_audio_played(self, session: VoiceSession, audio: AudioFrame | bytes) -> None:
        """Fire ON_OUTPUT_AUDIO_LEVEL at real playback pace (PortAudio callback)."""
        raw = audio if isinstance(audio, bytes) else audio.data
        self._fire_audio_level_task(session, rms_db(raw), HookTrigger.ON_OUTPUT_AUDIO_LEVEL)

    def _resample_outbound_with(
        self,
        audio: bytes,
        resamplers: tuple[Any, Any] | None,
        transport_rate: int | None,
    ) -> bytes:
        """Resample outbound audio using pre-snapshotted resamplers/rate."""
        if resamplers and transport_rate and transport_rate != self._output_sample_rate:
            from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

            frame = _AudioFrame(
                data=audio,
                sample_rate=self._output_sample_rate,
                channels=1,
                sample_width=2,
            )
            frame = resamplers[1].resample(frame, transport_rate, 1, 2)
            return bytes(frame.data)
        return audio
