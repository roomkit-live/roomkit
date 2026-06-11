"""Audio inbound/outbound paths for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
        _resample_executor: Single-thread executor that owns every resampler
            state mutation (resample/flush/reset/close). One thread = FIFO,
            so frame order and state integrity hold without locking, and the
            event loop stays free to pace RTP while a resample runs.
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
        _user_turn_start_at: Wall-clock of the last user-turn start, per session.
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
    _resample_executor: ThreadPoolExecutor | None
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
    _user_turn_start_at: dict[str, Any]
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
    _resample_executor: ThreadPoolExecutor | None
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
    _user_turn_start_at: dict[str, Any]
    _barge_in_active: set[str]

    _track_task: Any  # see RealtimeAudioHost — cross-mixin
    _fire_audio_level_task: Any  # see RealtimeAudioHost — cross-mixin
    _update_idle_event: Any  # see RealtimeAudioHost — cross-mixin
    _fire_barge_in_hook: Any  # see RealtimeAudioHost — cross-mixin
    _handle_speech_event: Any  # see RealtimeAudioHost — cross-mixin
    _send_client_message: Any  # see RealtimeAudioHost — cross-mixin

    # -----------------------------------------------------------------
    # Off-loop resampling
    # -----------------------------------------------------------------

    def _get_resample_executor(self) -> ThreadPoolExecutor:
        """Lazily create the channel's single-thread resample executor.

        Dedicated (not the loop's default executor) so a 20 ms-budget audio
        frame never queues behind unrelated host work; single-thread so jobs
        run FIFO — frame order and resampler-state integrity hold without
        locking. Only ever called from the event loop thread, so the lazy
        init needs no guard.
        """
        if self._resample_executor is None:
            self._resample_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rk-resample"
            )
        return self._resample_executor

    async def _run_in_resample_executor(self, fn: Any, *args: Any) -> Any:
        """Run a resampler-state mutation (resample/flush) in the executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._get_resample_executor(), fn, *args)

    def _reset_outbound_resampler(self, resamplers: tuple[Any, Any] | None) -> None:
        """Reset the outbound resampler through the resample executor.

        FIFO behind any in-flight resample, so the reset never races state
        mutation. Stale outputs are discarded by the generation check in
        ``_send_outbound_audio``. Direct call when no executor exists — no
        resample ever ran, so there is nothing to race.
        """
        if not resamplers:
            return
        ex = self._resample_executor
        if ex is not None:
            ex.submit(resamplers[1].reset)
        else:
            resamplers[1].reset()

    async def _resample_off_loop(
        self,
        resampler: Any,
        frame: Any,
        target_rate: int,
        *,
        direction: str,
    ) -> Any:
        """Resample one frame in the executor, warning past a 20 ms budget.

        The measured time is the executor round-trip (queue wait + compute):
        past 20 ms — one RTP frame — the resampler is not keeping up with
        real time, even though the event loop itself stays free.
        """
        src_rate = frame.sample_rate
        in_bytes = len(frame.data)
        t0 = time.monotonic()
        result = await self._run_in_resample_executor(resampler.resample, frame, target_rate, 1, 2)
        dt_ms = (time.monotonic() - t0) * 1000
        if dt_ms > 20.0:
            logger.warning(
                "%s resample slow: %.1fms in_bytes=%d rate=%d→%d",
                direction,
                dt_ms,
                in_bytes,
                src_rate,
                target_rate,
            )
        return result

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
        etc.).  Snapshots per-session state, then hands the frame to a task
        so inbound resampling runs off the event loop; task-creation order
        plus the single-thread resample executor keep frames in order.
        """
        if session.state != VoiceSessionState.ACTIVE:
            return

        with self._state_lock:
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
            rec = self._recording_tracks.get(session.id)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._forward_pipeline_frame(session, frame.data, resamplers, transport_rate, rec),
            name=f"rt_send_audio:{session.id}",
        )

    async def _forward_pipeline_frame(
        self,
        session: VoiceSession,
        audio: bytes,
        resamplers: tuple[Any, Any] | None,
        transport_rate: int | None,
        rec: Any,
    ) -> None:
        """Resample (off-loop), tap recording, send pipeline audio to provider."""
        # Inbound resampling: transport rate -> provider rate (e.g. SIP 8kHz -> 16kHz)
        if resamplers and transport_rate and transport_rate != self._input_sample_rate:
            from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

            f = _AudioFrame(data=audio, sample_rate=transport_rate, channels=1, sample_width=2)
            f = await self._resample_off_loop(
                resamplers[0], f, self._input_sample_rate, direction="inbound"
            )
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
        await self._provider.send_audio(session, audio)

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
        from datetime import UTC, datetime

        with self._state_lock:
            self._user_speaking[session.id] = True
            # Stamp the start-of-turn so transcription emission can use it
            # as created_at — keeps user utterances sorted before any
            # tool_calls the agent fires mid-turn.
            self._user_turn_start_at[session.id] = datetime.now(UTC)
            self._audio_generation[session.id] = self._audio_generation.get(session.id, 0) + 1
            resamplers = self._session_resamplers.get(session.id)
            fwd_count = self._audio_forward_count.get(session.id, 0)
            is_barge_in = fwd_count > 0
            if is_barge_in:
                self._barge_in_active.add(session.id)
            self._reset_outbound_resampler(resamplers)
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
            self._barge_in_active.discard(session.id)
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
                frame = await self._resample_off_loop(
                    resamplers[0], frame, self._input_sample_rate, direction="inbound"
                )
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
        """Snapshot session state and hand provider audio to a send task.

        Returns ``None`` so the provider's ``_fire_audio_callbacks`` does not
        await anything — the receive loop is never blocked. Resampling
        happens inside the task, off the event loop; task-creation order
        plus the single-thread resample executor keep frames in order.

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
            # Counted at acceptance (not after the off-loop resample) so the
            # count is settled by the time response-end bookkeeping pops it.
            self._audio_forward_count[session.id] = (
                self._audio_forward_count.get(session.id, 0) + 1
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._process_provider_audio(session, audio, resamplers, transport_rate, gen),
            name=f"rt_send_audio:{session.id}",
        )

    async def _process_provider_audio(
        self,
        session: VoiceSession,
        audio: bytes,
        resamplers: tuple[Any, Any] | None,
        transport_rate: int | None,
        gen: int,
    ) -> None:
        """Resample (off-loop), run pipeline/taps/recording, send to transport."""
        if resamplers and transport_rate and transport_rate != self._output_sample_rate:
            from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

            frame = _AudioFrame(
                data=audio,
                sample_rate=self._output_sample_rate,
                channels=1,
                sample_width=2,
            )
            frame = await self._resample_off_loop(
                resamplers[1], frame, transport_rate, direction="outbound"
            )
            audio = bytes(frame.data)
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

        await self._send_outbound_audio(session, audio, gen)

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
