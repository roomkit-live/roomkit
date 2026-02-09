"""Voice channel for real-time audio communication."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Coroutine
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
from roomkit.voice.interruption import InterruptionConfig

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.delivery import InboundMessage
    from roomkit.models.event import RoomEvent
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.diarization.base import DiarizationResult
    from roomkit.voice.pipeline.vad.base import VADEvent
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.voice")


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


# Buffer ~200ms of audio before sending to STT stream.
# Small per-frame chunks cause resampling artifacts at frame boundaries
# when the provider resamples (e.g. 16kHz -> 24kHz).  Larger chunks
# give the stateless resampler enough context for clean interpolation.
_STT_STREAM_BUFFER_BYTES = 6400  # 200ms at 16kHz mono 16-bit


@dataclass
class _STTStreamState:
    """Track an active streaming STT session."""

    queue: asyncio.Queue[Any]  # Queue[AudioChunk | None]
    task: asyncio.Task[Any]  # consumer task running transcribe_stream
    frame_buffer: bytearray = field(default_factory=bytearray)
    frame_buffer_rate: int = 16000
    final_text: str | None = None
    error: bool = False
    cancelled: bool = False


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
    - Routes frames through the AudioPipeline inbound chain:
      [Resampler] -> [Recorder] -> [AEC] -> [AGC] -> [Denoiser] -> VAD ->
      [Diarization] + [DTMF]
    - Fires hooks based on pipeline events (speech, silence, DTMF, recording, etc.)
    - Transcribes speech using the STT provider
    - Optionally evaluates turn completion via TurnDetector
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
        interruption: InterruptionConfig | None = None,
    ) -> None:
        super().__init__(channel_id)
        self._stt = stt
        self._tts = tts
        self._backend = backend
        self._pipeline_config = pipeline
        self._streaming = streaming
        self._framework: RoomKit | None = None
        # Map session_id -> (room_id, binding) for routing
        self._session_bindings: dict[str, tuple[str, ChannelBinding]] = {}
        # Track TTS playback for barge-in detection
        self._playing_sessions: dict[str, TTSPlaybackState] = {}
        # The instantiated pipeline engine (if config provided)
        self._pipeline: Any = None  # AudioPipeline | None
        # Pending turns for turn detection (session_id -> list of TurnEntry)
        self._pending_turns: dict[str, list[Any]] = {}
        # Active streaming STT sessions (session_id -> state)
        self._stt_streams: dict[str, _STTStreamState] = {}
        # Continuous STT mode: stream all audio to STT, no local VAD
        self._continuous_stt = False
        self._continuous_stt_tasks: dict[str, asyncio.Task[Any]] = {}

        # Build InterruptionHandler: explicit config > pipeline config > legacy params
        from roomkit.voice.interruption import InterruptionHandler, InterruptionStrategy

        interruption_config = interruption
        if interruption_config is None and pipeline is not None:
            interruption_config = pipeline.interruption
        if interruption_config is None:
            # Map legacy boolean params: enable_barge_in + threshold mapped
            # to IMMEDIATE with allow_during_first_ms (old behaviour was:
            # interrupt at SPEECH_START if playback_position >= threshold).
            if not enable_barge_in:
                interruption_config = InterruptionConfig(
                    strategy=InterruptionStrategy.DISABLED,
                )
            else:
                interruption_config = InterruptionConfig(
                    strategy=InterruptionStrategy.IMMEDIATE,
                    allow_during_first_ms=barge_in_threshold_ms,
                )
        # Preserve legacy attrs for backwards compat in existing barge-in path
        self._enable_barge_in = interruption_config.strategy != InterruptionStrategy.DISABLED
        self._barge_in_threshold_ms = interruption_config.min_speech_ms

        backchannel_det = pipeline.backchannel_detector if pipeline else None
        self._interruption_handler = InterruptionHandler(
            interruption_config, backchannel_detector=backchannel_det
        )

        # Wire up pipeline if both backend and pipeline config are provided
        if backend and pipeline:
            self._setup_pipeline(backend, pipeline)
        elif backend and VoiceCapability.BARGE_IN in backend.capabilities:
            backend.on_barge_in(self._on_backend_barge_in)

    def _setup_pipeline(self, backend: VoiceBackend, config: AudioPipelineConfig) -> None:
        """Create AudioPipeline and wire backend -> pipeline -> callbacks."""
        from roomkit.voice.pipeline.engine import AudioPipeline

        self._pipeline = AudioPipeline(
            config,
            backend_capabilities=backend.capabilities,
            backend_feeds_aec_reference=backend.feeds_aec_reference,
        )

        # Backend delivers raw audio -> pipeline processes it
        backend.on_audio_received(self._on_audio_received)

        # Continuous STT: no local VAD, stream all audio to STT provider
        self._continuous_stt = (
            config.vad is None and self._stt is not None and self._stt.supports_streaming
        )

        # Pipeline events -> VoiceChannel hooks
        if self._continuous_stt:
            self._pipeline.on_processed_frame(self._on_processed_frame_for_stt)
        else:
            self._pipeline.on_speech_end(self._on_pipeline_speech_end)
            self._pipeline.on_vad_event(self._on_pipeline_vad_event)
            self._pipeline.on_speech_frame(self._on_pipeline_speech_frame)
        if config.diarization is not None:
            self._pipeline.on_speaker_change(self._on_pipeline_speaker_change)
        if config.dtmf is not None:
            self._pipeline.on_dtmf(self._on_pipeline_dtmf)
        if config.recorder is not None:
            self._pipeline.on_recording_started(self._on_pipeline_recording_started)
            self._pipeline.on_recording_stopped(self._on_pipeline_recording_stopped)

        # Barge-in from backend (transport-level)
        if VoiceCapability.BARGE_IN in backend.capabilities:
            backend.on_barge_in(self._on_backend_barge_in)

        # Out-of-band DTMF from backend (e.g. RFC 4733 via RTP)
        if VoiceCapability.DTMF_SIGNALING in backend.capabilities and hasattr(
            backend, "on_dtmf_received"
        ):
            backend.on_dtmf_received(self._on_pipeline_dtmf)

    def _on_audio_received(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Handle raw audio frame from backend — feed into pipeline."""
        if self._pipeline is not None:
            self._pipeline.process_frame(session, frame)

    def _on_pipeline_speech_frame(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Handle processed audio frame during speech — feed to STT stream.

        Frames are buffered (~200ms) before being sent to the queue to
        avoid per-frame resampling artifacts in providers that resample
        (e.g. 16kHz → 24kHz).
        """
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
                        # Keep latest partial as fallback final_text
                        state.final_text = result.text
                        self._schedule(
                            self._fire_partial_transcription_hook(session, result, room_id),
                            name=f"partial_stt:{session.id}",
                        )
            except Exception:
                logger.exception("STT stream error for session %s", session.id)
                state.error = True

        state = _STTStreamState(queue=queue, task=asyncio.Task.__new__(asyncio.Task))
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
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._stt:
            return
        room_id, _ = binding_info
        logger.info("Starting continuous STT for session %s", session.id)

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=500)
        state = _STTStreamState(queue=queue, task=asyncio.Task.__new__(asyncio.Task))
        self._stt_streams[session.id] = state

        async def run_continuous(state: _STTStreamState) -> None:
            while not state.cancelled:
                # Fresh queue + WebSocket per turn (avoids server-side overlap)
                state.queue = asyncio.Queue[Any](maxsize=500)
                state.frame_buffer.clear()
                cur_queue = state.queue

                async def audio_gen(
                    q: asyncio.Queue[Any] = cur_queue,
                ) -> AsyncIterator[AudioChunk]:
                    while True:
                        chunk = await q.get()
                        if chunk is None:
                            return
                        yield chunk

                try:
                    assert self._stt is not None
                    barge_in_fired = False
                    async for result in self._stt.transcribe_stream(audio_gen()):
                        if state.cancelled:
                            break
                        if result.is_final and result.text:
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
                                    decision = self._interruption_handler.evaluate(
                                        playback_position_ms=(playback.position_ms),
                                        speech_duration_ms=0,
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
                    await asyncio.sleep(1.0)

                if not state.cancelled:
                    logger.debug(
                        "STT stream cycle ended for %s, reconnecting",
                        session.id,
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
            playback = self._playing_sessions.get(session.id)
            if playback:
                logger.warning("Transcription during playback (likely echo): %r", text)
            logger.info("Transcription: %s", text)

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

    def _on_pipeline_speech_end(self, session: VoiceSession, audio: bytes) -> None:
        """Handle speech end from pipeline — fire hooks and transcribe."""
        # Signal end-of-audio to streaming STT (if active)
        stream_state = self._stt_streams.get(session.id)
        if stream_state is not None:
            # Flush remaining buffered frames before sending sentinel
            self._flush_stt_buffer(stream_state, session.id)
            try:
                stream_state.queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning("STT stream queue full on sentinel for %s", session.id)

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
        from roomkit.voice.pipeline.vad.base import VADEventType

        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        if vad_event.type == VADEventType.SPEECH_START:
            # Check for barge-in using InterruptionHandler
            playback = self._playing_sessions.get(session.id)
            if playback:
                decision = self._interruption_handler.evaluate(
                    playback_position_ms=playback.position_ms,
                    speech_duration_ms=0,
                )
                if decision.should_interrupt:
                    self._schedule(
                        self._handle_barge_in(session, playback, room_id),
                        name=f"barge_in:{session.id}",
                    )
                elif decision.is_backchannel:
                    self._schedule(
                        self._fire_backchannel_hook(session, "", room_id),
                        name=f"backchannel:{session.id}",
                    )

            # Start streaming STT if provider supports it
            if self._stt and self._stt.supports_streaming:
                self._start_stt_stream(session, room_id, pre_roll=vad_event.audio_bytes)

            self._schedule(
                self._fire_speech_start_hooks(session, room_id),
                name=f"speech_start:{session.id}",
            )
        elif vad_event.type == VADEventType.SPEECH_END:
            # ON_SPEECH_END hooks are fired by _process_speech_end() to
            # guarantee ordering (ON_SPEECH_END before ON_TRANSCRIPTION).
            # Do NOT fire them here to avoid duplicate invocations.
            pass
        elif vad_event.type == VADEventType.SILENCE:
            self._schedule(
                self._fire_vad_silence_hook(session, int(vad_event.duration_ms or 0), room_id),
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

    def _on_pipeline_dtmf(self, session: VoiceSession, dtmf_event: Any) -> None:
        """Handle DTMF event from pipeline — fire ON_DTMF hook."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        self._schedule(
            self._fire_dtmf_hook(session, dtmf_event, room_id),
            name=f"dtmf:{session.id}",
        )

    def _on_pipeline_recording_started(self, session: VoiceSession, handle: Any) -> None:
        """Handle recording started from pipeline — fire hook."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        self._schedule(
            self._fire_recording_started_hook(session, handle, room_id),
            name=f"recording_started:{session.id}",
        )

    def _on_pipeline_recording_stopped(self, session: VoiceSession, result: Any) -> None:
        """Handle recording stopped from pipeline — fire hook."""
        binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        self._schedule(
            self._fire_recording_stopped_hook(session, result, room_id),
            name=f"recording_stopped:{session.id}",
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
        # Notify pipeline of session activation
        if self._pipeline is not None:
            self._pipeline.on_session_active(session)
        # Start continuous STT if enabled (no local VAD)
        if self._continuous_stt:
            self._start_continuous_stt(session)
        # Emit voice_session_started framework event
        if self._framework:
            self._schedule(
                self._emit_session_started(session, room_id),
                name=f"session_started:{session.id}",
            )

    def unbind_session(self, session: VoiceSession) -> None:
        """Remove session binding."""
        # Cancel any active streaming STT
        self._cancel_stt_stream(session.id)
        binding_info = self._session_bindings.pop(session.id, None)
        # Notify pipeline of session end
        if self._pipeline is not None:
            self._pipeline.on_session_ended(session)
        # Clear pending turns
        self._pending_turns.pop(session.id, None)
        # Emit voice_session_ended framework event
        if binding_info and self._framework:
            room_id, _ = binding_info
            self._schedule(
                self._emit_session_ended(session, room_id),
                name=f"session_ended:{session.id}",
            )

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

    @property
    def supports_streaming_delivery(self) -> bool:
        """Whether this channel can accept streaming text delivery."""
        return (
            self._tts is not None
            and getattr(self._tts, "supports_streaming_input", False)
            and self._backend is not None
        )

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.TEXT],
            supports_audio=True,
            supported_audio_formats=["wav", "mp3", "ogg", "webm"],
            max_audio_duration_seconds=3600,
        )

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

    # -------------------------------------------------------------------------
    # Barge-in handling
    # -------------------------------------------------------------------------

    async def _handle_barge_in(
        self, session: VoiceSession, playback: TTSPlaybackState, room_id: str
    ) -> None:
        if not self._framework:
            return
        # Cancel streaming STT on barge-in (skip in continuous mode —
        # the long-lived stream must stay open)
        if not self._continuous_stt:
            self._cancel_stt_stream(session.id)
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
        """Process speech end: fire hooks, transcribe, route inbound.

        ON_SPEECH_END hooks are fired here (not in _on_pipeline_vad_event)
        to guarantee ordering: ON_SPEECH_END always fires before
        ON_TRANSCRIPTION and before routing to the AI.
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
            stream_state = self._stt_streams.pop(session.id, None)
            if stream_state is not None and not stream_state.error and not stream_state.cancelled:
                try:
                    await asyncio.wait_for(stream_state.task, timeout=5.0)
                    text = stream_state.final_text
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
                await self._evaluate_turn(session, final_text, room_id, context)
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

    # -------------------------------------------------------------------------
    # Turn detection
    # -------------------------------------------------------------------------

    async def _evaluate_turn(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        context: RoomContext,
    ) -> None:
        """Evaluate turn completion using the configured TurnDetector."""
        if not self._framework or not self._pipeline_config:
            return
        turn_detector = self._pipeline_config.turn_detector
        if turn_detector is None:
            await self._route_text(session, text, room_id)
            return

        from roomkit.voice.pipeline.turn.base import TurnContext, TurnEntry

        # Accumulate entry
        entries = self._pending_turns.setdefault(session.id, [])
        entries.append(TurnEntry(text=text, role="user"))

        turn_ctx = TurnContext(
            conversation_history=list(entries),
            silence_duration_ms=0.0,
            transcript=text,
            is_final=True,
            session_id=session.id,
        )
        decision = turn_detector.evaluate(turn_ctx)

        if decision.is_complete:
            # Fire ON_TURN_COMPLETE hook
            combined = " ".join(e.text for e in entries)
            self._pending_turns.pop(session.id, None)
            try:
                from roomkit.voice.events import TurnCompleteEvent

                event = TurnCompleteEvent(
                    session=session,
                    text=combined,
                    confidence=decision.confidence,
                )
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_TURN_COMPLETE,
                    event,
                    context,
                    skip_event_filter=True,
                )
            except Exception:
                logger.exception("Error firing ON_TURN_COMPLETE hook")

            await self._route_text(session, combined, room_id)
        else:
            # Fire ON_TURN_INCOMPLETE hook
            combined_so_far = " ".join(e.text for e in entries)
            try:
                from roomkit.voice.events import TurnIncompleteEvent

                incomplete_event = TurnIncompleteEvent(
                    session=session,
                    text=combined_so_far,
                    confidence=decision.confidence,
                )
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_TURN_INCOMPLETE,
                    incomplete_event,
                    context,
                    skip_event_filter=True,
                )
            except Exception:
                logger.exception("Error firing ON_TURN_INCOMPLETE hook")

    async def _route_text(self, session: VoiceSession, text: str, room_id: str) -> None:
        """Route transcribed text through the inbound pipeline."""
        if not self._framework:
            return
        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import TextContent

        inbound = InboundMessage(
            channel_id=self.channel_id,
            sender_id=session.participant_id,
            content=TextContent(body=text),
            metadata={"voice_session_id": session.id, "source": "voice"},
        )
        await self._framework.process_inbound(inbound)

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
                timestamp_ms=processed.timestamp_ms,
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
                    self._playing_sessions.pop(session.id, None)

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

    async def close(self) -> None:
        if self._stt:
            await self._stt.close()
        if self._tts:
            await self._tts.close()
        if self._pipeline is not None:
            self._pipeline.close()
        if self._backend:
            await self._backend.close()
        for sid in list(self._stt_streams):
            self._cancel_stt_stream(sid)
        self._session_bindings.clear()
        self._playing_sessions.clear()
