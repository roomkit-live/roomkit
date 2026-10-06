"""Shared pipeline infrastructure for voice channels.

This mixin provides AudioPipeline creation, inbound audio gating,
AEC reference wiring, and session lifecycle management.  Both
VoiceChannel and RealtimeVoiceChannel inherit this to ensure the
pipeline is owned and managed identically.

Channel-specific concerns (VAD handling, STT, bridge, audio level
hooks) are NOT part of this mixin — each channel registers its own
callbacks on the pipeline after creation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.enums import Access
from roomkit.voice.base import VoiceCapability
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.offload import InboundFrameOffload

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig

logger = logging.getLogger("roomkit.channels.voice_pipeline")


@runtime_checkable
class PipelineHost(Protocol):
    """Contract: capabilities a host class must provide for VoicePipelineMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_bindings: Maps session IDs to binding info.  Format varies:
            VoiceChannel uses ``dict[str, tuple[str, ChannelBinding]]``,
            RealtimeVoiceChannel uses ``dict[str, ChannelBinding]``.
            Channels with non-default formats override
            :meth:`~VoicePipelineMixin._pipeline_on_audio_received`.
        _pipeline: The active audio pipeline instance (set by the mixin).
    """

    _state_lock: threading.Lock
    _session_bindings: dict[str, Any]
    _pipeline: AudioPipeline | None


class VoicePipelineMixin:
    """Pipeline infrastructure shared between VoiceChannel and RealtimeVoiceChannel.

    Host contract: :class:`PipelineHost`.
    """

    _state_lock: threading.Lock
    # Format varies: VoiceChannel uses dict[str, tuple[str, ChannelBinding]],
    # RealtimeVoiceChannel uses dict[str, ChannelBinding].  Channels that don't
    # match the default format should override _pipeline_on_audio_received.
    _session_bindings: dict[str, Any]
    _pipeline: AudioPipeline | None
    # Class-level default so channels that never build a pipeline still have
    # the attribute; _create_pipeline sets the instance one from the config.
    _inbound_offload: InboundFrameOffload | None = None

    def _create_pipeline(
        self,
        config: AudioPipelineConfig,
        backend: VoiceBackend,
    ) -> AudioPipeline:
        """Create an AudioPipeline and wire common infrastructure.

        Creates the pipeline, wires the backend's raw audio delivery to
        :meth:`_pipeline_on_audio_received`, and sets up AEC reference
        feeding from the backend's speaker playback callback when
        applicable.

        Returns the created pipeline.  The caller should register
        channel-specific callbacks (VAD, STT, bridge, audio levels)
        on the returned pipeline.
        """
        pipeline = AudioPipeline(
            config,
            backend_capabilities=backend.capabilities,
            backend_feeds_aec_reference=backend.feeds_aec_reference,
        )
        self._pipeline = pipeline
        threads = config.inbound_dsp_threads
        self._inbound_offload = InboundFrameOffload(threads) if threads else None

        # Backend delivers raw AudioFrame → pipeline processes it
        self._pipeline_unsubscribers = [
            backend.on_audio_received(self._pipeline_on_audio_received)
        ]

        # Wire speaker output → pipeline AEC for time-aligned reference.
        # Only when the backend doesn't already feed AEC at transport level.
        if (
            config.aec is not None
            and backend.supports_playback_callback
            and not backend.feeds_aec_reference
            and VoiceCapability.NATIVE_AEC not in backend.capabilities
        ):

            def _on_audio_played(session: VoiceSession, frame: AudioFrame) -> None:
                if self._pipeline is None:
                    return
                # Timeline pairing: while capture is paused (session mute,
                # gating, half-duplex) the backend drops mic frames, so the
                # reference must pause too — feeding it alone desyncs AEC3's
                # render/capture alignment by the mute's full duration
                # (measured: a 6 s mute left the filter cancelling against
                # audio the capture never saw, then a false barge-in).  The
                # backend keeps broadcasting the frames because playback
                # genuinely continues — levels and position stay live — and
                # states the pause in metadata for this consumer to honour.
                if not frame.metadata.get("capture_paused"):
                    self._pipeline.feed_aec_reference(frame, session.id)
                if frame.metadata.get("playback_ended"):
                    self._pipeline.set_aec_active(session.id, False)

            self._pipeline_unsubscribers.append(backend.on_audio_played(_on_audio_played))
            pipeline.enable_playback_aec_feed()

        # A transport that cancels echo itself shows its AEC to the debug taps
        # (RFC §12.3.15): the pipeline's raw is already echo-cancelled.
        taps = config.debug_taps
        if taps is not None and taps.output_dir and backend.feeds_aec_reference:

            def _on_aec_tap(session: VoiceSession, stage: str, frame: AudioFrame) -> None:
                if self._pipeline is not None:
                    self._pipeline.debug_tap(session.id, stage, frame)

            self._pipeline_unsubscribers.append(backend.on_aec_tap(_on_aec_tap))

        return pipeline

    def _pipeline_on_audio_received(
        self,
        session: VoiceSession,
        frame: AudioFrame,
    ) -> None:
        """Handle raw audio from backend — gate by binding, feed pipeline.

        Enforces ``ChannelBinding.access`` and ``muted`` per RFC S7.5:
        audio is dropped when the binding is READ_ONLY, NONE, or muted.
        """
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if binding_info is not None:
            binding = binding_info[1]
            if binding.access in (Access.READ_ONLY, Access.NONE) or binding.muted:
                return

        self._pipeline_submit_inbound(session, frame)

    def _pipeline_submit_inbound(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Feed one gated frame to the pipeline, inline or via the DSP pool.

        With ``AudioPipelineConfig.inbound_dsp_threads`` unset the stage
        chain runs on the caller's thread exactly as before. With a pool,
        the frame is queued FIFO under the session's stream and processed
        by one worker at a time — the RFC §12 stage order is untouched,
        only *where* the chain executes moves. The callbacks the chain
        fires do not move: ``AudioPipeline._fanout`` sends them back to
        the pipeline's home loop in firing order, because the channels'
        handlers are loop code (asyncio queues and tasks, which only work
        from the loop's thread).
        """
        pipeline = self._pipeline
        if pipeline is None:
            return
        offload = self._inbound_offload
        if offload is None:
            pipeline.process_inbound(session, frame)
        else:
            offload.submit(session.id, pipeline.process_inbound, session, frame)

    def _pipeline_audio_rate(self, session: VoiceSession) -> int:
        """Sample rate of the audio the pipeline hands this channel.

        The backend states its transport rate in ``input_sample_rate``; the
        pipeline's inbound resampler may change it before VAD, so a speech
        segment is labelled with what the pipeline says, never the transport.
        """
        transport_rate = int(session.metadata.get("input_sample_rate", 16000))
        if self._pipeline is None:
            return transport_rate
        return self._pipeline.inbound_sample_rate(transport_rate)

    def _pipeline_session_active(
        self, session: VoiceSession, *, parent_span: str | None = None
    ) -> None:
        """Activate a session in the pipeline, its speech segment spans under *parent_span*.

        Call this when a voice session starts (after binding or accepting).
        Starts recording, debug taps, and per-session state. The span is handed
        over after the activation, which clears what a previous session left
        under the same id, spans included.
        """
        if self._pipeline is None:
            return
        self._pipeline.on_session_active(session)
        if parent_span is not None:
            self._pipeline.set_parent_span(session.id, parent_span)

    def _pipeline_session_ending(self, session: VoiceSession) -> None:
        """Tell the pipeline a session's end has begun.

        Call this when a teardown starts and still has awaits ahead of it:
        audio arriving meanwhile is dropped, along with the frames the DSP
        pool still holds for the session. :meth:`_pipeline_session_ended`
        follows once the teardown is done.
        """
        if self._inbound_offload is not None:
            self._inbound_offload.release(session.id)
        if self._pipeline is not None:
            self._pipeline.on_session_ending(session)

    def _pipeline_session_ended(self, session: VoiceSession) -> None:
        """Notify the pipeline that a session has ended.

        Call this when a voice session disconnects.  Stops recording
        and cleans up per-session state. Frames still queued on the DSP
        pool for this session are dropped first — audio for a session
        that ended has nowhere to go.
        """
        if self._inbound_offload is not None:
            self._inbound_offload.release(session.id)
        if self._pipeline is not None:
            self._pipeline.on_session_ended(session)

    async def _pipeline_quiesce(self) -> None:
        """Bring the inbound audio path to rest. Call first in close().

        The backends stop handing frames to the pipeline, and the frames the
        DSP pool already holds are processed. Their callbacks run on the loop
        before this returns (they were queued ahead of its resumption), so a
        stream or task one of them opens is there for the teardown to sweep,
        and nothing reaches a session after it ended.
        """
        for unsubscribe in getattr(self, "_pipeline_unsubscribers", []):
            if unsubscribe is not None:
                unsubscribe()
        self._pipeline_unsubscribers = []
        if self._inbound_offload is not None:
            await asyncio.to_thread(self._pipeline_offload_shutdown)

    def _pipeline_offload_shutdown(self) -> None:
        """Drain and stop the DSP pool."""
        if self._inbound_offload is not None:
            self._inbound_offload.shutdown()
            self._inbound_offload = None
