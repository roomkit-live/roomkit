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
        backend.on_audio_received(self._pipeline_on_audio_received)

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

            backend.on_audio_played(_on_audio_played)
            pipeline.enable_playback_aec_feed()

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
        only *where* the chain executes moves. Sync pipeline callbacks
        run wherever the chain runs. An *async* callback's coroutine is
        sent to the pipeline's home loop by ``_maybe_schedule`` — a pool
        worker has no running loop, and before that fallback existed the
        coroutines (the realtime provider's audio feed, the audio-level
        hooks) were silently dropped while every sync path kept working.
        """
        pipeline = self._pipeline
        if pipeline is None:
            return
        offload = self._inbound_offload
        if offload is None:
            pipeline.process_inbound(session, frame)
        else:
            offload.submit(session.id, pipeline.process_inbound, session, frame)

    def _pipeline_session_active(self, session: VoiceSession) -> None:
        """Notify the pipeline that a session is active.

        Call this when a voice session starts (after binding or accepting).
        Starts recording, debug taps, and per-session state.
        """
        if self._pipeline is not None:
            self._pipeline.on_session_active(session)

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

    def _pipeline_offload_shutdown(self) -> None:
        """Drain and stop the DSP pool. Call from the channel's close()."""
        if self._inbound_offload is not None:
            self._inbound_offload.shutdown()
            self._inbound_offload = None
