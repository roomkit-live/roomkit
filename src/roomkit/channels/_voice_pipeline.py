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
from roomkit.voice.pipeline.engine import AudioPipeline

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

        # Backend delivers raw AudioFrame → pipeline processes it
        backend.on_audio_received(self._pipeline_on_audio_received)

        # Wire speaker output → pipeline AEC for time-aligned reference.
        # Only when the backend doesn't already feed AEC at transport level.
        if (
            config.aec is not None
            and backend.supports_playback_callback
            and not backend.feeds_aec_reference
        ):

            def _on_audio_played(session: VoiceSession, frame: AudioFrame) -> None:
                if self._pipeline is not None:
                    self._pipeline.feed_aec_reference(frame)

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

        if self._pipeline is not None:
            self._pipeline.process_inbound(session, frame)

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
        and cleans up per-session state.
        """
        if self._pipeline is not None:
            self._pipeline.on_session_ended(session)
