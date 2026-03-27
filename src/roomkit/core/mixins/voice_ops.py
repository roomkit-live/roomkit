"""VoiceOpsMixin — voice and video session lifecycle."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.channels.voice import VoiceChannel
from roomkit.core.exceptions import (
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    VoiceBackendNotConfiguredError,
    VoiceNotConfiguredError,
)
from roomkit.core.mixins.helpers import HelpersMixin

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.event import AudioContent
    from roomkit.recorder._room_recorder_manager import RoomRecorderManager
    from roomkit.store.base import ConversationStore
    from roomkit.video.base import VideoSession
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import TranscriptionResult, VoiceSession
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.framework")


@runtime_checkable
class VoiceOpsHost(Protocol):
    """Contract: capabilities a host class must provide for VoiceOpsMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _channels: Registry of channel-id to :class:`Channel` instances.
        _voice: Default voice backend (or ``None``).
        _stt: Default speech-to-text provider (or ``None``).
        _tts: Default text-to-speech provider (or ``None``).
        _room_recorder_mgr: Manager for room-level media recording.

    Cross-mixin methods (provided by other mixins in the MRO):
        get_room: From :class:`RoomLifecycleMixin`.
        _wire_audio_recording: From :class:`RecordingMixin`.
        _wire_av_video_recording: From :class:`RecordingMixin`.
        _wire_backend_video_recording: From :class:`RecordingMixin`.
        _wire_video_recording: From :class:`RecordingMixin`.
        _make_audio_track: From :class:`RecordingMixin`.
        _make_video_track: From :class:`RecordingMixin`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _voice: VoiceBackend | None
    _stt: STTProvider | None
    _tts: TTSProvider | None
    _room_recorder_mgr: RoomRecorderManager


class VoiceOpsMixin(HelpersMixin):
    """Voice and video session connect/disconnect operations.

    Host contract: :class:`VoiceOpsHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _voice: VoiceBackend | None
    _stt: STTProvider | None
    _tts: TTSProvider | None
    _room_recorder_mgr: RoomRecorderManager

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # see VoiceOpsHost
    _wire_audio_recording: Any  # see VoiceOpsHost
    _wire_av_video_recording: Any  # see VoiceOpsHost
    _wire_backend_video_recording: Any  # see VoiceOpsHost
    _wire_video_recording: Any  # see VoiceOpsHost
    _make_audio_track: Any  # see VoiceOpsHost
    _make_video_track: Any  # see VoiceOpsHost

    # ------------------------------------------------------------------
    # join / leave — unified session lifecycle
    # ------------------------------------------------------------------

    async def join(
        self,
        room_id: str,
        channel_id: str,
        *,
        session: VoiceSession | VideoSession | None = None,
        participant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        backend: VoiceBackend | None = None,
        connection: Any = None,
    ) -> VoiceSession | VideoSession:
        """Join a participant to a room via a channel.

        **Pull model** (``session=None``): the framework creates a session
        via the channel's backend, binds it to the room, wires recording,
        and starts listening.  ``participant_id`` is auto-generated if
        omitted.

        **Push model** (``session=<externally created>``): the framework
        binds the existing session and wires recording.  Used by SIP
        ``on_call`` handlers.

        Args:
            room_id: The room to join.
            channel_id: The channel to join through.
            session: Externally-created session (push model).
                Omit for pull model.
            participant_id: Participant ID.  Auto-generated if omitted.
            metadata: Optional session metadata.
            backend: Override backend for cross-transport bridging.
                When bridging sessions from different transports
                (e.g. SIP + WebRTC), pass the session's own backend
                so the bridge sends audio through the correct transport.
            connection: Protocol-specific connection for realtime voice
                channels (e.g. WebSocket).  Required when joining a
                :class:`RealtimeVoiceChannel`.

        Returns:
            The voice or video session (created or passed in).

        Raises:
            RoomNotFoundError: If the room does not exist.
            ChannelNotRegisteredError: If the channel is not registered.
            ChannelNotFoundError: If the channel is not attached to the room.
        """
        await self.get_room(room_id)
        channel = self._channels.get(channel_id)
        if channel is None:
            raise ChannelNotRegisteredError(f"Channel {channel_id} not registered")

        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not attached to room {room_id}")

        if isinstance(channel, VoiceChannel):
            return await self._join_voice(
                room_id,
                channel_id,
                channel,
                binding,
                session=session,  # type: ignore[arg-type]
                participant_id=participant_id,
                metadata=metadata,
                backend=backend,
            )

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        if isinstance(channel, RealtimeVoiceChannel):
            pid = participant_id or f"participant-{uuid4().hex[:8]}"
            return await channel.start_session(
                room_id,
                pid,
                connection,
                metadata=metadata,
            )

        from roomkit.channels.video import VideoChannel

        if isinstance(channel, VideoChannel):
            return await self._join_video(
                room_id,
                channel_id,
                channel,
                binding,
                session=session,
                participant_id=participant_id,
                metadata=metadata,
            )

        raise ChannelNotRegisteredError(
            f"Channel {channel_id} ({type(channel).__name__}) does not support join()"
        )

    async def _join_voice(
        self,
        room_id: str,
        channel_id: str,
        channel: VoiceChannel,
        binding: ChannelBinding,
        *,
        session: VoiceSession | None = None,
        participant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        backend: VoiceBackend | None = None,
    ) -> VoiceSession:
        """Join a voice session to a room (internal)."""
        created = session is None
        if created:
            create_backend = backend or channel._backend
            if create_backend is None:
                raise VoiceBackendNotConfiguredError("VoiceChannel has no backend configured")
            pid = participant_id or f"participant-{uuid4().hex[:8]}"
            session = await create_backend.connect(
                room_id,
                pid,
                channel_id,
                metadata=metadata,
            )

        if session is None:
            raise RuntimeError("Voice session was not created by the backend")

        channel.bind_session(session, room_id, binding, backend=backend)
        self._wire_audio_recording(room_id, channel_id, session, channel)
        # AudioVideoChannel — wire video via channel tap
        from roomkit.channels.av import AudioVideoChannel

        if isinstance(channel, AudioVideoChannel):
            self._wire_av_video_recording(room_id, channel_id, session, channel)
        else:
            # Legacy fallback: plain VoiceChannel with combined A/V backend
            from roomkit.video.backends.base import VideoBackend

            if isinstance(channel._backend, VideoBackend) and not (
                channel._recording is not None and not channel._recording.video
            ):
                video_session = channel._backend.get_video_session(session.id)
                if video_session is not None:
                    self._wire_backend_video_recording(
                        room_id,
                        channel_id,
                        session,
                        channel._backend,
                    )

        # Pull model: start listening (push model backends stream already)
        backend_ref = channel._backend
        if created and backend_ref is not None and hasattr(backend_ref, "start_listening"):
            await backend_ref.start_listening(session)

        await self._emit_framework_event(
            "voice_session_started",
            room_id=room_id,
            channel_id=channel_id,
            data={
                "session_id": session.id,
                "participant_id": session.participant_id,
                "channel_id": channel_id,
            },
        )
        return session

    async def _join_video(
        self,
        room_id: str,
        channel_id: str,
        channel: Any,
        binding: ChannelBinding,
        *,
        session: Any = None,
        participant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSession:
        """Join a video session to a room (internal)."""
        created = session is None
        if created:
            pid = participant_id or f"participant-{uuid4().hex[:8]}"
            session = await channel.backend.connect(
                room_id,
                pid,
                channel_id,
                metadata=metadata,
            )

        channel.bind_session(session, room_id, binding)
        self._wire_video_recording(room_id, channel_id, session, channel)
        return session  # type: ignore[no-any-return]

    async def leave(self, session: VoiceSession | VideoSession) -> None:
        """Remove a participant from a room.

        Stops listening, unbinds the session, removes recording tracks,
        and disconnects the backend.

        Args:
            session: The session to leave.
        """
        channel = self._channels.get(session.channel_id)

        if isinstance(channel, VoiceChannel):
            await self._leave_voice(session, channel)  # type: ignore[arg-type]
            return

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        if isinstance(channel, RealtimeVoiceChannel):
            await channel.end_session(session)  # type: ignore[arg-type]
            return

        from roomkit.channels.video import VideoChannel

        if isinstance(channel, VideoChannel):
            await self._leave_video(session, channel)
            return

        channel_type = type(channel).__name__ if channel is not None else "None"
        logger.warning(
            "leave() called for unhandled channel %r (%s) (session %s)",
            session.channel_id,
            channel_type,
            session.id,
        )

    async def _leave_voice(self, session: VoiceSession, channel: VoiceChannel) -> None:
        """Leave a voice session (internal)."""
        # Remove recording tracks
        if self._room_recorder_mgr.has_recorders(session.room_id):
            audio_track = self._make_audio_track(
                session.id, session.channel_id, session.participant_id
            )
            self._room_recorder_mgr.on_track_removed(session.room_id, audio_track)

            video_track = self._make_video_track(
                session.id, session.channel_id, session.participant_id
            )
            self._room_recorder_mgr.on_track_removed(session.room_id, video_track)

        channel.unbind_session(session)

        backend = channel._backend
        if backend:
            if hasattr(backend, "stop_listening"):
                await backend.stop_listening(session)
            await backend.disconnect(session)

        await self._emit_framework_event(
            "voice_session_ended",
            room_id=session.room_id,
            channel_id=session.channel_id,
            data={
                "session_id": session.id,
                "participant_id": session.participant_id,
                "channel_id": session.channel_id,
            },
        )

    async def _leave_video(self, session: Any, channel: Any) -> None:
        """Leave a video session (internal)."""
        if self._room_recorder_mgr.has_recorders(session.room_id):
            track = self._make_video_track(session.id, session.channel_id, session.participant_id)
            self._room_recorder_mgr.on_track_removed(session.room_id, track)

        channel.unbind_session(session)
        await channel.backend.disconnect(session)

    # ------------------------------------------------------------------
    # Deprecated methods — delegate to join/leave
    # ------------------------------------------------------------------

    async def connect_voice(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        auto_greet: bool = False,
    ) -> VoiceSession:
        """Connect a participant to a voice session.

        .. deprecated::
            Use :meth:`join` instead::

                session = await kit.join(room_id, channel_id,
                                         participant_id=participant_id)
        """
        warnings.warn(
            "connect_voice() is deprecated. Use kit.join() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if auto_greet:
            warnings.warn(
                "connect_voice(auto_greet=True) is deprecated. "
                "Set auto_greet=True on the Agent instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return await self.join(  # type: ignore[return-value]
            room_id,
            channel_id,
            participant_id=participant_id,
            metadata=metadata,
        )

    async def disconnect_voice(self, session: VoiceSession) -> None:
        """Disconnect a voice session.

        .. deprecated::
            Use :meth:`leave` instead::

                await kit.leave(session)
        """
        warnings.warn(
            "disconnect_voice() is deprecated. Use kit.leave() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        await self.leave(session)

    async def connect_video(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSession:
        """Connect a participant to a video session.

        .. deprecated::
            Use :meth:`join` instead.
        """
        warnings.warn(
            "connect_video() is deprecated. Use kit.join() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.join(  # type: ignore[return-value]
            room_id,
            channel_id,
            participant_id=participant_id,
            metadata=metadata,
        )

    async def disconnect_video(self, session: VideoSession) -> None:
        """Disconnect a video session.

        .. deprecated::
            Use :meth:`leave` instead.
        """
        warnings.warn(
            "disconnect_video() is deprecated. Use kit.leave() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        await self.leave(session)

    async def bind_voice_session(
        self,
        session: VoiceSession,
        room_id: str,
        channel_id: str,
    ) -> None:
        """Bind an externally-created voice session (push model).

        .. deprecated::
            Use :meth:`join` instead::

                await kit.join(room_id, channel_id, session=session)
        """
        warnings.warn(
            "bind_voice_session() is deprecated. Use kit.join() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        await self.join(room_id, channel_id, session=session)

    async def connect_realtime_voice(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        connection: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Connect a participant to a realtime voice session.

        .. deprecated::
            Use :meth:`join` instead::

                session = await kit.join(room_id, channel_id,
                                         participant_id=participant_id,
                                         connection=websocket)
        """
        warnings.warn(
            "connect_realtime_voice() is deprecated. Use kit.join() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.join(
            room_id,
            channel_id,
            participant_id=participant_id,
            connection=connection,
            metadata=metadata,
        )

    async def disconnect_realtime_voice(self, session: Any) -> None:
        """Disconnect a realtime voice session.

        .. deprecated::
            Use :meth:`leave` instead.
        """
        warnings.warn(
            "disconnect_realtime_voice() is deprecated. Use kit.leave() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        await self.leave(session)

    async def transcribe(self, audio: AudioContent) -> TranscriptionResult:
        """Transcribe audio to text using configured STT provider.

        Args:
            audio: AudioContent with URL to audio file.

        Returns:
            TranscriptionResult with text and metadata.

        Raises:
            VoiceNotConfiguredError: If no STT provider is configured.
        """
        if self._stt is None:
            raise VoiceNotConfiguredError("No STT provider configured")
        return await self._stt.transcribe(audio)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio using configured TTS provider.

        Args:
            text: Text to synthesize.
            voice: Optional voice ID (uses provider default if not specified).

        Returns:
            AudioContent with URL to generated audio.

        Raises:
            VoiceNotConfiguredError: If no TTS provider is configured.
        """
        if self._tts is None:
            raise VoiceNotConfiguredError("No TTS provider configured")
        return await self._tts.synthesize(text, voice=voice)
