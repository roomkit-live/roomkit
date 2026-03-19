"""VoiceOpsMixin — voice and video session lifecycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    from roomkit.models.event import AudioContent
    from roomkit.recorder._room_recorder_manager import RoomRecorderManager
    from roomkit.store.base import ConversationStore
    from roomkit.video.base import VideoSession
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import TranscriptionResult, VoiceSession
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.framework")


class VoiceOpsMixin(HelpersMixin):
    """Voice and video session connect/disconnect operations."""

    _store: ConversationStore
    _channels: dict[str, Channel]
    _voice: VoiceBackend | None
    _stt: STTProvider | None
    _tts: TTSProvider | None
    _room_recorder_mgr: RoomRecorderManager

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

        Creates a voice session via the configured VoiceBackend and binds it
        to the specified room and voice channel for message routing.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            channel_id: The voice channel ID.
            metadata: Optional session metadata.
            auto_greet: Deprecated. Use ``Agent(auto_greet=True)`` instead.
                When ``True``, emits a :class:`DeprecationWarning`.

        Returns:
            A VoiceSession representing the connection.

        Raises:
            VoiceBackendNotConfiguredError: If no voice backend is configured.
            ChannelNotRegisteredError: If the channel is not a VoiceChannel.
            RoomNotFoundError: If the room doesn't exist.
        """
        import warnings

        if self._voice is None:
            raise VoiceBackendNotConfiguredError("No voice backend configured")

        if auto_greet:
            warnings.warn(
                "connect_voice(auto_greet=True) is deprecated. "
                "Set auto_greet=True on the Agent instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Verify room exists
        await self.get_room(room_id)  # type: ignore[attr-defined]

        # Get the voice channel
        channel = self._channels.get(channel_id)
        if not isinstance(channel, VoiceChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered VoiceChannel"
            )

        # Get the binding
        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not attached to room {room_id}")

        # Create the session
        session = await self._voice.connect(room_id, participant_id, channel_id, metadata=metadata)

        # Bind session to channel for routing
        channel.bind_session(session, room_id, binding)
        self._wire_audio_recording(room_id, channel_id, session, channel)  # type: ignore[attr-defined]

        await self._emit_framework_event(
            "voice_session_started",
            room_id=room_id,
            channel_id=channel_id,
            data={
                "session_id": session.id,
                "participant_id": participant_id,
                "channel_id": channel_id,
            },
        )

        return session

    async def disconnect_voice(self, session: VoiceSession) -> None:
        """Disconnect a voice session.

        Args:
            session: The session to disconnect.

        Raises:
            VoiceBackendNotConfiguredError: If no voice backend is configured.
        """
        if self._voice is None:
            raise VoiceBackendNotConfiguredError("No voice backend configured")

        # Remove room-level recording tracks before unbind
        if self._room_recorder_mgr.has_recorders(session.room_id):
            audio_track = self._make_audio_track(  # type: ignore[attr-defined]
                session.id, session.channel_id, session.participant_id
            )
            self._room_recorder_mgr.on_track_removed(session.room_id, audio_track)

            # Combined A/V backend (SIPVideoBackend) — also remove video track
            video_track = self._make_video_track(  # type: ignore[attr-defined]
                session.id, session.channel_id, session.participant_id
            )
            self._room_recorder_mgr.on_track_removed(session.room_id, video_track)

        # Get the voice channel and unbind
        channel = self._channels.get(session.channel_id)
        if isinstance(channel, VoiceChannel):
            channel.unbind_session(session)

        await self._voice.disconnect(session)

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

    async def connect_video(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSession:
        """Connect a participant to a video session.

        Creates a video session via the channel's VideoBackend and binds
        it to the specified room and video channel for event routing.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            channel_id: The video channel ID.
            metadata: Optional session metadata.

        Returns:
            A VideoSession representing the connection.

        Raises:
            ChannelNotRegisteredError: If the channel is not a VideoChannel.
            RoomNotFoundError: If the room doesn't exist.
        """
        from roomkit.channels.video import VideoChannel

        await self.get_room(room_id)  # type: ignore[attr-defined]

        channel = self._channels.get(channel_id)
        # AudioVideoChannel sessions are bound via bind_voice_session,
        # not connect_video.  This path is for standalone VideoChannel only.
        if not isinstance(channel, VideoChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered VideoChannel"
            )

        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not attached to room {room_id}")

        session = await channel.backend.connect(
            room_id, participant_id, channel_id, metadata=metadata
        )
        channel.bind_session(session, room_id, binding)
        self._wire_video_recording(room_id, channel_id, session, channel)  # type: ignore[attr-defined]
        return session

    async def disconnect_video(self, session: VideoSession) -> None:
        """Disconnect a video session.

        Args:
            session: The session to disconnect.

        Raises:
            ChannelNotRegisteredError: If the channel is not a VideoChannel.
        """
        from roomkit.channels.video import VideoChannel

        channel = self._channels.get(session.channel_id)
        if isinstance(channel, VideoChannel):
            # Remove room-level recording track before unbind
            if self._room_recorder_mgr.has_recorders(session.room_id):
                track = self._make_video_track(  # type: ignore[attr-defined]
                    session.id, session.channel_id, session.participant_id
                )
                self._room_recorder_mgr.on_track_removed(session.room_id, track)
            channel.unbind_session(session)
            await channel.backend.disconnect(session)

    async def bind_voice_session(
        self,
        session: VoiceSession,
        room_id: str,
        channel_id: str,
    ) -> None:
        """Bind an externally-created voice session (push model).

        SIP ``on_call`` handlers should call this instead of
        ``channel.bind_session()`` directly — it wires recording taps
        for both audio and video (when the backend is a combined A/V
        backend like :class:`SIPVideoBackend`).

        Args:
            session: The voice session from the backend's on_call.
            room_id: The room to bind into.
            channel_id: The voice channel ID.

        Raises:
            ChannelNotRegisteredError: If the channel is not a VoiceChannel.
            ChannelNotFoundError: If the channel is not attached to the room.
        """
        channel = self._channels.get(channel_id)
        if not isinstance(channel, VoiceChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered VoiceChannel"
            )

        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not attached to room {room_id}")

        channel.bind_session(session, room_id, binding)
        self._wire_audio_recording(room_id, channel_id, session, channel)  # type: ignore[attr-defined]

        # AudioVideoChannel — wire video via channel tap
        from roomkit.channels.av import AudioVideoChannel

        if isinstance(channel, AudioVideoChannel):
            self._wire_av_video_recording(room_id, channel_id, session, channel)  # type: ignore[attr-defined]
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

        Creates a realtime voice session via the channel's provider and
        transport, binding it to the specified room.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            channel_id: The realtime voice channel ID.
            connection: Protocol-specific connection (e.g. WebSocket).
            metadata: Optional session metadata (may include overrides
                for system_prompt, voice, tools, temperature).

        Returns:
            A VoiceSession representing the connection.

        Raises:
            ChannelNotRegisteredError: If the channel is not a RealtimeVoiceChannel.
            RoomNotFoundError: If the room doesn't exist.
            ChannelNotFoundError: If the channel is not attached to the room.
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        # Verify room exists
        await self.get_room(room_id)  # type: ignore[attr-defined]

        # Get the realtime voice channel
        channel = self._channels.get(channel_id)
        if not isinstance(channel, RealtimeVoiceChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered RealtimeVoiceChannel"
            )

        # Verify binding exists
        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not attached to room {room_id}")

        return await channel.start_session(room_id, participant_id, connection, metadata=metadata)

    async def disconnect_realtime_voice(self, session: Any) -> None:
        """Disconnect a realtime voice session.

        Args:
            session: The VoiceSession to disconnect.

        Raises:
            ChannelNotRegisteredError: If the channel is not found.
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        channel = self._channels.get(session.channel_id)
        if isinstance(channel, RealtimeVoiceChannel):
            await channel.end_session(session)

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
