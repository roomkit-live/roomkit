"""RoomKit - central orchestrator for multi-channel conversations."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.models.delivery import InboundMessage, InboundResult
    from roomkit.models.event import AudioContent
    from roomkit.providers.sms.meta import WebhookMeta
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.telemetry.config import TelemetryConfig
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import TranscriptionResult, VoiceSession
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

from roomkit.channels.base import Channel
from roomkit.channels.voice import VoiceChannel
from roomkit.channels.websocket import SendFn, WebSocketChannel
from roomkit.core._channel_ops import ChannelOpsMixin
from roomkit.core._helpers import FrameworkEventHandler, HelpersMixin, IdentityHookFn
from roomkit.core._inbound import InboundMixin
from roomkit.core._room_lifecycle import RoomLifecycleMixin
from roomkit.core.event_router import EventRouter
from roomkit.core.hooks import (
    AsyncHookFn,
    HookEngine,
    HookRegistration,
    IdentityHookRegistration,
    SyncHookFn,
)
from roomkit.core.inbound_router import DefaultInboundRoomRouter, InboundRoomRouter
from roomkit.core.locks import InMemoryLockManager, RoomLockManager
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.identity.base import IdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import DeliveryStatus
from roomkit.models.enums import (
    ChannelDirection,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    RoomStatus,
)
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.task import Observation, Task
from roomkit.realtime.base import (
    EphemeralCallback,
    EphemeralEvent,
    EphemeralEventType,
    RealtimeBackend,
)
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.sources.base import SourceHealth, SourceProvider, SourceStatus
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore

# Re-export type aliases so existing imports continue to work
__all__ = [
    "ChannelNotFoundError",
    "ChannelNotRegisteredError",
    "FrameworkEventHandler",
    "IdentityHookFn",
    "IdentityNotFoundError",
    "ParticipantNotFoundError",
    "RoomKit",
    "RoomKitError",
    "RoomNotFoundError",
    "SourceAlreadyAttachedError",
    "SourceNotFoundError",
    "VoiceBackendNotConfiguredError",
    "VoiceNotConfiguredError",
]


class RoomKitError(Exception):
    """Base exception for all RoomKit errors."""


class RoomNotFoundError(RoomKitError):
    """Room does not exist."""


class ChannelNotFoundError(RoomKitError):
    """Channel binding not found in room."""


class ChannelNotRegisteredError(RoomKitError):
    """Channel type not registered."""


class ParticipantNotFoundError(RoomKitError):
    """Participant not found in room."""


class IdentityNotFoundError(RoomKitError):
    """Identity not found."""


class SourceAlreadyAttachedError(RoomKitError):
    """Source already attached to channel."""


class SourceNotFoundError(RoomKitError):
    """Source not found for channel."""


class VoiceNotConfiguredError(RoomKitError):
    """Raised when voice operation attempted without configured provider."""


class VoiceBackendNotConfiguredError(RoomKitError):
    """Raised when voice backend operation attempted without configured backend."""


class RoomKit(InboundMixin, ChannelOpsMixin, RoomLifecycleMixin, HelpersMixin):
    """Central orchestrator tying rooms, channels, hooks, and storage."""

    def __init__(
        self,
        store: ConversationStore | None = None,
        identity_resolver: IdentityResolver | None = None,
        identity_channel_types: set[ChannelType] | None = None,
        inbound_router: InboundRoomRouter | None = None,
        lock_manager: RoomLockManager | None = None,
        realtime: RealtimeBackend | None = None,
        max_chain_depth: int = 5,
        identity_timeout: float = 10.0,
        process_timeout: float = 30.0,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        voice: VoiceBackend | None = None,
        telemetry: TelemetryConfig | TelemetryProvider | None = None,
    ) -> None:
        """Initialise the RoomKit orchestrator.

        Args:
            store: Persistent storage backend. Defaults to ``InMemoryStore``.
            identity_resolver: Optional resolver for identifying inbound senders.
            identity_channel_types: Restrict identity resolution to specific channel
                types. If ``None`` (default), resolution runs for all channels.
                Set to e.g. ``{ChannelType.SMS}`` to only resolve identity for SMS.
            inbound_router: Strategy for routing inbound messages to rooms.
                Defaults to ``DefaultInboundRoomRouter``.
            lock_manager: Per-room locking backend. Defaults to
                ``InMemoryLockManager``.  For multi-process deployments,
                supply a distributed implementation (e.g. Redis-backed).
            realtime: Realtime backend for ephemeral events (typing, presence).
                Defaults to ``InMemoryRealtime``. For multi-process deployments,
                supply a distributed implementation (e.g. Redis pub/sub).
            max_chain_depth: Maximum reentry chain depth to prevent infinite loops.
            identity_timeout: Timeout in seconds for identity resolution calls.
            process_timeout: Timeout in seconds for the locked processing phase.
            stt: Optional speech-to-text provider for transcription.
            tts: Optional text-to-speech provider for synthesis.
            voice: Optional voice backend for real-time audio transport.
            telemetry: Optional telemetry provider or config for span/metric
                collection. Accepts a ``TelemetryProvider`` instance or a
                ``TelemetryConfig``. Defaults to ``NoopTelemetryProvider``.
        """
        from roomkit.telemetry.base import TelemetryProvider as _TelemetryProviderCls
        from roomkit.telemetry.config import TelemetryConfig as _TelemetryConfigCls
        from roomkit.telemetry.noop import NoopTelemetryProvider

        self._store = store or InMemoryStore()
        self._identity_resolver = identity_resolver
        self._identity_channel_types = identity_channel_types
        self._max_chain_depth = max_chain_depth
        self._identity_timeout = identity_timeout
        self._process_timeout = process_timeout
        self._channels: dict[str, Channel] = {}
        self._hook_engine = HookEngine()
        self._lock_manager = lock_manager or InMemoryLockManager()
        self._realtime = realtime or InMemoryRealtime()
        self._transcoder = DefaultContentTranscoder()
        self._event_handlers: list[tuple[str, FrameworkEventHandler]] = []
        self._identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]] = {}
        self._inbound_router = inbound_router or DefaultInboundRoomRouter(self._store)
        self._event_router: EventRouter | None = None
        # Event-driven sources
        self._sources: dict[str, SourceProvider] = {}
        self._source_tasks: dict[str, asyncio.Task[None]] = {}
        # Voice support
        self._stt = stt
        self._tts = tts
        self._voice = voice
        # Traces received before the room exists (flushed on attach_channel)
        self._pending_traces: dict[str, list[object]] = {}
        # Telemetry
        if isinstance(telemetry, _TelemetryProviderCls):
            self._telemetry: _TelemetryProviderCls = telemetry
        elif isinstance(telemetry, _TelemetryConfigCls):
            self._telemetry = telemetry.provider or NoopTelemetryProvider()
        else:
            self._telemetry = NoopTelemetryProvider()
        self._hook_engine._telemetry = self._telemetry
        self._store._telemetry = self._telemetry  # type: ignore[attr-defined]

    @property
    def store(self) -> ConversationStore:
        """The backing conversation store."""
        return self._store

    @property
    def hook_engine(self) -> HookEngine:
        """The hook engine used for sync/async hook pipelines."""
        return self._hook_engine

    @property
    def realtime(self) -> RealtimeBackend:
        """The realtime backend for ephemeral events."""
        return self._realtime

    @property
    def stt(self) -> STTProvider | None:
        """Speech-to-text provider (optional)."""
        return self._stt

    @property
    def tts(self) -> TTSProvider | None:
        """Text-to-speech provider (optional)."""
        return self._tts

    @property
    def voice(self) -> VoiceBackend | None:
        """Voice backend for real-time audio (optional)."""
        return self._voice

    @property
    def telemetry(self) -> TelemetryProvider:
        """The telemetry provider for span and metric collection."""
        return self._telemetry

    async def connect_voice(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        """Connect a participant to a voice session.

        Creates a voice session via the configured VoiceBackend and binds it
        to the specified room and voice channel for message routing.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            channel_id: The voice channel ID.
            metadata: Optional session metadata.

        Returns:
            A VoiceSession representing the connection.

        Raises:
            VoiceBackendNotConfiguredError: If no voice backend is configured.
            ChannelNotRegisteredError: If the channel is not a VoiceChannel.
            RoomNotFoundError: If the room doesn't exist.
        """
        if self._voice is None:
            raise VoiceBackendNotConfiguredError("No voice backend configured")

        # Verify room exists
        await self.get_room(room_id)

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
            A RealtimeSession representing the connection.

        Raises:
            ChannelNotRegisteredError: If the channel is not a RealtimeVoiceChannel.
            RoomNotFoundError: If the room doesn't exist.
            ChannelNotFoundError: If the channel is not attached to the room.
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        # Verify room exists
        await self.get_room(room_id)

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
            session: The RealtimeSession to disconnect.

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

    def _get_router(self) -> EventRouter:
        if self._event_router is None:
            self._event_router = EventRouter(
                channels=self._channels,
                transcoder=self._transcoder,
                max_chain_depth=self._max_chain_depth,
                telemetry=self._telemetry,
            )
        return self._event_router

    async def close(self) -> None:
        """Close all sources, channels, voice backend, and the realtime backend."""
        # Stop all event sources first
        for channel_id in list(self._sources.keys()):
            await self.detach_source(channel_id)
        # Then close channels
        for channel in self._channels.values():
            await channel.close()
        # Close voice backend
        if self._voice:
            await self._voice.close()
        await self._realtime.close()

    async def __aenter__(self) -> RoomKit:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- Queries --

    async def get_timeline(
        self,
        room_id: str,
        offset: int = 0,
        limit: int = 50,
        visibility_filter: str | None = None,
    ) -> list[RoomEvent]:
        """Query the event timeline for a room."""
        await self.get_room(room_id)
        return await self._store.list_events(
            room_id,
            offset=offset,
            limit=limit,
            visibility_filter=visibility_filter,
        )

    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        """List tasks for a room, optionally filtered by status."""
        return await self._store.list_tasks(room_id, status=status)

    async def list_observations(self, room_id: str) -> list[Observation]:
        """List observations for a room."""
        return await self._store.list_observations(room_id)

    # -- Direct send --

    async def send_event(
        self,
        room_id: str,
        channel_id: str,
        content: Any,
        event_type: EventType = EventType.MESSAGE,
        chain_depth: int = 0,
        participant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        visibility: str = "all",
        provider: str | None = None,
    ) -> RoomEvent:
        """Send an event directly into a room from a channel.

        Args:
            room_id: Target room ID
            channel_id: Source channel ID
            content: Event content (TextContent, RichContent, etc.)
            event_type: Type of event (default MESSAGE)
            chain_depth: Depth in response chain (for loop prevention)
            participant_id: Optional participant/sender ID for the event source
            metadata: Optional event metadata
            visibility: Event visibility ("all" or "internal")
            provider: Optional provider/backend name for event attribution
        """
        await self.get_room(room_id)
        binding = await self._get_binding(room_id, channel_id)

        event = RoomEvent(
            room_id=room_id,
            type=event_type,
            source=EventSource(
                channel_id=channel_id,
                channel_type=binding.channel_type,
                participant_id=participant_id,
                provider=provider,
            ),
            content=content,
            chain_depth=chain_depth,
            status=EventStatus.DELIVERED,
            metadata=metadata or {},
            visibility=visibility,
        )

        async with self._lock_manager.locked(room_id):
            count = await self._store.get_event_count(room_id)
            event = event.model_copy(update={"index": count})
            await self._store.add_event(event)

            context = await self._build_context(room_id)
            router = self._get_router()
            await router.broadcast(event, binding, context)

            # Run AFTER_BROADCAST hooks for observability and fan-out
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.AFTER_BROADCAST, event, context
            )

        return event

    # -- WebSocket lifecycle --

    async def connect_websocket(
        self, channel_id: str, connection_id: str, send_fn: SendFn
    ) -> None:
        """Register a WebSocket connection and emit framework event."""
        channel = self._channels.get(channel_id)
        if not isinstance(channel, WebSocketChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered WebSocket channel"
            )
        channel.register_connection(connection_id, send_fn)
        await self._emit_framework_event(
            "channel_connected",
            channel_id=channel_id,
            data={"connection_id": connection_id},
        )

    async def disconnect_websocket(self, channel_id: str, connection_id: str) -> None:
        """Unregister a WebSocket connection and emit framework event."""
        channel = self._channels.get(channel_id)
        if isinstance(channel, WebSocketChannel):
            channel.unregister_connection(connection_id)
        await self._emit_framework_event(
            "channel_disconnected",
            channel_id=channel_id,
            data={"connection_id": connection_id},
        )

    # -- Read tracking --

    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        """Mark an event as read for a channel."""
        await self._store.mark_read(room_id, channel_id, event_id)

    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        """Mark all events as read for a channel."""
        await self._store.mark_all_read(room_id, channel_id)

    # -- Realtime (ephemeral events) --

    async def publish_typing(
        self,
        room_id: str,
        user_id: str,
        is_typing: bool = True,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Publish a typing indicator for a user in a room.

        Args:
            room_id: The room to publish the typing event in.
            user_id: The user who is typing.
            is_typing: True for typing_start, False for typing_stop.
            data: Optional additional data (e.g., {"name": "User Name"}).
        """
        event = EphemeralEvent(
            room_id=room_id,
            type=EphemeralEventType.TYPING_START if is_typing else EphemeralEventType.TYPING_STOP,
            user_id=user_id,
            data=data or {},
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_presence(self, room_id: str, user_id: str, status: str) -> None:
        """Publish a presence update for a user in a room.

        Args:
            room_id: The room to publish the presence event in.
            user_id: The user whose presence changed.
            status: One of "online", "away", or "offline".
        """
        type_map = {
            "online": EphemeralEventType.PRESENCE_ONLINE,
            "away": EphemeralEventType.PRESENCE_AWAY,
            "offline": EphemeralEventType.PRESENCE_OFFLINE,
        }
        event_type = type_map.get(status, EphemeralEventType.CUSTOM)
        event = EphemeralEvent(
            room_id=room_id,
            type=event_type,
            user_id=user_id,
            data={"status": status} if event_type == EphemeralEventType.CUSTOM else {},
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_read_receipt(self, room_id: str, user_id: str, event_id: str) -> None:
        """Publish a read receipt for a user in a room.

        Args:
            room_id: The room to publish the read receipt in.
            user_id: The user who read the message.
            event_id: The ID of the event that was read.
        """
        event = EphemeralEvent(
            room_id=room_id,
            type=EphemeralEventType.READ_RECEIPT,
            user_id=user_id,
            data={"event_id": event_id},
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_reaction(
        self,
        room_id: str,
        user_id: str,
        target_event_id: str,
        emoji: str,
        action: str = "add",
    ) -> None:
        """Publish a reaction to a message in a room.

        Args:
            room_id: The room containing the target event.
            user_id: The user reacting.
            target_event_id: The ID of the event being reacted to.
            emoji: The emoji reaction (empty string to remove).
            action: ``"add"`` or ``"remove"``.
        """
        event = EphemeralEvent(
            room_id=room_id,
            type=EphemeralEventType.REACTION,
            user_id=user_id,
            data={
                "target_event_id": target_event_id,
                "emoji": emoji,
                "action": action,
            },
        )
        await self._realtime.publish_to_room(room_id, event)

    async def subscribe_room(self, room_id: str, callback: EphemeralCallback) -> str:
        """Subscribe to ephemeral events for a room.

        Args:
            room_id: The room to subscribe to.
            callback: Async callback invoked for each ephemeral event.

        Returns:
            A subscription ID that can be used to unsubscribe.
        """
        return await self._realtime.subscribe_to_room(room_id, callback)

    async def unsubscribe_room(self, subscription_id: str) -> bool:
        """Unsubscribe from ephemeral events.

        Args:
            subscription_id: The subscription ID returned by subscribe_room.

        Returns:
            True if the subscription existed and was removed.
        """
        return await self._realtime.unsubscribe(subscription_id)

    # -- Event-driven sources --

    async def attach_source(
        self,
        channel_id: str,
        source: SourceProvider,
        *,
        auto_restart: bool = True,
        restart_delay: float = 5.0,
        max_restart_delay: float = 300.0,
        max_restart_attempts: int | None = None,
        max_concurrent_emits: int | None = 10,
    ) -> None:
        """Attach an event-driven source to a channel.

        The source will start listening for messages and emit them into
        RoomKit's inbound pipeline via ``process_inbound()``.

        Args:
            channel_id: The channel ID to associate with this source.
                Messages from this source will be tagged with this channel_id.
            source: The source provider instance to attach.
            auto_restart: If True (default), automatically restart the source
                if it exits unexpectedly. Set to False for one-shot sources.
            restart_delay: Initial delay in seconds before restarting after
                failure. Doubles on each consecutive failure (exponential backoff).
            max_restart_delay: Maximum delay between restart attempts in seconds.
                Backoff is capped at this value. Defaults to 300 (5 minutes).
            max_restart_attempts: Maximum number of consecutive restart attempts
                before giving up. If None (default), retries indefinitely.
                When exhausted, emits ``source_exhausted`` framework event.
            max_concurrent_emits: Maximum number of concurrent ``emit()`` calls
                to prevent backpressure buildup. Defaults to 10. Set to None
                for unlimited concurrency (not recommended for high-volume sources).

        Raises:
            SourceAlreadyAttachedError: If a source is already attached to
                this channel_id.

        Example:
            from roomkit.sources.neonize import NeonizeSource

            source = NeonizeSource(session_path="~/.roomkit/wa.db")
            await kit.attach_source(
                "whatsapp-personal",
                source,
                max_restart_attempts=5,      # Give up after 5 failures
                max_concurrent_emits=20,     # Allow 20 concurrent messages
            )
        """
        if channel_id in self._sources:
            raise SourceAlreadyAttachedError(f"Source already attached to channel {channel_id}")

        logger = logging.getLogger("roomkit.sources")

        # Create emit callback with optional backpressure control
        if max_concurrent_emits is not None:
            semaphore = asyncio.Semaphore(max_concurrent_emits)

            async def emit(msg: InboundMessage) -> InboundResult:
                async with semaphore:
                    return await self.process_inbound(msg)
        else:

            async def emit(msg: InboundMessage) -> InboundResult:
                return await self.process_inbound(msg)

        self._sources[channel_id] = source
        self._source_tasks[channel_id] = asyncio.create_task(
            self._run_source(
                channel_id,
                source,
                emit,
                auto_restart,
                restart_delay,
                max_restart_delay,
                max_restart_attempts,
                logger,
            ),
            name=f"source:{channel_id}",
        )

        await self._emit_framework_event(
            "source_attached",
            channel_id=channel_id,
            data={"source_name": source.name},
        )

    async def detach_source(self, channel_id: str) -> None:
        """Detach and stop an event-driven source.

        Args:
            channel_id: The channel ID of the source to detach.

        Raises:
            SourceNotFoundError: If no source is attached to this channel_id.
        """
        if channel_id not in self._sources:
            raise SourceNotFoundError(f"No source attached to channel {channel_id}")

        source = self._sources.pop(channel_id)
        task = self._source_tasks.pop(channel_id)

        # Stop the source
        await source.stop()

        # Cancel the runner task and await its completion
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

        await self._emit_framework_event(
            "source_detached",
            channel_id=channel_id,
            data={"source_name": source.name},
        )

    async def _run_source(
        self,
        channel_id: str,
        source: SourceProvider,
        emit: Callable[[InboundMessage], Awaitable[InboundResult]],
        auto_restart: bool,
        restart_delay: float,
        max_restart_delay: float,
        max_restart_attempts: int | None,
        logger: logging.Logger,
    ) -> None:
        """Run a source with optional auto-restart on failure.

        Uses exponential backoff: delay doubles on each failure, capped at
        max_restart_delay. Delay resets after a successful start.
        """

        attempt = 0
        current_delay = restart_delay

        while True:
            try:
                logger.info("Starting source %s for channel %s", source.name, channel_id)
                await source.start(emit)
                # Clean exit - source stopped normally
                logger.info("Source %s stopped cleanly", source.name)
                self._sources.pop(channel_id, None)
                self._source_tasks.pop(channel_id, None)
                break
            except asyncio.CancelledError:
                logger.debug("Source %s cancelled", source.name)
                raise
            except Exception as e:
                attempt += 1
                logger.exception("Source %s failed (attempt %d): %s", source.name, attempt, e)
                await self._emit_framework_event(
                    "source_error",
                    channel_id=channel_id,
                    data={"source_name": source.name, "error": str(e), "attempt": attempt},
                )

                if not auto_restart:
                    raise

                # Check if max attempts exceeded
                if max_restart_attempts is not None and attempt >= max_restart_attempts:
                    logger.error(
                        "Source %s exhausted after %d attempts, giving up",
                        source.name,
                        attempt,
                    )
                    await self._emit_framework_event(
                        "source_exhausted",
                        channel_id=channel_id,
                        data={
                            "source_name": source.name,
                            "attempts": attempt,
                            "last_error": str(e),
                        },
                    )
                    break

                logger.info(
                    "Restarting source %s in %.1f seconds (attempt %d%s)",
                    source.name,
                    current_delay,
                    attempt,
                    f"/{max_restart_attempts}" if max_restart_attempts else "",
                )
                await asyncio.sleep(current_delay)

                # Exponential backoff: double delay, cap at max
                current_delay = min(current_delay * 2, max_restart_delay)

    async def source_health(self, channel_id: str) -> SourceHealth | None:
        """Get health information for an attached source.

        Args:
            channel_id: The channel ID of the source.

        Returns:
            SourceHealth if a source is attached, None otherwise.
        """
        source = self._sources.get(channel_id)
        if source is None:
            return None
        return await source.healthcheck()

    def list_sources(self) -> dict[str, SourceStatus]:
        """List all attached sources and their status.

        Returns:
            Dict mapping channel_id to current SourceStatus.
        """
        return {cid: source.status for cid, source in self._sources.items()}

    # -- Hook decorators --

    def hook(
        self,
        trigger: HookTrigger,
        execution: HookExecution = HookExecution.SYNC,
        priority: int = 0,
        name: str = "",
        timeout: float = 30.0,
        channel_types: set[ChannelType] | None = None,
        channel_ids: set[str] | None = None,
        directions: set[ChannelDirection] | None = None,
    ) -> Callable[..., Any]:
        """Decorator to register a global hook.

        Args:
            trigger: When the hook fires (BEFORE_BROADCAST, AFTER_BROADCAST, etc.)
            execution: SYNC (can block/modify) or ASYNC (fire-and-forget)
            priority: Lower numbers run first (default: 0)
            name: Optional name for logging and removal
            timeout: Max execution time in seconds (default: 30.0)
            channel_types: Only run for events from these channel types (None = all)
            channel_ids: Only run for events from these channel IDs (None = all)
            directions: Only run for events with these directions (None = all)
        """

        def decorator(fn: SyncHookFn | AsyncHookFn) -> SyncHookFn | AsyncHookFn:
            self._hook_engine.register(
                HookRegistration(
                    trigger=trigger,
                    execution=execution,
                    fn=fn,
                    priority=priority,
                    name=name or fn.__name__,
                    timeout=timeout,
                    channel_types=channel_types,
                    channel_ids=channel_ids,
                    directions=directions,
                )
            )
            return fn

        return decorator

    def on(self, event_type: str) -> Callable[..., Any]:
        """Decorator to register a framework event handler filtered by type."""

        def decorator(fn: FrameworkEventHandler) -> FrameworkEventHandler:
            self._event_handlers.append((event_type, fn))
            return fn

        return decorator

    def identity_hook(
        self,
        trigger: HookTrigger,
        channel_types: set[ChannelType] | None = None,
        channel_ids: set[str] | None = None,
        directions: set[ChannelDirection] | None = None,
    ) -> Callable[..., Any]:
        """Decorator to register an identity-resolution hook.

        The decorated function receives ``(event, context, id_result)`` and
        returns an ``IdentityHookResult`` or ``None``.

        Args:
            trigger: When the hook fires (ON_IDENTITY_AMBIGUOUS, ON_IDENTITY_UNKNOWN).
            channel_types: Only run for events from these channel types (None = all).
            channel_ids: Only run for events from these channel IDs (None = all).
            directions: Only run for events with these directions (None = all).
        """

        def decorator(fn: IdentityHookFn) -> IdentityHookFn:
            registration = IdentityHookRegistration(
                trigger=trigger,
                fn=fn,
                channel_types=channel_types,
                channel_ids=channel_ids,
                directions=directions,
            )
            self._identity_hooks.setdefault(trigger, []).append(registration)
            return fn

        return decorator

    def on_delivery_status(
        self, fn: Callable[[DeliveryStatus], Any]
    ) -> Callable[[DeliveryStatus], Any]:
        """Decorator to register a delivery status handler.

        The decorated function is called when ``process_delivery_status()`` is
        invoked with a ``DeliveryStatus`` from a provider webhook.  Handlers
        are dispatched through the hook engine with room context.

        Example:
            @kit.on_delivery_status
            async def track_delivery(status: DeliveryStatus):
                if status.status == "delivered":
                    logger.info("Message %s delivered to %s", status.message_id, status.recipient)
                elif status.status == "failed":
                    logger.error("Message %s failed: %s", status.message_id, status.error_message)
        """
        if not asyncio.iscoroutinefunction(fn):
            orig = fn

            async def _sync_wrap(status: DeliveryStatus) -> Any:
                return orig(status)

            _sync_wrap.__name__ = getattr(orig, "__name__", "unknown")
            adapted: Callable[[DeliveryStatus], Any] = _sync_wrap
        else:
            adapted = fn

        async def _hook_fn(event: Any, context: RoomContext) -> None:
            await adapted(event)

        _hook_fn.__name__ = getattr(fn, "__name__", "unknown")
        self._hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_DELIVERY_STATUS,
                execution=HookExecution.ASYNC,
                fn=_hook_fn,
                name=getattr(fn, "__name__", "unknown"),
            )
        )
        return fn

    async def process_webhook(
        self,
        meta: WebhookMeta,
        channel_id: str,
    ) -> None:
        """Process any SMS provider webhook automatically.

        This is the simplest integration method. It handles:
        - Inbound messages → process_inbound() with all hooks
        - Delivery status → process_delivery_status() with ON_DELIVERY_STATUS hooks
        - Unknown webhooks → silently ignored (acknowledged)

        Args:
            meta: WebhookMeta from extract_sms_meta().
            channel_id: The channel ID for inbound messages.

        Example:
            @app.post("/webhooks/sms/{provider}/inbound")
            async def sms_webhook(provider: str, payload: dict):
                meta = extract_sms_meta(provider, payload)
                await kit.process_webhook(meta, channel_id=f"sms-{provider}")
                return {"ok": True}
        """
        if meta.is_inbound:
            inbound = meta.to_inbound(channel_id)
            await self.process_inbound(inbound)
        elif meta.is_status:
            status = meta.to_status()
            status.channel_id = channel_id
            await self.process_delivery_status(status)
        # else: unknown webhook type, silently acknowledge

    async def process_delivery_status(self, status: DeliveryStatus) -> None:
        """Process a delivery status through the hook engine.

        Resolves the room from ``status.room_id`` or ``status.channel_id``
        (via the store) and dispatches ON_DELIVERY_STATUS hooks with full
        room context.

        Args:
            status: The DeliveryStatus from meta.to_status().
        """
        room_id = status.room_id
        if not room_id and status.channel_id:
            room_id = await self._store.find_room_id_by_channel(
                status.channel_id, status=str(RoomStatus.ACTIVE)
            )

        if not room_id:
            logging.getLogger("roomkit.framework").warning(
                "Cannot dispatch ON_DELIVERY_STATUS for message %s: no room_id resolved",
                status.message_id,
            )
            return

        try:
            context = await self._build_context(room_id)
        except Exception:
            room = await self._store.get_room(room_id)
            if room is None:
                return
            context = RoomContext(room=room, bindings=[])

        await self._hook_engine.run_async_hooks(
            room_id,
            HookTrigger.ON_DELIVERY_STATUS,
            status,
            context,
            skip_event_filter=True,
        )

    def add_room_hook(
        self,
        room_id: str,
        trigger: HookTrigger,
        execution: HookExecution,
        fn: SyncHookFn | AsyncHookFn,
        priority: int = 0,
        name: str = "",
    ) -> None:
        """Add a hook for a specific room."""
        self._hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=trigger,
                execution=execution,
                fn=fn,
                priority=priority,
                name=name,
            ),
        )

    def remove_room_hook(self, room_id: str, name: str) -> bool:
        """Remove a room hook by name."""
        return self._hook_engine.remove_room_hook(room_id, name)
