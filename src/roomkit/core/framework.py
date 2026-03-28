"""RoomKit - central orchestrator for multi-channel conversations."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.delivery.base import DeliveryBackend
    from roomkit.orchestration.base import Orchestration
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.telemetry.config import TelemetryConfig
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

from roomkit.channels.base import Channel
from roomkit.channels.websocket import SendFn, StreamSendFn, WebSocketChannel
from roomkit.core.delivery import DeliveryStrategy
from roomkit.core.event_router import EventRouter
from roomkit.core.exceptions import (
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    IdentityNotFoundError,
    ParticipantNotFoundError,
    RoomKitError,
    RoomNotFoundError,
    SourceAlreadyAttachedError,
    SourceNotFoundError,
    VoiceBackendNotConfiguredError,
    VoiceNotConfiguredError,
)
from roomkit.core.hooks import (
    HookEngine,
    IdentityHookRegistration,
)
from roomkit.core.inbound_router import DefaultInboundRoomRouter, InboundRoomRouter
from roomkit.core.locks import InMemoryLockManager, RoomLockManager
from roomkit.core.mixins import (
    ChannelOpsMixin,
    DelegationMixin,
    DeliverMixin,
    FrameworkEventHandler,
    GreetingMixin,
    HelpersMixin,
    HooksApiMixin,
    IdentityHookFn,
    InboundIdentityMixin,
    InboundLockedMixin,
    InboundMixin,
    InboundStreamingMixin,
    RealtimeOpsMixin,
    RecordingMixin,
    RoomLifecycleMixin,
    SourceOpsMixin,
    VoiceOpsMixin,
)
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.identity.base import IdentityResolver
from roomkit.models.channel import RateLimit
from roomkit.models.enums import (
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.task import Observation, Task
from roomkit.orchestration.status_bus import StatusBus, StatusEntry
from roomkit.realtime.base import (
    RealtimeBackend,
)
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.sources.base import SourceProvider
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.tasks.base import TaskRunner

logger = logging.getLogger("roomkit.framework")

# Re-export exception classes so existing ``from roomkit.core.framework import ...``
# statements continue to work without changes.
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


class RoomKit(
    InboundMixin,
    InboundIdentityMixin,
    InboundLockedMixin,
    InboundStreamingMixin,
    ChannelOpsMixin,
    RoomLifecycleMixin,
    VoiceOpsMixin,
    RecordingMixin,
    GreetingMixin,
    DelegationMixin,
    DeliverMixin,
    RealtimeOpsMixin,
    SourceOpsMixin,
    HooksApiMixin,
    HelpersMixin,
):
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
        task_runner: TaskRunner | None = None,
        delivery_strategy: DeliveryStrategy | str | None = None,
        delivery_backend: DeliveryBackend | None = None,
        status_bus: StatusBus | None = None,
        telemetry: TelemetryConfig | TelemetryProvider | None = None,
        inbound_rate_limit: RateLimit | None = None,
        orchestration: Orchestration | None = None,
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
            task_runner: Pluggable backend for delegated background tasks.
                Defaults to ``InMemoryTaskRunner``.
            delivery_strategy: Controls proactive delivery of background task
                results.  When set, ``strategy.deliver()`` is called after
                system prompt injection and the ``ON_TASK_COMPLETED`` hook.
                Can be overridden per-task via ``delegate()``.
            delivery_backend: Persistent delivery backend.  When set,
                ``kit.deliver()`` enqueues items instead of executing
                in-process, and a worker loop dequeues and executes them.
                Defaults to ``None`` (in-process delivery).
            status_bus: Shared status bus for multi-agent coordination.
                Defaults to a ``StatusBus`` with ``InMemoryStatusBackend``.
                Access via ``kit.status_bus``.
            telemetry: Optional telemetry provider or config for span/metric
                collection. Accepts a ``TelemetryProvider`` instance or a
                ``TelemetryConfig``. Defaults to ``NoopTelemetryProvider``.
            inbound_rate_limit: Optional rate limit applied to all inbound
                messages before any processing. Messages exceeding the limit
                are dropped with ``reason="rate_limited"``. Keyed per
                ``channel_id``.
            orchestration: Default orchestration strategy applied to rooms
                created via ``create_room()`` unless overridden per-room.
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
        # Inbound rate limiting
        self._inbound_rate_limit = inbound_rate_limit
        if inbound_rate_limit is not None:
            from roomkit.core.rate_limiter import TokenBucketRateLimiter

            self._inbound_rate_limiter: TokenBucketRateLimiter | None = TokenBucketRateLimiter()
        else:
            self._inbound_rate_limiter = None
        # Event-driven sources
        self._sources: dict[str, SourceProvider] = {}
        self._source_tasks: dict[str, asyncio.Task[None]] = {}
        # Voice support
        self._stt = stt
        self._tts = tts
        self._voice = voice
        # Background task delegation
        from roomkit.tasks.memory import InMemoryTaskRunner

        self._task_runner: TaskRunner = task_runner or InMemoryTaskRunner()
        from roomkit.core.delivery import resolve_strategy as _resolve

        self._delivery_strategy = _resolve(delivery_strategy)
        # Persistent delivery backend (optional)
        self._delivery_backend: DeliveryBackend | None = delivery_backend
        # Status bus for multi-agent coordination
        self._status_bus = status_bus or StatusBus()

        async def _on_status_posted(entry: StatusEntry) -> None:
            await self._emit_framework_event("status_posted", data=entry.model_dump())

        self._status_bus_callback = _on_status_posted
        # Subscribe lazily — requires a running event loop (deferred to first use)
        self._status_bus_subscribed = False
        # Greeting gates: block intelligence channels until greeting is stored.
        # Reference-counted so multi-agent rooms release only when ALL agents finish.
        self._greeting_gates: dict[str, asyncio.Event] = {}
        self._greeting_gate_counts: dict[str, int] = {}
        # Traces received before the room exists (flushed on attach_channel)
        self._pending_traces: dict[str, list[object]] = {}
        # Track fire-and-forget trace hook tasks to prevent GC
        self._pending_hook_tasks: set[asyncio.Task[Any]] = set()
        # Telemetry
        if isinstance(telemetry, _TelemetryProviderCls):
            self._telemetry: _TelemetryProviderCls = telemetry
        elif isinstance(telemetry, _TelemetryConfigCls):
            self._telemetry = telemetry.provider or NoopTelemetryProvider()
        else:
            self._telemetry = NoopTelemetryProvider()
        self._hook_engine._telemetry = self._telemetry
        if isinstance(telemetry, _TelemetryConfigCls):
            self._hook_engine._suppressed_triggers = telemetry.suppressed_hook_triggers
            # Propagate global metadata to the provider for searchable tags
            if telemetry.metadata and hasattr(self._telemetry, "_metadata"):
                self._telemetry._metadata = telemetry.metadata  # ty: ignore[invalid-assignment]
        self._store._telemetry = self._telemetry  # ty: ignore[invalid-assignment]
        # Default orchestration strategy
        self._default_orchestration = orchestration
        # Room-level media recording
        from roomkit.recorder._room_recorder_manager import RoomRecorderManager

        self._room_recorder_mgr = RoomRecorderManager()

    # -- Properties --

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
    def task_runner(self) -> TaskRunner:
        """The task runner for background delegation."""
        return self._task_runner

    @property
    def delivery_backend(self) -> DeliveryBackend | None:
        """The persistent delivery backend, if configured."""
        return self._delivery_backend

    @property
    def telemetry(self) -> TelemetryProvider:
        """The telemetry provider for span and metric collection."""
        return self._telemetry

    @property
    def lock_manager(self) -> RoomLockManager:
        """The per-room lock manager."""
        return self._lock_manager

    @property
    def status_bus(self) -> StatusBus:
        """Shared status bus for multi-agent coordination."""
        return self._status_bus

    @property
    def channels(self) -> dict[str, Channel]:
        """Registered channels keyed by channel ID."""
        return self._channels

    # -- Core infrastructure --

    async def _ensure_status_bus_subscribed(self) -> None:
        """Subscribe the framework event callback to the status bus (once)."""
        if not self._status_bus_subscribed:
            await self._status_bus.subscribe(self._status_bus_callback)
            self._status_bus_subscribed = True

    def _get_router(self) -> EventRouter:
        if self._event_router is None:
            self._event_router = EventRouter(
                channels=self._channels,
                transcoder=self._transcoder,
                max_chain_depth=self._max_chain_depth,
                telemetry=self._telemetry,
                greeting_gate_fn=self._wait_greeting_gate,
            )
        return self._event_router

    async def close(self) -> None:
        """Close all sources, channels, voice backend, and the realtime backend."""
        # Stop room-level media recorders before channels close
        self._room_recorder_mgr.close()
        # Clear stale greeting gates
        for room_id in list(self._greeting_gates):
            self._force_clear_greeting_gate(room_id)
        # Stop delivery backend worker loop
        if self._delivery_backend is not None:
            await self._delivery_backend.close()
        # Cancel in-flight background tasks first
        await self._task_runner.close()
        # Cancel pending trace hook tasks
        for task in self._pending_hook_tasks:
            task.cancel()
        if self._pending_hook_tasks:
            await asyncio.gather(*self._pending_hook_tasks, return_exceptions=True)
            self._pending_hook_tasks.clear()
        # Stop all event sources
        for channel_id in list(self._sources.keys()):
            await self.detach_source(channel_id)
        # Then close channels
        for channel in self._channels.values():
            await channel.close()
        # Close voice backend
        if self._voice:
            await self._voice.close()
        await self._realtime.close()
        # Close status bus
        await self._status_bus.close()
        # Flush telemetry (ends active spans, flushes exporter)
        self._telemetry.close()

    async def __aenter__(self) -> RoomKit:
        await self._ensure_status_bus_subscribed()
        if self._delivery_backend is not None:
            await self._delivery_backend.start(self)
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
        *,
        after_index: int | None = None,
        before_index: int | None = None,
    ) -> list[RoomEvent]:
        """Query the event timeline for a room.

        Supports offset-based (``offset``/``limit``) and cursor-based
        (``after_index``/``before_index``) pagination.  When a cursor
        parameter is set, ``offset`` is ignored.

        Args:
            room_id: Room to query.
            offset: Number of events to skip (offset-based mode).
            limit: Maximum number of events to return.
            visibility_filter: Optional visibility value to filter by.
            after_index: Return events with ``index > after_index``.
            before_index: Return events with ``index < before_index``.
        """
        await self.get_room(room_id)
        return await self._store.list_events(
            room_id,
            offset=offset,
            limit=limit,
            visibility_filter=visibility_filter,
            after_index=after_index,
            before_index=before_index,
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
        response_visibility: str | None = None,
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
            response_visibility: Controls where the AI's response is delivered.
                Uses the same vocabulary as visibility. None means no restriction.
        """
        from roomkit.telemetry.base import SpanKind
        from roomkit.telemetry.context import get_current_span, reset_span, set_current_span

        await self._ensure_status_bus_subscribed()
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
            response_visibility=response_visibility,
        )

        telemetry = self._telemetry
        span_id = telemetry.start_span(
            SpanKind.INBOUND_PIPELINE,
            "framework.send_event",
            parent_id=get_current_span(),
            room_id=room_id,
            channel_id=channel_id,
            attributes={"event_type": str(event_type)},
        )
        token = set_current_span(span_id, telemetry_ctx=telemetry.get_span_context(span_id))
        try:
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

            telemetry.end_span(span_id)
        except Exception as exc:
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise
        finally:
            reset_span(token)

        return event

    # -- WebSocket lifecycle --

    async def connect_websocket(
        self,
        channel_id: str,
        connection_id: str,
        send_fn: SendFn,
        *,
        stream_send_fn: StreamSendFn | None = None,
    ) -> None:
        """Register a WebSocket connection and emit framework event."""
        await self._ensure_status_bus_subscribed()
        channel = self._channels.get(channel_id)
        if not isinstance(channel, WebSocketChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered WebSocket channel"
            )
        channel.register_connection(connection_id, send_fn, stream_send_fn=stream_send_fn)
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
