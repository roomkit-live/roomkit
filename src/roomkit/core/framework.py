"""RoomKit - central orchestrator for multi-channel conversations."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator
from datetime import datetime
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
    EventOpsMixin,
    FrameworkEventHandler,
    GreetingMixin,
    HelpersMixin,
    HooksApiMixin,
    IdentityHookFn,
    InboundIdentityMixin,
    InboundLockedMixin,
    InboundMixin,
    InboundStreamingMixin,
    MembershipMixin,
    RealtimeOpsMixin,
    RecordingMixin,
    RegenerateMixin,
    RoomLifecycleMixin,
    SourceOpsMixin,
    VoiceOpsMixin,
)
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.identity.base import IdentityResolver
from roomkit.models.channel import RateLimit
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.store_filter import PersistencePolicy
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
    RegenerateMixin,
    ChannelOpsMixin,
    RoomLifecycleMixin,
    MembershipMixin,
    VoiceOpsMixin,
    RecordingMixin,
    GreetingMixin,
    DelegationMixin,
    DeliverMixin,
    RealtimeOpsMixin,
    SourceOpsMixin,
    EventOpsMixin,
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
        persistence_policy: PersistencePolicy | None = None,
    ) -> None:
        """Initialise the RoomKit orchestrator.

        Args:
            store: Persistent storage backend. Defaults to ``InMemoryStore``.
            identity_resolver: Optional resolver for identifying inbound senders.
            identity_channel_types: Restrict identity resolution to specific channel
                types. If ``None`` (default), resolution runs for all channels.
                Set to e.g. ``{ChannelType.SMS}`` to only resolve identity for SMS.
                Applies wherever resolution runs, the conference arrival path
                included — see :meth:`identity_enabled_for`.
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
            persistence_policy: Controls which event types are persisted.
                When ``None`` (default), all events are persisted. Use
                ``PersistencePolicy(exclude_types={...})`` to skip specific
                types or ``PersistencePolicy(persist_types={...})`` to
                whitelist.
        """
        from roomkit.telemetry.base import TelemetryProvider as _TelemetryProviderCls
        from roomkit.telemetry.config import TelemetryConfig as _TelemetryConfigCls
        from roomkit.telemetry.noop import NoopTelemetryProvider

        self._store = store or InMemoryStore()
        self._persistence_policy = persistence_policy
        self._identity_resolver = identity_resolver
        self._identity_channel_types = identity_channel_types
        self._max_chain_depth = max_chain_depth
        self._identity_timeout = identity_timeout
        self._process_timeout = process_timeout
        self._channels: dict[str, Channel] = {}
        # (room_id, channel_id) pairs an integrator has explicitly detached.
        # Detaching is how access is revoked, so the inbound path's convenience
        # auto-attach MUST NOT hand it back (RFC §7.5-7); without this the next
        # message naming the room silently re-attaches at default permissions.
        # In-process, like `_channels` itself: it records a decision made
        # against this framework instance, and a restart re-reads bindings from
        # the store, where the revoked one is already absent.
        self._detached_bindings: set[tuple[str, str]] = set()
        self._hook_engine = HookEngine()
        self._lock_manager = lock_manager or InMemoryLockManager()
        # A persistent store paired with an in-process lock is unsafe if the
        # store is shared across processes: per-process locks do not coordinate,
        # so concurrent workers can assign duplicate event indices (RFC §13.5).
        if not isinstance(self._store, InMemoryStore) and isinstance(
            self._lock_manager, InMemoryLockManager
        ):
            logger.warning(
                "%s is paired with InMemoryLockManager. This is safe only in a "
                "single process; if the store is shared across processes (e.g. a "
                "load-balanced deployment), use a distributed lock manager such "
                "as PostgresAdvisoryLockManager to avoid duplicate event indices.",
                type(self._store).__name__,
            )
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
        self._resource_leases: set[asyncio.Event] = set()
        self._resource_leases_sealed = False
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
    def identity_resolver(self) -> IdentityResolver | None:
        """The pluggable identity resolver, if one was configured.

        Exposed because the inbound pipeline is not the only place identity is
        resolved: a conference participant the framework did not name must be
        identified when it arrives rather than when it first speaks (RFC
        §12.10.2), and there is no inbound message at that point to carry it
        through the pipeline.
        """
        return self._identity_resolver

    @property
    def identity_timeout(self) -> float:
        """How long identity resolution may take before it counts as UNKNOWN.

        The same budget wherever resolution runs (RFC §11.5), so a resolver
        cannot be slow in one caller and bounded in another.
        """
        return self._identity_timeout

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
        """Close every channel, then release what they share.

        In two phases (RFC 12.10.4). First the channels close, one at a time
        and shielded from one another, each on its own bounded budgets — a
        conference channel's bot is out of its meeting when this phase ends,
        whatever the store is doing. Then, with every channel's media
        released, the framework waits — with no deadline — for the operations
        the store and the lock manager already have (each runs under a
        resource lease), seals both against new work, and releases them. An
        operation arriving after the seal is refused with
        :class:`RoomKitError` rather than started against a resource being
        released.

        Raises:
            ExceptionGroup: one or more channels failed to close. Raised only
                once the rest of the shutdown has run to completion — every
                other channel is closed and the shared resources are released
                — so nothing else is skipped on a failure's account. The
                channel that failed may still be holding its own resources (a
                bot possibly still in its meeting); the group names each one.
        """
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
        # Then close the channels. In sequence, but shielded from one another:
        # they close one at a time, so a close that raises would leave every
        # channel behind it holding its media — for a conference channel, a
        # bot left sitting in a meeting (RFC 12.10.4). Collected rather than
        # swallowed: the failed channel's own resources are in an unknown
        # state — its bot may still be in its conference — and a close() that
        # returns cleanly over that turns a logged error into an operational
        # and disclosure risk. Raised at the very end, so the failure of one
        # channel costs nothing else its shutdown.
        channel_failures: list[Exception] = []
        for channel_id, channel in self._channels.items():
            try:
                await channel.close()
            except Exception as error:
                logger.exception(
                    "Channel %r failed to close; the remaining channels still close",
                    channel_id,
                )
                error.add_note(f"while closing channel {channel_id!r}")
                channel_failures.append(error)
        # Close voice backend
        if self._voice:
            await self._voice.close()
        await self._realtime.close()
        # Every channel is closed and its media released. Only now wait for
        # the operations the store or the lock manager already has — they are
        # bracketed by resource leases, and this wait is what lets a channel's
        # own close() stay bounded (RFC 12.10.4).
        await self._await_resource_leases()
        # Close the conversation store (e.g. release a PostgreSQL pool). The
        # store's close() is idempotent and a no-op for a caller-owned pool.
        await self._store.close()
        # Close the lock manager (e.g. release an advisory-lock pool).
        await self._lock_manager.close()
        # Close status bus
        await self._status_bus.close()
        # Flush telemetry (ends active spans, flushes exporter)
        self._telemetry.close()
        # Last, the failures the channel loop collected. After everything, so
        # the caller learns of them without any other part of the shutdown
        # having been skipped on their account — what cannot be reported as a
        # success is a close that returns cleanly while a channel that failed
        # may still be holding its media (RFC 12.10.4).
        if channel_failures:
            raise ExceptionGroup(
                f"{len(channel_failures)} channel(s) failed to close; "
                "the rest of the shutdown ran to completion",
                channel_failures,
            )

    @contextlib.contextmanager
    def _resource_lease(self) -> Iterator[None]:
        """Hold the store and the lock manager open across one operation.

        Taken by a channel around work those resources have already been
        given — a roster write inside the room lock, from taking the lock to
        letting it go — and released when the last of it is out of them.
        ``close()`` waits for every lease after the channels have closed and
        their media is released, and only then releases the store and the
        lock manager: an operation a shared resource is running cannot be
        taken back, so the owner of the resource is the one that waits
        (RFC 12.10.4). Everything under a lease must be framework or channel
        code and the resource calls themselves — never integrator code, which
        can suspend forever and must only ever be waited for on a budget.

        Refused once the shutdown has sealed the registry. A callback can
        suspend in a backend past every closing budget while holding no lease
        at all — there is nothing for the shutdown to wait for — and resume
        after the store has been released, asking for a lease as its first
        act. Granting one then would be a use-after-free with a registration
        on it; the seal turns it into an error that says what happened.
        """
        if self._resource_leases_sealed:
            raise RoomKitError(
                "RoomKit.close() has sealed the store and the lock manager; an operation "
                "arriving now would run against resources that are being released"
            )
        released = asyncio.Event()
        self._resource_leases.add(released)
        try:
            yield
        finally:
            self._resource_leases.discard(released)
            released.set()

    async def _await_resource_leases(self) -> None:
        """Wait for every resource lease, with no deadline — then seal.

        No deadline, because there is no third option: the work under a lease
        is already inside the store or the lock manager, giving up on it is
        releasing the resource under it, and the media was all released before
        this wait begins — so it costs the shutdown its latency and nothing
        else. And it terminates: nothing integrator-owned ever runs under a
        lease, so every lease is first-party code and the resource calls
        themselves. The loop re-reads the registry because a lease can still
        register while this waits — a straggler announcement building its
        room context, say — and the wait owes it the same patience.

        The seal is set the moment the registry is last seen empty, with no
        await in between, so there is no instant at which the wait has
        concluded and a new lease could still be granted. What arrives after
        is refused by :meth:`_resource_lease` — the alternative was a lease
        registered onto a registry nothing will read again, over a resource
        already being released.
        """
        if self._resource_leases:
            logger.warning(
                "Shutdown is waiting for %d operation(s) the store or the lock manager "
                "is still running. Every channel is closed and its media released, so "
                "this costs the shutdown its latency and nothing else",
                len(self._resource_leases),
            )
        while self._resource_leases:
            waiters = [
                asyncio.ensure_future(released.wait()) for released in list(self._resource_leases)
            ]
            try:
                await asyncio.wait(waiters)
            finally:
                for waiter in waiters:
                    waiter.cancel()
        self._resource_leases_sealed = True

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
        visibility: str = Visibility.ALL,
        provider: str | None = None,
        response_visibility: str | None = None,
        created_at: datetime | None = None,
        parent_event_id: str | None = None,
        idempotency_key: str | None = None,
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
            parent_event_id: In-app thread parent. The locked pipeline normalises
                it to the thread root (flat two-level model); see
                :meth:`_resolve_thread_root`.
            idempotency_key: Stable de-duplication key. When set, the locked
                pipeline's idempotency check (backed by the unique
                ``events(room_id, idempotency_key)`` index) skips a re-send that
                carries the same key — so a caller that may replay a publish (an
                outbox dispatcher redelivering after a crash) gets at-most-once
                persistence for a given key. None keeps the prior behaviour (no
                de-duplication), matching inbound events that carry no key.
        """
        from roomkit.telemetry.base import SpanKind
        from roomkit.telemetry.context import get_current_span, reset_span, set_current_span

        await self._ensure_status_bus_subscribed()
        await self.get_room(room_id)
        binding = await self._get_binding(room_id, channel_id)

        event_kwargs: dict[str, Any] = dict(
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
            parent_event_id=parent_event_id,
            status=EventStatus.DELIVERED,
            metadata=metadata or {},
            visibility=visibility,
            response_visibility=response_visibility,
            idempotency_key=idempotency_key,
        )
        if created_at is not None:
            event_kwargs["created_at"] = created_at
        event = RoomEvent(**event_kwargs)

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
            # Direct injection traverses the SAME locked pipeline as inbound
            # (RFC §10.5): index assignment, BEFORE_BROADCAST hooks, edit/delete
            # handling, source write-permission gate, persistence, broadcast,
            # reentry drain and AFTER_BROADCAST hooks. This keeps a single
            # validation/hooks/indexing/persistence model across entry points.
            pending_async_hooks: list[tuple[HookTrigger, RoomEvent, RoomContext]] = []
            pending_error_hooks: list[tuple[RoomContext, Any, dict[str, Any]]] = []
            pending_streams: list[Any] = []
            async with self._lock_manager.locked(room_id):
                context = await self._build_context(room_id)
                result = await self._process_locked(
                    event,
                    room_id,
                    context,
                    pending_after_broadcast_out=pending_async_hooks,
                    pending_error_hooks_out=pending_error_hooks,
                    pending_streams_out=pending_streams,
                )
            if isinstance(result.event, RoomEvent):
                event = result.event
            # AFTER_BROADCAST/mutation and ON_ERROR run outside the room lock (RFC §10.1)
            await self._run_deferred_async_hooks(room_id, pending_async_hooks)
            await self._run_deferred_error_hooks(room_id, pending_error_hooks)
            # Streaming AI responses to a directly-injected event are consumed
            # outside the lock, exactly like the inbound path — without this a
            # streaming provider's reply is generated and then silently dropped.
            if pending_streams:
                await self._process_streaming_responses(pending_streams, room_id)

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

    async def list_read_markers(self, room_id: str) -> dict[str, int]:
        """Return every channel's read high-water-mark (event index) in a room.

        Maps ``channel_id`` -> the highest read event ``index``. With one
        channel per member, this is the per-member read position used to
        aggregate "seen by" receipts; resolve channels to members via the
        bindings/participants (see :meth:`list_members`).
        """
        return await self._store.list_read_markers(room_id)
