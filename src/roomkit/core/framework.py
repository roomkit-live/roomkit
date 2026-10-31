"""RoomKit - central orchestrator for multi-channel conversations."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import weakref
from collections.abc import Awaitable, Callable, Coroutine, Generator, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.delivery.base import DeliveryBackend
    from roomkit.models.steering import SteeringDirective
    from roomkit.orchestration.base import Orchestration
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.telemetry.config import TelemetryConfig
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

from roomkit.channels.base import Channel
from roomkit.channels.websocket import SendFn, StreamSendFn, WebSocketChannel
from roomkit.core import _host_post
from roomkit.core._changes import CHANGE_REFUSALS
from roomkit.core._closing import Cascades, DoorRuns, event_cut_result, inbound_cut_result
from roomkit.core._instructions import names_an_agent, without_fingerprint
from roomkit.core._kit_turn_hooks import KitTurnHooks
from roomkit.core._kit_wiring import KitEvents, KitWiringEnv
from roomkit.core._publication_keys import refuse_reserved
from roomkit.core._reentrance import REENTRY_LOOP_CAP, Entry, RoomWork, chain_floor
from roomkit.core._timer_transitions import TimerTransitions
from roomkit.core._turn_host import KitHost
from roomkit.core._turn_runners import TurnRunners
from roomkit.core.admission import RoomAdmission
from roomkit.core.delivery import DeliveryStrategy
from roomkit.core.event_router import EventRouter
from roomkit.core.exceptions import (
    ChangeRefusedError,
    ChannelAlreadyRegisteredError,
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    IdentityNotFoundError,
    KitClosedError,
    ParticipantNotFoundError,
    ProcessTimeoutError,
    RoomClosedError,
    RoomKitError,
    RoomNotFoundError,
    ShutdownCancelledError,
    SourceAlreadyAttachedError,
    SourceNotFoundError,
    VoiceBackendNotConfiguredError,
    VoiceNotConfiguredError,
)
from roomkit.core.hooks import (
    TURN_TRIGGERS,
    HookEngine,
    IdentityHookRegistration,
)
from roomkit.core.inbound_router import DefaultInboundRoomRouter, InboundRoomRouter
from roomkit.core.locks import (
    InMemoryLockManager,
    ReleaseTolerantLockManager,
    RoomLockManager,
    close_bounded,
)
from roomkit.core.mixins import (
    BackgroundMixin,
    ChannelOpsMixin,
    ChannelStateMixin,
    DelegationMixin,
    DeliverMixin,
    DiscussionMixin,
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
    InjectedEventsMixin,
    LaneExecutionMixin,
    MembershipMixin,
    RealtimeOpsMixin,
    RecordingMixin,
    RegenerateMixin,
    RoomLifecycleMixin,
    RoomRecordingMixin,
    SourceOpsMixin,
    StrategyMixin,
    VerificationMixin,
    VoiceOpsMixin,
)
from roomkit.core.mixins.helpers import _refuses_writes
from roomkit.core.task_utils import (
    CLOSE_GRACE_S,
    CUT_UNWIND_S,
    Callers,
    acting_tasks,
    calling_close,
    cut_once_started,
    held_runs,
    spared_by_close,
    start_named,
    wait_for_unspared,
)
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.identity.base import IdentityResolver
from roomkit.models._keys import _no_blank_key
from roomkit.models._visibility import parse_given_visibility, parse_visibility
from roomkit.models.channel import ChannelBinding, ChannelOutput, RateLimit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import STANDALONE, InboundResult
from roomkit.models.enums import (
    AgentResponsePolicy,
    ChannelCategory,
    ChannelType,
    EventDirection,
    EventStatus,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import EventContent, EventSource, RoomEvent, refuse_too_deep
from roomkit.models.store_filter import EventFilter, PersistencePolicy
from roomkit.models.task import Observation, Task
from roomkit.models.verification import VerificationPolicy
from roomkit.orchestration.status_bus import StatusBus, StatusEntry
from roomkit.realtime.base import (
    RealtimeBackend,
)
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.sources.base import SourceProvider
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.tasks._room_tasks import RoomTaskRecord, TaskCancel
from roomkit.tasks.background import BackgroundSetup
from roomkit.tasks.base import TaskRunner
from roomkit.turns._registry_tools import OwnerToolStore
from roomkit.turns.base import TurnBackend
from roomkit.turns.executor import TurnExecutor
from roomkit.turns.owner import registry_turn_client

logger = logging.getLogger("roomkit.framework")


def _raise_unless_committed(result: InboundResult, room_id: str) -> None:
    """Raise what refused a direct injection before its commit point (RFC
    §10.5): its contract is the committed event, and handing back one marked
    DELIVERED for a write that never happened would be a lie the caller acts
    on. A room that refuses events (RFC §5.1), consistent with get_room(),
    which raises for a room that is gone, and ``process_timeout`` (§13.6)."""
    if not result.blocked:
        return
    if result.reason == "room_closed":
        raise RoomClosedError(f"Room {room_id} does not accept new events")
    if result.reason == "process_timeout":
        raise ProcessTimeoutError(
            f"Room {room_id}: process_timeout expired before the event was committed"
        )
    if result.event is None and result.reason in CHANGE_REFUSALS:
        raise ChangeRefusedError(result.reason)


def _direct_source(
    binding: ChannelBinding, participant_id: str | None, provider: str | None
) -> EventSource:
    """The source of a direct injection from *binding*'s channel (RFC §10.5):
    its channel's category gives its direction, a transport's event standing
    for a message coming in, an intelligence channel's being its own words
    going out, as its responses are."""
    return EventSource(
        channel_id=binding.channel_id,
        channel_type=binding.channel_type,
        direction=(
            EventDirection.OUTBOUND
            if binding.category == ChannelCategory.INTELLIGENCE
            else EventDirection.INBOUND
        ),
        participant_id=participant_id,
        provider=provider,
    )


# Re-export exception classes so existing ``from roomkit.core.framework import ...``
# statements continue to work without changes.
__all__ = [
    "ChannelNotFoundError",
    "ChannelAlreadyRegisteredError",
    "ChannelNotRegisteredError",
    "FrameworkEventHandler",
    "IdentityHookFn",
    "IdentityNotFoundError",
    "ParticipantNotFoundError",
    "RoomKit",
    "RoomKitError",
    "RoomClosedError",
    "RoomNotFoundError",
    "SourceAlreadyAttachedError",
    "SourceNotFoundError",
    "VoiceBackendNotConfiguredError",
    "VoiceNotConfiguredError",
]


async def _stop_step(step: str, work: Coroutine[Any, Any, None]) -> None:
    """One step of the kit's stop as it closes, its failure logged as it ends."""
    try:
        await work
    except Exception:
        logger.exception("Stopping the %s failed as the kit closed", step)


async def _await_shutdown(shutdown: asyncio.Task[None]) -> None:
    """Wait for the kit's *shutdown*, its failure raised. One cut itself is
    no cancellation of a caller nobody cancelled: it is told so."""
    try:
        await asyncio.shield(shutdown)
    except asyncio.CancelledError:
        caller = asyncio.current_task()
        if shutdown.cancelled() and (caller is None or not caller.cancelling()):
            raise ShutdownCancelledError(
                "The kit's shutdown was cancelled before it ended: "
                "its channels and its store may not all be closed"
            ) from None
        raise


class RoomKit(
    InboundMixin,
    InboundIdentityMixin,
    InboundLockedMixin,
    InboundStreamingMixin,
    InjectedEventsMixin,
    LaneExecutionMixin,
    RegenerateMixin,
    DiscussionMixin,
    StrategyMixin,
    ChannelOpsMixin,
    ChannelStateMixin,
    RoomLifecycleMixin,
    MembershipMixin,
    VoiceOpsMixin,
    RecordingMixin,
    RoomRecordingMixin,
    GreetingMixin,
    DelegationMixin,
    BackgroundMixin,
    DeliverMixin,
    RealtimeOpsMixin,
    SourceOpsMixin,
    EventOpsMixin,
    VerificationMixin,
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
        agent_response_policy: AgentResponsePolicy = AgentResponsePolicy.AGENT_CHAIN,
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
        delivery_gap_timeout: float = 30.0,
        delivery_claim_lock_manager: RoomLockManager | None = None,
        verification_policy: VerificationPolicy | None = None,
        media_resolver: MediaResolver | None = None,
        attached_media: AttachedMedia | None = None,
        turn_backend: TurnBackend | None = None,
        error_classes: Iterable[type[Exception]] = (),
    ) -> None:
        """Initialise the RoomKit orchestrator.

        Args:
            store: Persistent storage backend. Defaults to ``InMemoryStore``.
            identity_resolver: Optional resolver for identifying inbound senders.
                Without one, an event's ``participant_id`` is whatever the
                channel stamped on it (RFC §11.6) — so it is only as
                trustworthy as the transport that supplied it, and the
                edit/delete author check (RFC §10.3) is decided from it. Derive
                the sender id from an authenticated session rather than a
                client-supplied field, or install a resolver. See
                the Security Hardening guide.
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
            verification_policy: How long an identity verification and its
                requests last, and how many attempts a request accepts
                (RFC §11.7). Passing one also turns on the inbound side: an
                identified sender's messages are labelled with their
                verification and, while it is not verified, held from the
                room's agents. Omitted, the API works with the defaults and
                the pipeline is unchanged.
            media_resolver: The bytes of a media the application holds
                (RFC §6.4): ``async (media, room_id) -> bytes | None``. A
                ``MediaContent`` whose URL is neither a ``data:`` URI nor a
                link (``http``, ``https``) is a reference, in a scheme the
                application chooses; every AI channel registered here asks
                this resolver for its bytes as it builds a turn, so the store
                and broadcasts carry the reference alone. Any participant
                can write a reference: answer only for a media the room may
                show, None otherwise. Without a resolver, a referenced image
                reads as unavailable to the model.
            attached_media: The media an event carries outside its content,
                for an application that keeps a message's files beside its
                words (RFC §6.4): ``(event) -> Sequence[MediaContent]``, the
                same media in the same order each time it is asked. Every AI
                channel registered here reads the events of a turn, history
                and input alike, with the media named after their own
                content, as a composite; a reference among them is resolved
                by ``media_resolver``. The store and broadcasts keep the event
                as stored. A reader that raises names none for that event.
            turn_backend: What carries the turns of the intelligence channels
                (RFC §10.6). Defaults to ``InMemoryTurnBackend``: every turn
                runs in this process, as serialized messages, on the kit's own
                executor, which reads the channels registered here.
            error_classes: The application's exception classes, with their
                subclasses, that an error crossing the turn backend is
                rebuilt as (RFC §10.6.8), on this kit and on its own
                executor, beyond RoomKit's and Python's built-in ones: an error a
                turn raised reaches ``InboundResult.error``, ``ON_ERROR``
                and the chain of causes as the class it was raised as, its
                attributes kept. Any other class comes back as a
                ``RemoteTurnError`` that keeps its JSON attributes and reads
                as the nearest ancestor rebuilt here (a ``ProviderError``
                subclass, a ``TimeoutError``). An executor in another process
                declares its own (``TurnExecutor(error_classes=...)``).
            process_timeout: Timeout in seconds for the locked processing phase.
            stt: Optional speech-to-text provider for transcription.
            tts: Optional text-to-speech provider for synthesis.
            voice: Optional voice backend for real-time audio transport.
            task_runner: Pluggable backend for delegated background tasks.
                Defaults to ``InMemoryTaskRunner``.
            delivery_strategy: Controls when proactive content is delivered,
                a background task's result included: the result is handed back
                through ``deliver()`` after the ``ON_TASK_COMPLETED`` hook
                (RFC §23.3). ``deliver()`` takes a strategy per call.
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
            delivery_gap_timeout: Seconds a delivery lane waits on a cursor
                hole owned by an absent worker (one that committed events and
                crashed before delivering them) before skipping over it with
                a ``delivery_skipped`` framework event (RFC §10.2). The skip
                bounds the loss to the crash window; it never covers an
                index this process holds a plan for.
            delivery_claim_lock_manager: Lock manager used for the per-room
                delivery claims. Defaults to ``lock_manager``. Supply a
                separate ``PostgresAdvisoryLockManager`` (own pool) in
                multi-process deployments: a claim is held for the length of
                a delivery set — provider round trips included — and must
                not starve the room-lock pool that commits depend on.
        """
        from roomkit.telemetry.base import TelemetryProvider as _TelemetryProviderCls
        from roomkit.telemetry.config import TelemetryConfig as _TelemetryConfigCls
        from roomkit.telemetry.noop import NoopTelemetryProvider

        self._store = store or InMemoryStore()
        self._persistence_policy = persistence_policy
        self._identity_resolver = identity_resolver
        self._identity_channel_types = identity_channel_types
        self._media_resolver = media_resolver
        self._attached_media = attached_media
        self._max_chain_depth = max_chain_depth
        # Counted by _store_refusing_room: a stream reads the room's status
        # once, and again only when this moved (RFC §5.1).
        self._room_close_epoch = 0
        self._default_agent_response_policy = agent_response_policy
        self._identity_timeout = identity_timeout
        self._verification_policy = verification_policy or VerificationPolicy()
        # Labelling and holding inbound messages is opted into by passing a
        # policy: a kit that never asked keeps its pipeline as it was.
        self._verification_inbound = verification_policy is not None
        self._verification_timers: dict[str, asyncio.Task[None]] = {}
        self._process_timeout = process_timeout
        self._channels: dict[str, Channel] = {}
        self._start_turns(turn_backend, error_classes)
        # A strategy's runner takes an agent's turns on this owner (RFC §10.6.1).
        self._turn_runners = TurnRunners()
        # Explicit detaches (revocations, RFC §7.5-7) are recorded in room
        # metadata by detach_channel() — not here — so they survive restarts
        # and are visible to every process sharing the store.
        # Once close() starts, every call is refused and the kit's work cut
        # (RFC §12.10.4): the calls into a room under way, the deliveries of
        # the calls it took.
        self._closed = False
        self._door_runs = DoorRuns(self._store.get_event, self._close_spared)
        self._cascades = Cascades()
        # Set once the close released the store: a read the store refuses
        # then reads KitClosedError (RFC §12.10.4).
        self._store_released = False
        self._shutdown: asyncio.Task[None] | None = None
        self._shutdown_callers = Callers()
        # The shutdown's steps stopping the kit's own work, within its grace.
        self._stop_steps: set[asyncio.Task[None]] = set()
        # The work the close abandoned, ignoring its cut: named once.
        self._abandoned: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()
        self._hook_engine = HookEngine()
        # A hook on a trigger a turn fires registered on the kit is its own
        # executor's (RFC §10.6.6).
        self._kit_turn_hooks = KitTurnHooks(self._turn_hooks, self._channels)
        self._hook_engine.turn_side = self._kit_turn_hooks
        self._given_lock_manager = lock_manager or InMemoryLockManager()
        # A failed release never undoes a locked section (RFC §13.5).
        self._lock_manager = ReleaseTolerantLockManager(
            self._given_lock_manager, waiting=self._door_runs.waiting_for_lock
        )
        # Admission tickets for off-lock BEFORE_BROADCAST checks (RFC §9.5.1).
        self._admission = RoomAdmission(
            self._hook_engine, self._build_context, self._announce_hook_errors
        )
        # What a call into a room is to the room's own work (RFC §9.5.1).
        self._room_work = RoomWork(self._lock_manager, self._max_chain_depth * 10)
        # A persistent store paired with an in-process lock is unsafe if the
        # store is shared across processes: per-process locks do not coordinate
        # the full inbound pipeline, its idempotency decisions or the room's
        # register of authors (RFC §10.1 step 12, §13.5).
        if not self._store.is_process_local and isinstance(
            self._given_lock_manager, InMemoryLockManager
        ):
            logger.warning(
                "%s is paired with InMemoryLockManager. This is safe only in a "
                "single process; if the store is shared across processes (e.g. a "
                "load-balanced deployment), use a distributed lock manager such "
                "as PostgresAdvisoryLockManager with PostgresStore, or two "
                "processes may give two senders whose names read alike one rank.",
                type(self._store).__name__,
            )
        # Delivery lanes (RFC §10.2): broadcast planned under the room lock,
        # executed per room in index order off it. The claim defaults to the
        # room lock manager (derived key space); a dedicated manager keeps
        # long claim tenures off the commit pool in multi-process setups.
        from roomkit.core.lanes import LaneConfig, RoomLaneRegistry

        self._claim_locks = delivery_claim_lock_manager or self._lock_manager
        self._lanes = RoomLaneRegistry(
            self, self._claim_locks, LaneConfig(gap_timeout=delivery_gap_timeout)
        )
        self._realtime = realtime or InMemoryRealtime()
        self._transcoder = DefaultContentTranscoder()
        self._event_handlers: list[tuple[str, FrameworkEventHandler]] = []
        self._identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]] = {}
        self._inbound_router = inbound_router or DefaultInboundRoomRouter(
            self._store, channel_of=self._channels.get
        )
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
            # RFC §9.2 ON_STATUS_POSTED. The bus is global, so a status
            # carrying no room reaches no room's hooks — the same scoping the
            # framework event above has.
            room_id = (entry.metadata or {}).get("room_id")
            if not isinstance(room_id, str) or not room_id:
                return
            try:
                context = await self._build_context(room_id)
            except RoomNotFoundError:
                logger.debug("No such room %s for ON_STATUS_POSTED", room_id)
                return
            except KitClosedError:
                # Posted once the kit's close sealed its store (a run's
                # terminal entry, published late): no hook fires (RFC §12.10.4).
                logger.debug("The kit is closed: no ON_STATUS_POSTED for room %s", room_id)
                return
            self._hook_engine._start_async_hooks(
                room_id,
                HookTrigger.ON_STATUS_POSTED,
                entry,
                context,
                skip_event_filter=True,
            )

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
        # The background work held until it ends (core.task_utils.hold_task),
        # so close() cuts it: the strategies' runs and the delegations'
        # hand-backs under way (RFC §19.7.3, §19.7.4, §23.3).
        self._background_runs: set[asyncio.Task[None]] = set()
        # The writes past a call's decision, run to their end whatever becomes
        # of its caller (core.task_utils.run_to_its_end, RFC §13.6): close()
        # gives them its grace, then cuts them (RFC §12.10.4).
        self._decided_writes: set[asyncio.Task[Any]] = set()
        # The ``cancel_task`` cancels under way, by task: the channels whose
        # calls asked are not handed the cancellation back (RFC §23.4).
        self._task_cancels: dict[str, TaskCancel] = {}
        # What the framework knows of the tasks it runs for each room: whether
        # a task is the room's, how it can end, how it ended; a cancel_task
        # call answers from it, never from the bus alone (RFC §23.4).
        self._room_tasks = RoomTaskRecord()
        # Each agent's background setup: its worker and how its results come
        # back (RFC §23.4).
        self._background_setups: dict[str, BackgroundSetup] = {}
        self._resource_leases: set[asyncio.Event] = set()
        self._resource_leases_sealed = False
        # Telemetry
        if isinstance(telemetry, _TelemetryProviderCls):
            self._telemetry: _TelemetryProviderCls = telemetry
        elif isinstance(telemetry, _TelemetryConfigCls):
            self._telemetry = telemetry.provider or NoopTelemetryProvider()
        else:
            self._telemetry = NoopTelemetryProvider()
        # A turn's request names the span it runs under (RFC §10.6.9).
        self._turns.telemetry = self._telemetry
        suppressed = (
            telemetry.suppressed_hook_triggers
            if isinstance(telemetry, _TelemetryConfigCls)
            else None
        )
        for engine in (self._hook_engine, self._turn_hooks):
            self._wire_hook_engine(engine, suppressed)
        # Propagate global metadata to the provider for searchable tags
        if (
            isinstance(telemetry, _TelemetryConfigCls)
            and telemetry.metadata
            and hasattr(self._telemetry, "_metadata")
        ):
            self._telemetry._metadata = telemetry.metadata  # ty: ignore[invalid-assignment]
        self._store._telemetry = self._telemetry  # ty: ignore[invalid-assignment]
        # Default orchestration strategy
        self._default_orchestration = orchestration
        # Rooms a discussion holds (RFC §19.7.5), by room id: the router asks
        # none of their agents at broadcast, the discussion gives the turns.
        self._discussions: dict[str, Any] = {}
        self._installing_discussions: set[str] = set()
        # The strategy each room holds, and what its install added (RFC §19.7).
        self._room_strategies: dict[str, Any] = {}
        # This process among those serving the same rooms: a discussion's lease
        # names its holder by it (RFC §19.7.5 rule 16).
        self._instance_id = uuid4().hex
        self._hook_engine._holds_discussion = self._holds_discussion
        self._hook_engine._enabled_room_hooks = self._turns.room_hook_names
        # Room-level media recording
        from roomkit.recorder._room_recorder_manager import RoomRecorderManager

        self._room_recorder_mgr = RoomRecorderManager()
        # Room timer transitions, applied by every locked section and by the
        # host's check (RFC §5.1).
        self._room_timers = TimerTransitions(
            store=self._store,
            lock_manager=self._lock_manager,
            read_room=self.get_room,
            count_closed=self._count_room_closed,
            stop_recordings=self._stop_room_recordings,
            write_record=self._emit_system_event,
            emit=self._emit_framework_event,
            fire_hook=self._fire_lifecycle_hook,
            build_context=self._build_context,
            background=self._pending_hook_tasks,
            writes=self._decided_writes,
        )

    def _start_turns(
        self, backend: TurnBackend | None, error_classes: Iterable[type[Exception]]
    ) -> None:
        """Set up the client every intelligence turn runs through (RFC §10.6),
        on *backend* or an in-memory one, with the kit's own executor wired
        from the kit; the triggers a turn fires are that executor's (RFC
        §10.6.6), the host's *error_classes* rebuilt by both (RFC §10.6.8)."""
        # The tools orchestration sets up for a channel, kept beside its
        # declaration, never on the live object (RFC §10.6.2).
        self._owner_tools = OwnerToolStore(self._channels.get)
        self._turns = registry_turn_client(
            self._channels,
            backend,
            host=KitHost(self),
            events=KitEvents(self),
            owner_tools=self._owner_tools,
            error_classes=error_classes,
        )
        # The kit's executor wires each channel the kit registers from the kit.
        self.turn_executor.use_wiring(KitWiringEnv(self))
        self._turn_hooks = self.turn_executor.hooks

    # -- Properties --

    @property
    def store(self) -> ConversationStore:
        """The backing conversation store."""
        return self._store

    def _wire_hook_engine(self, engine: HookEngine, suppressed: set[str] | None) -> None:
        """Give *engine* (the kit's, or its turn executor's, RFC §10.6.6) the
        kit's telemetry, the triggers it keeps out of it, and the framework
        events only an engine raises (RFC §8.2 ``hook_timeout``)."""
        engine._telemetry = self._telemetry
        if engine is not self._turn_hooks:
            # The executor's engine emits through the executor's wiring, as a
            # frame during a turn (RFC §10.6.2).
            engine._framework_emitter = self._emit_framework_event
        if suppressed is not None:
            engine._suppressed_triggers = suppressed

    @property
    def hook_engine(self) -> HookEngine:
        """The kit's hook engine (RFC §10.6.6): a hook on a trigger a turn
        fires registered here is the kit's own executor's."""
        return self._hook_engine

    async def _strategy_turn(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """A generation a strategy's runner asks of an agent: a turn of its
        own through the turn backend, past the agent's speak policy and its
        thinker (RFC §10.6.1)."""
        return await self._turns.run(event, binding, context, strategy_turn=True)

    async def steer(
        self,
        directive: SteeringDirective,
        channel_id: str,
        *,
        room_id: str | None = None,
        turn_id: str | None = None,
        organization_id: str | None = None,
    ) -> int:
        """Steer the running turns of an intelligence channel (RFC §21.3,
        §10.6.8), wherever they run: the turn *turn_id*; else the turns of
        *room_id*, every one for a ``Cancel`` and the most recent for any
        other directive; else the channel's most recent turn. How many loops
        it reached; a channel without steering takes a ``Cancel`` as its
        turn's cancel, counted as one.

        Only the turns this kit started are reached: with several kits owning
        rooms, steer from the one delivering the room.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): a *room_id* is read with it first, and only the turns of
        that organization's rooms are reached, a turn of another
        organization's room no more than a turn that does not run.

        Raises:
            RoomNotFoundError: *organization_id* and *room_id* are set and
                the room is missing or another organization's; nothing is
                steered.
            ValueError: both *room_id* and *turn_id* given.
        """
        if organization_id is not None and room_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        return await self._turns.steer(
            directive,
            channel_id,
            room_id=room_id,
            turn_id=turn_id,
            organization_id=organization_id,
        )

    async def answer_input(self, request_id: str, value: str) -> bool:
        """Answer the human-input request *request_id* of a turn this kit
        started, wherever the turn runs (RFC §10.6.5): ``True`` when the tool
        waiting on it reads *value*, ``False`` when the request was settled
        otherwise first (answered, rejected, timed out, withdrawn). An answer
        for a request not open yet waits for it, at most 60 seconds."""
        return await self._turns.inputs.answer(request_id, value)

    async def reject_input(self, request_id: str, reason: str = "") -> bool:
        """Reject the human-input request *request_id*: ``True`` when the tool
        reads it as a refusal (RFC §9.3), ``False`` for one settled otherwise
        first."""
        return await self._turns.inputs.reject(request_id, reason)

    def hooks_for(self, trigger: HookTrigger) -> HookEngine:
        """The engine *trigger*'s hooks register on (RFC §10.6.6): the kit's
        turn executor's for a trigger a turn fires, the kit's own otherwise.
        For code that registers hooks on triggers of both sides."""
        return self._turn_hooks if trigger in TURN_TRIGGERS else self._hook_engine

    @property
    def turn_executor(self) -> TurnExecutor:
        """The kit's own turn executor (RFC §10.6), where the hooks of the
        triggers a turn fires that the kit registers go."""
        executor = self._turns.executor
        assert executor is not None
        return executor

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
        return self._given_lock_manager

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
        """Subscribe the framework event callback to the status bus (once).

        Run by the first room activity (every context build), so a status naming
        a room fires its ON_STATUS_POSTED hooks on any kit, opened with
        ``async with`` or not (RFC §19.8). Claimed before the await: activities
        racing to be first subscribe once, and a subscription that failed is
        tried again by the next one.
        """
        if self._status_bus_subscribed:
            return
        self._status_bus_subscribed = True
        try:
            await self._status_bus.subscribe(self._status_bus_callback)
        except BaseException:
            self._status_bus_subscribed = False
            raise

    def _get_router(self) -> EventRouter:
        if self._event_router is None:
            self._event_router = EventRouter(
                channels=self._channels,
                transcoder=self._transcoder,
                max_chain_depth=self._max_chain_depth,
                telemetry=self._telemetry,
                greeting_gate_fn=self._wait_greeting_gate,
                turns=self._turns,
                runner_of=self._turn_runners.get,
                context_of=self._build_context,
            )
            self._event_router._framework_emitter = self._emit_framework_event
            self._event_router._defers = self._discussions.__contains__
        return self._event_router

    def _running_work(self) -> set[asyncio.Task[Any]]:
        """The kit's work still running (RFC §12.10.4), but what the current
        close runs under: the lanes' rounds, the calls into a room before
        their commit, the decided writes, the hook tasks and stream readers,
        the observers, the background runs, the cut turns unwinding. Read
        again at each wake: work started meanwhile is in it, the work of a
        caller that joined the close is not."""
        work = {
            *self._lanes.running(),
            *self._door_runs.running(),
            *self._decided_writes,
            *self._pending_hook_tasks,
            *self._background_runs,
            *self._hook_engine.running_observers(),
            *self._cascades.running(),
            *self._stop_steps,
        }
        # A turn the close cut ends without waiting for the close, the one a
        # close's own call runs in included: its end is waited for.
        return ({task for task in work if not task.done()} - spared_by_close()) | (
            self._cascades.unwinding() - acting_tasks()
        )

    def _cut_running_work(self) -> set[asyncio.Task[Any]]:
        """Cut the kit's work still running, but what the current close runs
        under, each task once it started so its own end runs (a run's
        terminal entry, a turn's stored text); what is left of it."""
        spared = spared_by_close()
        self._door_runs.cut_running(spared)
        self._lanes.cut(spared)
        self._cascades.cut(spared)
        running = self._running_work()
        for task in running:
            cut_once_started(task)
        return running

    def _hold_channel_work(self, task: asyncio.Task[Any]) -> None:
        """Hold *task*, the kit's work a channel runs on a task of its own (a
        session's framework event, its hook): the kit's close gives it the
        grace, then cuts it, as any running hook (RFC §12.10.4)."""
        task.add_done_callback(self._pending_hook_tasks.discard)
        self._pending_hook_tasks.add(task)

    def _spare_turns_run_under(self) -> None:
        """The turns the closing call runs in, on this kit's own executor:
        cut, they end for their owner at once, while the call runs on."""
        executor = self._turns.executor
        if executor is not None:
            executor._spare_turns_run_under()

    def _close_spared(self) -> set[asyncio.Task[Any]]:
        """The work every ``close()`` the shutdown answers is called from:
        the tasks each runs as and the kit-held runs each runs under, read
        as callers join (RFC §12.10.4)."""
        callers = self._shutdown_callers
        return {*callers.acting, *callers.under}

    def _cut_at_once(self) -> None:
        """What close() cuts as it starts (RFC §12.10.4), but the work it is
        called from: admission (the calls waiting for their room's lock),
        the deliveries queued in the lanes, every delivery's turns, the
        background runs, the verification timers."""
        spared = self._close_spared()
        self._door_runs.cut_waiting(spared)
        # Every model turn, the one a close's own call runs in included: the
        # call itself, the task the close runs as, runs on (RFC §12.10.4).
        self._spare_turns_run_under()
        self._cascades.cut(set(self._shutdown_callers.acting))
        # Then the queued deliveries: a drop cancels nothing of a cut cascade
        # again, so what the cut spared stays spared.
        self._lanes.seal()
        self._lanes.drop_placed_by(spared)
        self._cancel_verification_timers()
        for run in set(self._background_runs) - spared:
            cut_once_started(run)

    async def _cut_and_unwind(self, until: float) -> None:
        """Cut what is left of the kit's work and let it unwind until *until*
        (:data:`CUT_UNWIND_S` from the cut): what its end starts is cut in
        turn. What still runs then, ignoring its cancellation, is named in
        the log and abandoned (RFC §12.10.4)."""
        timeout = until - asyncio.get_running_loop().time()
        self._name_abandoned(await wait_for_unspared(self._cut_running_work, timeout=timeout))

    def _name_abandoned(self, left: Iterable[asyncio.Future[Any]]) -> None:
        """Note the work *left* running after its cut, as a wait of the close
        ran out: the close abandons it (:meth:`_log_abandoned`)."""
        self._abandoned.update(task for task in left if isinstance(task, asyncio.Task))

    def _log_abandoned(self) -> None:
        """Name in one line of the log, once the close's last unwind ran out,
        each task its waits left still running, ignoring its cut."""
        left = sorted(task.get_name() for task in self._abandoned if not task.done())
        if left:
            logger.warning(
                "Closing with %d task(s) still running after their cut: %s",
                len(left),
                ", ".join(left),
            )

    async def close(self) -> None:
        """Close the kit: refuse every new call, cut its work, close every
        channel, then release what they share (RFC §12.10.4).

        As it starts, every call made from then on raises
        :class:`KitClosedError`, whoever makes it (an outside caller, a
        hook, a tool, a channel's close), but the reads, which answer while
        the store is open, and the teardown calls, which return doing
        nothing. What has not started is cut at once: a delivery queued in
        a room's lane behind another (one placed in the idle lane runs from
        its placement), a call waiting for its room's lock, the background
        runs, the verification timers, every model turn (its stream and its
        tool calls; what it streamed is stored marked cancelled, an open
        tool call gets its end row). The work still running (a hook, a
        store write, a room change, a transport send, the kit's own steps
        that wait on integrator code) gets one grace of
        :data:`CLOSE_GRACE_S` seconds; then what is left is cut. A call the
        close cuts returns one result, its ``cancellation_reason``
        ``kit_closed``, with the answers already stored and each target's
        delivery (a send cut mid-way ``failed``, code ``kit_closed``,
        retryable); a call cut before its commit returns
        ``InboundResult(blocked=True, reason="kit_closed")`` or, for a call
        that returns the committed event, raises :class:`KitClosedError`;
        one cut as its commit landed reads its event back and returns it.

        What is left after the grace is given :data:`CUT_UNWIND_S` to
        unwind. Then the turn backend closes, its turns and the calls they
        made cut and waited for on its own budget (one unwind in process,
        its ``TurnTimeouts.cancel`` across Redis). Then the channels close,
        one at a time and shielded from one another, each on its own bounded
        budgets — a conference channel's bot is out of its meeting when this
        phase ends, whatever the store is doing. What their closes started
        is cut at once and given the second unwind, within which the
        framework waits for the operations the store and the lock manager
        already have (each runs under a resource lease); then it seals and
        releases them, a pool the store or the lock manager opened closed
        within a second, its connections still in use terminated (their
        own ``close()``, called by their owner, waits for them). Nothing is
        written or delivered after close() returns by work that honours its
        cut: it takes the grace, plus the turn backend's budget, plus the
        channels' and the kit's own backends' closes, each on its own
        budget, plus at most :data:`CUT_UNWIND_S` more, twice, for cut work
        that ignores its cancellation, abandoned then and named in one line
        of the log.

        Raises:
            ExceptionGroup: one or more channels failed to close. Raised only
                once the rest of the shutdown has run to completion — every
                other channel is closed and the shared resources are released
                — so nothing else is skipped on a failure's account. The
                channel that failed may still be holding its own resources (a
                bot possibly still in its meeting); the group names each one.
            ShutdownCancelledError: the shutdown was cut itself (every task
                of the process cancelled, as ``asyncio.run`` does as it ends),
                to a caller nobody cancelled.

        One shutdown per kit: the first call starts it in a task of its own,
        and every call, concurrent or later, waits for that one. A caller
        cancelled, or timed out, cancels only its own wait: the shutdown runs
        to its end. Once ended, its result answers every later call, a return
        after a success, the same ``ExceptionGroup`` after a failure; it never
        runs again — re-closing already-closed channels would raise a second,
        unrelated failure over the one already reported.

        A call from the kit's own work (a hook's body, a tool's call, a
        background run, a delivery, a session's end) waits for the shutdown
        without the shutdown waiting for that work or cutting it, whether it
        started the shutdown or joined one under way; a call from the
        shutdown's own work (a hook it fires) returns at once. The model turn
        such a call runs in is the exception: it is cut as every turn is, the
        call awaiting ``close()`` itself left to run, its open tool call's
        end row stored before the store is released.
        """
        with calling_close():
            await self._close_kit()

    async def _close_kit(self) -> None:
        """:meth:`close`, its calling task marked as calling it."""
        if self._shutdown is None:
            self._closed = True
            # Counted among the callers first: what the close cuts at once
            # spares its work (RFC §12.10.4).
            self._shutdown_callers.join()
            self._cut_at_once()
            self._shutdown = self._shutdown_callers.start(self._shut_down(), name="roomkit-close")
        elif self._shutdown in held_runs():
            # Waiting for the shutdown from its own work would wait for itself.
            return
        else:
            # Its work is spared from now on, where the shutdown already
            # waits for it included: the turn its call runs in, cut, ends.
            self._shutdown_callers.join()
            self._spare_turns_run_under()
        await _await_shutdown(self._shutdown)

    async def _shut_down(self) -> None:
        """The kit's one shutdown (:meth:`close`): its own work stops and its
        running work gets its grace, then is cut; the channels close, what
        their closes started is cut, then what they shared is released; the
        channels' failures raised last."""
        loop = asyncio.get_running_loop()
        grace_until = loop.time() + CLOSE_GRACE_S
        # The kit's own work stops within the grace too, its steps waiting
        # on integrator code (a source's stop, a delegation's on_complete).
        self._stop_kit_work()
        await wait_for_unspared(self._running_work, timeout=grace_until - loop.time())
        await self._cut_and_unwind(loop.time() + CUT_UNWIND_S)
        # A turn still open (a stream nobody reads) is given up before its
        # channel closes, within an unwind of its own.
        self._name_abandoned(await self._turns._close(loop.time() + CUT_UNWIND_S))
        channel_failures = await self._close_channels()
        # Close voice backend
        if self._voice:
            await self._voice.close()
        await self._realtime.close()
        # What the channels' closes started is cut before the store is
        # released: their grace is over. The operations the store and the
        # lock manager already have end within the same unwind.
        unwind_until = loop.time() + CUT_UNWIND_S
        await self._cut_and_unwind(unwind_until)
        self._log_abandoned()
        await self._release_shared_resources(unwind_until)
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

    def _stop_kit_work(self) -> None:
        """Stop the kit's own work while the channels it uses are still open:
        recordings, greeting gates, the delivery backend, discussions,
        delegations, sources. Each step runs as a task of its own, named for
        what it stops, within the close's one grace (RFC §12.10.4): one that
        waits on integrator code (a source's ``stop()``, a delegation's
        ``on_complete``) holds no other back, a failure is logged as its step
        ends, and the close's cut cuts every one still running."""
        # Clear stale greeting gates
        for room_id in list(self._greeting_gates):
            self._force_clear_greeting_gate(room_id)
        steps: dict[str, Coroutine[Any, Any, None]] = {
            # Room-level media recorders stop before channels close (RFC §12.11)
            "room recorders": self._close_room_recorders(),
            # The discussions' turns, their queues kept (RFC §19.7.5 rule 16)
            "discussions": self._stop_discussions(),
            "delegated tasks": self._task_runner.close(),
        }
        if self._delivery_backend is not None:
            steps["delivery backend"] = self._delivery_backend.close()
        for channel_id in list(self._sources):
            steps[f"source {channel_id}"] = self._detach_source(channel_id)
        for step, work in steps.items():
            task = start_named(
                _stop_step(step, work), name=f"roomkit-close-stop-{step.replace(' ', '-')}"
            )
            self._stop_steps.add(task)
            task.add_done_callback(self._stop_steps.discard)

    async def _close_channels(self) -> list[Exception]:
        """Close every channel, one at a time and shielded from one another;
        the failures, each noted with its channel."""
        # In sequence, but shielded from one another: they close one at a
        # time, so a close that raises would leave every channel behind it
        # holding its media — for a conference channel, a bot left sitting in
        # a meeting (RFC 12.10.4). Collected rather than
        # swallowed: the failed channel's own resources are in an unknown
        # state — its bot may still be in its conference — and a close() that
        # returns cleanly over that turns a logged error into an operational
        # and disclosure risk. Raised at the very end, so the failure of one
        # channel costs nothing else its shutdown.
        failures: list[Exception] = []
        for channel_id, channel in self._channels.items():
            try:
                await channel.close()
            except Exception as error:
                logger.exception(
                    "Channel %r failed to close; the remaining channels still close",
                    channel_id,
                )
                error.add_note(f"while closing channel {channel_id!r}")
                failures.append(error)
        return failures

    async def _release_shared_resources(self, until: float) -> None:
        """Release what the channels shared, once every one is closed: the
        store and the lock managers after the operations they already have,
        waited for until *until*, then the status bus and the telemetry."""
        # Every channel is closed and its media released. Only now wait for
        # the operations the store or the lock manager already has — they are
        # bracketed by resource leases, and this wait is what lets a channel's
        # own close() stay bounded (RFC 12.10.4).
        await self._await_resource_leases(until)
        # Close the conversation store (e.g. release a PostgreSQL pool). The
        # store's close() is idempotent and a no-op for a caller-owned pool;
        # a pool it opened is closed within its bound (RFC 12.10.4).
        await close_bounded(self._store)
        self._store_released = True
        # Close the lock manager (e.g. release an advisory-lock pool), and
        # the delivery-claim manager when the host supplied a dedicated one.
        await close_bounded(self._lock_manager)
        if self._claim_locks is not self._lock_manager:
            await close_bounded(self._claim_locks)
        # Close status bus
        await self._status_bus.close()
        # Flush telemetry (ends active spans, flushes exporter)
        self._telemetry.close()

    @contextlib.contextmanager
    def _resource_lease(self, *, read: bool = False) -> Generator[None, None, None]:
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
        on it; the seal turns it into an error that says what happened. One
        of the kit's reads (*read*) is the exception: the store answers it
        after the close while it is open, and refuses it once released (RFC
        §12.10.4). It asks so itself: work a read starts is no read.
        """
        if self._resource_leases_sealed and read:
            yield
            return
        if self._resource_leases_sealed:
            raise KitClosedError(
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

    async def _await_resource_leases(self, until: float) -> None:
        """Wait for every resource lease until *until* (the close's last
        unwind) — then seal.

        The work under a lease is already inside the store or the lock
        manager, and the media was all released before this wait begins, so
        it costs the shutdown its latency and nothing else; nothing
        integrator-owned runs under a lease. Still, the close is bounded
        (RFC 12.10.4): an operation still running at *until* is named in the
        log and the resources are released under it, its write failing. The
        loop re-reads the registry because a lease can still register while
        this waits — a straggler announcement building its room context, say.

        The seal is set with no await after the last read of the registry,
        so there is no instant at which the wait has concluded and a new
        lease could still be granted. What arrives after is refused by
        :meth:`_resource_lease`.
        """
        loop = asyncio.get_running_loop()
        while self._resource_leases and (remaining := until - loop.time()) > 0:
            waiters = [
                asyncio.ensure_future(released.wait()) for released in list(self._resource_leases)
            ]
            try:
                await asyncio.wait(waiters, timeout=remaining)
            finally:
                for waiter in waiters:
                    waiter.cancel()
        if self._resource_leases:
            logger.warning(
                "Closing with %d operation(s) the store or the lock manager still runs: "
                "the resources are released under them",
                len(self._resource_leases),
            )
        self._resource_leases_sealed = True

    async def __aenter__(self) -> RoomKit:
        if self._closed:
            # A closed kit starts nothing again (RFC §12.10.4).
            raise KitClosedError("The kit is closed: async with starts nothing")
        await self._ensure_status_bus_subscribed()
        await self._resume_verifications()
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
        newest_first: bool = False,
        include_blocked: bool = False,
        organization_id: str | None = None,
    ) -> list[RoomEvent]:
        """Query the event timeline for a room.

        Supports offset-based (``offset``/``limit``) and cursor-based
        (``after_index``/``before_index``) pagination.  When a cursor
        parameter is set, ``offset`` is ignored.

        The offset-based default is the *oldest* ``limit`` events — page 1 of
        a log reads from the beginning. Pass ``newest_first=True`` for the most
        recent ``limit`` instead (still ascending), which is the shape a
        reconnect snapshot wants: what was just said, not how the room opened.

        The page holds what the room received: a row stored ``BLOCKED`` is
        served only when ``include_blocked`` asks for it, the audit read
        (RFC §14.1).

        Args:
            room_id: Room to query.
            offset: Number of events to skip (offset-based mode), counted from
                the newest end when ``newest_first`` is set.
            limit: Maximum number of events to return.
            visibility_filter: Optional visibility value to filter by.
            after_index: Return events with ``index > after_index``.
            before_index: Return events with ``index < before_index``.
            newest_first: In offset-based mode, return the most recent
                ``limit`` events instead of the oldest. Ignored when a cursor
                is supplied.
            include_blocked: Serve the rows the room refused too (status
                ``BLOCKED``), for audit.
        """
        await self.get_room(room_id, organization_id=organization_id)
        return await self._store.list_events(
            room_id,
            offset=offset,
            limit=limit,
            visibility_filter=visibility_filter,
            after_index=after_index,
            before_index=before_index,
            newest_first=newest_first,
            event_filter=EventFilter(include_blocked=True) if include_blocked else None,
        )

    async def list_tasks(
        self,
        room_id: str,
        status: str | None = None,
        *,
        organization_id: str | None = None,
    ) -> list[Task]:
        """List tasks for a room, optionally filtered by status."""
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        return await self._store.list_tasks(room_id, status=status)

    async def list_observations(
        self, room_id: str, *, organization_id: str | None = None
    ) -> list[Observation]:
        """List observations for a room."""
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
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
        addressed_to: list[str] | None = None,
        organization_id: str | None = None,
        standalone: bool = False,
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
            visibility: Event visibility, a keyword (``all``, ``none``,
                ``transport``, ``intelligence``) or channel ids (RFC §7.3);
                committed intersected with the source binding's. ``internal``,
                which marks a record the framework wrote, raises
                ``ValueError``: an event stored for no channel takes ``none``
            provider: Optional provider/backend name for event attribution
            response_visibility: Controls where the AI's response is delivered.
                Uses the same vocabulary as visibility. None means no restriction.
                ``internal`` raises ``ValueError``, as for *visibility*.
            addressed_to: The intelligence channels this event asks to act
                (RFC §19.3). An address is set by a sender, and direct injection
                has a sender like any other entry point: a caller storing a
                message whose answer it triggers separately passes ``[]`` so the
                stored message solicits nobody, and one talking to a specific
                agent in a room that holds several names it. ``None`` leaves the
                event unaddressed — the router decides, or every eligible
                intelligence channel acts.
            parent_event_id: In-app thread parent. The locked pipeline normalises
                it to the thread root (flat two-level model); see
                :meth:`_resolve_thread_root`.
            idempotency_key: Stable de-duplication key, unique per room and
                per source channel (RFC §13.4). When set, the locked pipeline's
                idempotency check skips a re-send from the same channel that
                carries the same key — so a caller that may replay a publish (an
                outbox dispatcher redelivering after a crash) gets at-most-once
                persistence for a given key. None keeps the prior behaviour (no
                de-duplication), matching inbound events that carry no key.
            standalone: For an ``INSTRUCTION`` only: the turn it opens reads
                nothing of the room — no history, no memory provider call
                (RFC §10.1.1 step 7). Raises on any other event type.

        Called from the room's own work, the event is part of it (RFC §9.5.1):
        under the room lock (a locked hook, code under the lock) it commits
        ahead of the event being processed, takes no ticket and runs every
        hook under the lock, what its hooks inject committing once the section
        holding the lock released it; from an off-lock hook's body or a
        channel's parse it is adopted beside that event, takes no ticket and
        takes the lock for its own commit. Either way it draws on that event's
        reentry budget, and from a turn or a delivery on the budget of the
        cascade that delivers: past it, it is stored BLOCKED
        ``reentry_loop_cap`` with no hook run, and that record is returned. A
        call made within a turn is at least as deep as the turn's response
        (RFC §8.3).

        Raises:
            ChannelNotRegisteredError: the kit registers no channel
                *channel_id*, raised before anything is read (RFC §10.5),
                whether or not the room still binds it.
            RoomNotFoundError: the room is missing, or another organization's.
            ChannelNotFoundError: the room does not bind *channel_id*.
            RoomClosedError: the room's status refuses new events (RFC §5.1).
            ChangeRefusedError: an edit or a delete step 8a refused.
            ProcessTimeoutError: ``process_timeout`` passed before the commit.
            ValueError: an argument the pipeline would refuse, *content*
                holding more than five composite levels included (RFC §5.3).
        """
        # The channel an injection names is registered, checked before
        # anything is read (RFC §10.5): one the room still binds once
        # unregistered included. The framework's own records take its door
        # (_send_event) and are not injections: a task's posted result is the
        # agent's message, which any process posts (§23.3, §19.7.5 rule 19).
        if channel_id not in self._channels:
            raise ChannelNotRegisteredError(f"Channel {channel_id} not registered")
        return await self._send_event(
            room_id,
            channel_id,
            content,
            event_type=event_type,
            chain_depth=chain_depth,
            participant_id=participant_id,
            metadata=metadata,
            # Only the framework writes ``internal`` (RFC §7.3), refused here
            # before anything is read, where the answers may go included.
            visibility=parse_given_visibility(visibility),
            provider=provider,
            response_visibility=(
                response_visibility
                if response_visibility is None
                else parse_given_visibility(response_visibility)
            ),
            created_at=created_at,
            parent_event_id=parent_event_id,
            idempotency_key=idempotency_key,
            addressed_to=addressed_to,
            organization_id=organization_id,
            standalone=standalone,
        )

    async def post(
        self,
        room_id: str,
        content: str | EventContent,
        *,
        participant_id: str | None = None,
        addressed_to: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
        channel_id: str | None = None,
        organization_id: str | None = None,
        idempotency_key: str | None = None,
        chain_depth: int = 0,
    ) -> RoomEvent:
        """Write to a room's people in the host's own name (RFC §10.5).

        A notice, a workflow's question, a reminder: what the application says,
        not what an agent says. It comes from a :class:`HostChannel`, a
        transport of type ``HOST``, so the room reads it as a party's message
        and broadcasts it to its transports (an SMS to the room's
        correspondent). :meth:`deliver` is the other way in: content for the
        room's agents.

        Args:
            room_id: The room to write to.
            content: The text, or any event content.
            participant_id: Who speaks: a participant of the room the host
                channel the post comes from reaches (its ``channel_id`` or
                ``connected_via`` names it), by their id or their identity's,
                as a turn's sender is read; readers, agents included, are
                shown their name. An id names its participant only on the
                channels they are reached through: any other is refused,
                nothing written. ``add_member(room_id, "host", ...)`` makes a
                member one, ``"host"`` the host channel's id.
            addressed_to: The agents asked to answer. None by default: the
                host's message solicits no agent unless it names one.
            metadata: Where the message comes from (a workflow, its step, a
                job), stored on the event.
            channel_id: A registered ``HostChannel`` to speak from; by
                default the kit's own, registered under ``"host"`` on first
                use.
            organization_id: The organization the caller acts for (RFC
                §17.2).
            idempotency_key: A key the host channel holds in the room is a
                replay: the post returns the event it committed, sends
                nothing again and, from the room's own work, draws no pass
                (RFC §13.4, §9.5.1). A blank key is no key.
            chain_depth: The chain the post continues (RFC §8.3), as
                :meth:`send_event`'s; 0 opens one. Work that follows an
                agent's answer once the answer's own work is over, such as a
                task a hook starts, posts one deeper than the answer, so its
                round trips with the agent end at ``max_chain_depth``. Made
                while a turn runs, the post is at least as deep as the turn's
                response.

        A room that never bound the host channel is given it on its first
        post, an ordinary attachment, decided in one section under the room
        lock: a binding the integrator made, before or concurrently, keeps its
        settings. A room the host channel was detached from is refused, not
        attached again, and so is a room that refuses events; nothing is
        written either way. One ``process_timeout`` bounds the whole call, the
        wait for that lock included; once the lock is held, the attach runs to
        its end, a cancelled caller's included, and stays when the deadline
        passes during it (RFC §13.6). A first post made from the room's own
        work draws its pass before the attach (RFC §9.5.1): past the budget,
        it is stored ``BLOCKED`` with ``reentry_loop_cap`` and binds nothing.

        Returns:
            The committed event, as :meth:`send_event` returns it.

        Raises:
            ChannelNotRegisteredError: *channel_id* names no registered channel.
            ChannelNotFoundError: the host channel was detached from the room.
            ParticipantNotFoundError: *participant_id* names nobody the host
                channel reaches (RFC §10.5, §5.5).
            ProcessTimeoutError: ``process_timeout`` passed before the commit.
                Nothing is written when it passed before a first post's attach
                held the room lock. Once the lock is held, the attach runs to
                its end and stays, its binding and its ``CHANNEL_ATTACHED``
                record, even when the deadline passes during it: only the
                message is not written.
            RoomClosedError: the room's status refuses new events (RFC §5.1).
            RoomNotFoundError: the room does not exist, or is another
                organization's.
            ValueError: *channel_id* names a channel that is not of type
                ``HOST``, ``"host"`` is taken by another channel, or an
                argument :meth:`send_event` refuses (a key in the framework's
                namespace, content past five composite levels, a negative
                *chain_depth*).
        """
        return await _host_post.post(
            self,
            room_id,
            content,
            participant_id=participant_id,
            addressed_to=addressed_to,
            metadata=metadata,
            channel_id=channel_id,
            organization_id=organization_id,
            idempotency_key=idempotency_key,
            chain_depth=chain_depth,
        )

    async def _send_event(
        self,
        room_id: str,
        channel_id: str,
        content: Any,
        *,
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
        addressed_to: list[str] | None = None,
        organization_id: str | None = None,
        standalone: bool = False,
        deadline: float | None = None,
        first_post: _host_post.FirstPost | None = None,
    ) -> RoomEvent:
        """:meth:`send_event` for the framework's own records too, whose
        *visibility* may be ``internal`` (an interruption record, RFC §6.4,
        §7.3).

        The framework's door for the records it writes itself, never the
        integrator's: integrator code calls :meth:`send_event`, which refuses
        ``internal`` before anything is read (RFC §7.3). *deadline* is the
        call's when a caller started it earlier (:meth:`post`, whose first
        attach spends it too); ``process_timeout`` from now otherwise.
        :meth:`post` passes *first_post* for a room that does not bind its
        host channel yet, once it has read the room: the event comes from its
        source, and its attach binds the channel once the call has drawn its
        pass (RFC §10.5, §9.5.1).
        """
        # One deadline from the call's start to its decision (RFC §13.6,
        # §10.5): the room's read, the off-lock check, the wait for its turn
        # and for the room lock, and the locked gates.
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + self._process_timeout
        # A sixth composite level is refused before anything is read (RFC §5.3).
        refuse_too_deep(content)
        # The visibility grammar (RFC §7.3): an invalid scope is refused.
        visibility = parse_visibility(visibility)
        if response_visibility is not None:
            response_visibility = parse_visibility(response_visibility)
        # A blank key is no key (RFC §13.4), as the event's model reads it.
        idempotency_key = _no_blank_key(idempotency_key)

        # An instruction the pipeline would refuse (RFC §10.1.1) is refused
        # here, loudly, before anything is read: this API's contract is the
        # committed event, and an instruction that was never going to be
        # delivered has none to return.
        if event_type == EventType.INSTRUCTION:
            if not names_an_agent(addressed_to, self._channels):
                raise ValueError(
                    "An INSTRUCTION must name an agent it directs (addressed_to): "
                    "an intelligence channel the kit registers"
                )
            # An empty or blank key is no key, as on the inbound path.
            if idempotency_key is not None:
                raise ValueError("An INSTRUCTION is never stored and takes no idempotency_key")
            # The parameter's word alone: a metadata key of the same name is
            # dropped, never honoured (the instruction is never stored).
            metadata = {k: v for k, v in (metadata or {}).items() if k != STANDALONE}
            if standalone:
                metadata[STANDALONE] = True
        elif standalone:
            raise ValueError("standalone applies to an INSTRUCTION only")
        refuse_reserved(idempotency_key)

        # The kit's close cuts it before its commit (RFC §12.10.4).
        async with self._door_runs.run("event"):
            await self._ensure_status_bus_subscribed()
            if first_post is None:
                binding = await self._injection_binding(
                    room_id, channel_id, organization_id, deadline
                )
                source = _direct_source(binding, participant_id, provider)
            else:
                source = first_post.source
            fields: dict[str, Any] = dict(
                room_id=room_id,
                type=event_type,
                source=source,
                content=content,
                # A call made within a turn is at least as deep as the turn's
                # response (RFC §8.3, §10.6).
                chain_depth=max(chain_depth, chain_floor()),
                parent_event_id=parent_event_id,
                status=EventStatus.DELIVERED,
                # The key the pipeline reserves (RFC §5.2) is not the caller's.
                metadata=without_fingerprint(metadata or {}),
                visibility=visibility,
                response_visibility=response_visibility,
                addressed_to=addressed_to,
                idempotency_key=idempotency_key,
            )
            if created_at is not None:
                fields["created_at"] = created_at
            return await self._enter_injection(RoomEvent(**fields), deadline, first_post)

    async def _enter_injection(
        self, event: RoomEvent, deadline: float, first_post: _host_post.FirstPost | None
    ) -> RoomEvent:
        """A direct injection of *event* entering its room (RFC §10.5): as
        the room's own work, it draws its pass first (§9.5.1). A first post
        attaches its host channel only once the pass is drawn, so past the
        budget it binds nothing."""
        room_id, channel_id = event.room_id, event.source.channel_id
        entry = self._room_work.entry(room_id)
        if entry.charged:
            # A replay of a key its channel holds is no pass: it is read
            # before the budget, and draws nothing (RFC §13.4, §9.5.1).
            replayed = await self._by_deadline(
                self._replay_of_key(room_id, event), room_id, channel_id, deadline
            )
            if replayed is not None:
                return replayed
            if not entry.draw():
                # Past the budget of the work that made it, it is not a pass:
                # its hooks never run (RFC §8.3, §9.5.1).
                return await self._store_entry_past_cap(room_id, event)
        if first_post is not None:
            await first_post.attach()
        return await self._inject(event, entry, deadline)

    async def _inject(self, event: RoomEvent, entry: Entry, deadline: float) -> RoomEvent:
        """Direct injection's pipeline for *event* (RFC §10.5), *entry* what
        it is to its room's work (§9.5.1): the same pipeline as an inbound
        message, its off-lock check, its locked section and commit, then its
        delivery set, reentry passes and AFTER_BROADCAST hooks.

        Raises what refused it before its commit point, nothing written.
        """
        from roomkit.telemetry.base import SpanKind
        from roomkit.telemetry.context import get_current_span, reset_span, set_current_span

        room_id, channel_id = event.room_id, event.source.channel_id
        telemetry = self._telemetry
        span_id = telemetry.start_span(
            SpanKind.INBOUND_PIPELINE,
            "framework.send_event",
            parent_id=get_current_span(),
            room_id=room_id,
            channel_id=channel_id,
            attributes={"event_type": str(event.type)},
        )
        token = set_current_span(span_id, telemetry_ctx=telemetry.get_span_context(span_id))
        cascade = self._cascades.new(room_id, reentry_budget=entry.budget)
        try:
            result, handed = await self._injection_section(event, entry, cascade, deadline)
            # Past its commit: the kit's close no longer cuts the call, but its
            # delivery (RFC §12.10.4).
            self._door_runs.committed()
            # What its hooks injected commits once its section released the
            # lock, each in a section of its own (RFC §9.4), before this call
            # returns; nested in a pass's section, that section runs them
            # once it releases the lock its caller holds.
            await self._finish_queued(handed)
            _raise_unless_committed(result, room_id)
            if isinstance(result.event, RoomEvent):
                event = result.event
            # The caller observes its event's delivery-set completion (RFC
            # §10.1 step 18) — the wait short-circuits when this call was made
            # from inside the room's own lane or under its lock, where waiting
            # would deadlock; delivery then follows in lane order. A call the
            # room's work adopted answers at its commit, its delivery behind
            # that work's (§9.5.1). Streaming AI responses to a
            # directly-injected event are consumed outside the lane, exactly
            # like the inbound path, or a streaming provider's reply would be
            # generated and then silently dropped.
            await self._finish_cascade(cascade, room_id, detached=entry.adopted)

            telemetry.end_span(span_id)
        except BaseException as exc:
            # The caller owns the cascade: a caller cancelled stops its
            # turns, off the lock, as on the inbound path.
            await cascade.abandon("caller_cancelled")
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise
        finally:
            reset_span(token)
        return event

    async def _injection_section(
        self, event: RoomEvent, entry: Entry, cascade: DeliveryCascade, deadline: float
    ) -> tuple[InboundResult, asyncio.Task[None] | None]:
        """A direct injection's off-lock check and locked section (RFC §10.5,
        §9.5.1): its result, and the passes of what its hooks injected to
        wait for once the lock is released.

        The off-lock check runs before the lock, when a ``needs_lock=False``
        hook applies and the room's work did not adopt the event; its ticket
        is held until the locked pass has committed. The check and the waits
        spend the same pre-commit budget as the locked gates (§13.6).
        """
        room_id = event.room_id
        queued = self._queue_of(entry)
        async with contextlib.AsyncExitStack() as stack:
            try:
                async with asyncio.timeout_at(deadline):
                    precheck = await stack.enter_async_context(
                        self._admission.admitted(
                            room_id, event, None, budget=entry.budget, adopted=entry.adopted
                        )
                    )
                    await stack.enter_async_context(self._lock_manager.locked(room_id))
            except TimeoutError:
                result = await self._refuse_on_timeout(room_id, channel_id=event.source.channel_id)
                return result, None
            # No pre-lock context is carried: the off-lock check, when one
            # ran, built its own, and the locked pass rebuilds one under the
            # lock.
            try:
                result = await self._process_locked(
                    event,
                    room_id,
                    None,
                    cascade,
                    queued=queued,
                    precheck=precheck,
                    deadline=deadline,
                )
            finally:
                # Decided, or its caller gone: what its hooks injected no
                # longer depends on this caller (RFC §9.4, §13.6).
                handed = self._hand_off_decided(entry, queued, room_id, cascade)
            return result, handed

    async def _store_entry_past_cap(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Store a direct injection the room's work made past the reentry
        budget it draws on as its BLOCKED record, ``reentry_loop_cap``, with
        the visibility its pass would have committed it with, no hook run
        (RFC §8.3, §7.3): the committed event the call returns. It waits for
        the room lock alone; an instruction, never stored, is returned
        uncommitted. A replay of a key its channel holds is no new pass: it
        returns what the key committed, as a replay within the budget does
        (RFC §13.4).

        Raises:
            RoomClosedError: the room's status refuses events (RFC §5.1);
                nothing is stored.
        """
        logger.warning(
            "An event sent into room %s from its own work is past the reentry budget: "
            "stored BLOCKED",
            room_id,
            extra={"room_id": room_id, "channel_id": event.source.channel_id},
        )
        record = event.model_copy(
            update={"status": EventStatus.BLOCKED, "blocked_by": REENTRY_LOOP_CAP}
        )
        if event.type == EventType.INSTRUCTION:
            return record
        async with self._lock_manager.locked(room_id):
            room = await self._room_timers.room_due(room_id)
            if _refuses_writes(room):
                await self._refuse_closed_room(
                    room_id,
                    status=room.status if room is not None else None,
                    operation="inbound",
                    event=record,
                )
                raise RoomClosedError(f"Room {room_id} does not accept new events")
            replayed = await self._replay_of_key(room_id, event)
            if replayed is not None:
                return replayed
            record = await self._with_thread_root(room_id, record)
            return await self._commit_indexed(room_id, await self._scoped_record(record))

    async def _replay_of_key(self, room_id: str, event: RoomEvent) -> RoomEvent | None:
        """What *event*'s source channel committed in the room with its
        idempotency key, which a direct injection of the room's own work
        returns rather than draw a pass or store anything (RFC §13.4,
        §9.5.1): read before its budget, and again under the room lock before
        a record past the cap is stored. ``None`` for a key not taken;
        *event* itself when the store holds the key but cannot read its event
        back, as the pipeline's replay returns it."""
        key, channel_id = event.idempotency_key, event.source.channel_id
        if not key or not await self._store.check_idempotency(room_id, key, channel_id=channel_id):
            return None
        original = await self._store.get_event_by_idempotency_key(
            room_id, key, channel_id=channel_id
        )
        logger.info(
            "Duplicate event %s from the room's own work: replayed, no pass drawn",
            key,
            extra={"room_id": room_id, "idempotency_key": key},
        )
        return original if original is not None else event

    async def _injection_binding(
        self, room_id: str, channel_id: str, organization_id: str | None, deadline: float
    ) -> ChannelBinding:
        """The binding a direct injection's channel holds in its room, read
        scoped to *organization_id* (RFC §10.5, §17.2) within the call's
        *deadline* (§13.6), past which it raises once the
        ``process_timeout`` framework event is emitted.

        Raises:
            RoomNotFoundError: the room is missing, or another organization's.
            ChannelNotFoundError: the room does not bind the channel.
            ProcessTimeoutError: the deadline passed first; nothing written.
        """
        return await self._by_deadline(
            self._scoped_binding(room_id, channel_id, organization_id),
            room_id,
            channel_id,
            deadline,
        )

    async def _scoped_binding(
        self, room_id: str, channel_id: str, organization_id: str | None
    ) -> ChannelBinding:
        """:meth:`_injection_binding`'s reads."""
        await self.get_room(room_id, organization_id=organization_id)
        return await self._get_binding(room_id, channel_id)

    async def _by_deadline[T](
        self, work: Coroutine[Any, Any, T], room_id: str, channel_id: str, deadline: float
    ) -> T:
        """*work*, a direct injection's read before its commit, within the
        call's *deadline* (RFC §13.6), past which it raises once the
        ``process_timeout`` framework event is emitted.

        Raises:
            ProcessTimeoutError: the deadline passed first; nothing written.
        """
        timeout = asyncio.timeout_at(deadline)
        try:
            async with timeout:
                return await work
        except TimeoutError:
            if not timeout.expired():
                raise
            await self._refuse_on_timeout(room_id, channel_id=channel_id)
            raise ProcessTimeoutError(
                f"Room {room_id}: process_timeout expired before the event was committed"
            ) from None

    # -- WebSocket lifecycle --

    async def connect_websocket(
        self,
        channel_id: str,
        connection_id: str,
        send_fn: SendFn,
        *,
        room_id: str,
        stream_send_fn: StreamSendFn | None = None,
        organization_id: str | None = None,
    ) -> None:
        """Register a WebSocket connection for a room and emit framework event.

        ``room_id`` says which conversation this socket belongs to. It is
        required because one channel instance can serve several rooms, and a
        connection the channel cannot place would otherwise have to receive
        every room's events. Add more rooms to the same socket with
        :meth:`subscribe_websocket`.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): the room is read with it before the socket is registered, and
        another organization's room raises :class:`RoomNotFoundError` with
        no connection registered and no event emitted; the connection is
        registered for the organization, as
        :meth:`WebSocketChannel.register_connection` registers one, and
        receives no room of another. Unset, no room is read, as before.

        Raises:
            RoomNotFoundError: *organization_id* is set and the room is
                missing or another organization's.
            ValueError: *organization_id* is set and *connection_id* is
                registered for another organization, or unscoped.
        """
        await self._ensure_status_bus_subscribed()
        channel = self._channels.get(channel_id)
        if not isinstance(channel, WebSocketChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered WebSocket channel"
            )
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        channel.register_connection(
            connection_id,
            send_fn,
            room_id=room_id,
            stream_send_fn=stream_send_fn,
            organization_id=organization_id,
        )
        await self._emit_framework_event(
            "channel_connected",
            channel_id=channel_id,
            room_id=room_id,
            data={"connection_id": connection_id, "room_id": room_id},
        )

    def subscribe_websocket(
        self,
        channel_id: str,
        connection_id: str,
        room_id: str,
        *,
        organization_id: str | None = None,
    ) -> None:
        """Also deliver *room_id* to an already-connected socket.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): the call reads no room, and only a connection registered for
        it is subscribed; a room of another organization, or of none,
        delivers nothing to a connection registered for one
        (:meth:`WebSocketChannel.subscribe`).

        Raises:
            ValueError: *organization_id* is set and the connection is not
                registered for it; nothing changes.
        """
        channel = self._channels.get(channel_id)
        if not isinstance(channel, WebSocketChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered WebSocket channel"
            )
        channel.subscribe(connection_id, room_id, organization_id=organization_id)

    def unsubscribe_websocket(
        self,
        channel_id: str,
        connection_id: str,
        room_id: str,
        *,
        organization_id: str | None = None,
    ) -> None:
        """Stop delivering *room_id* to a connected socket; with
        *organization_id* (RFC §17.2), only a connection registered for it,
        ``ValueError`` otherwise (:meth:`WebSocketChannel.unsubscribe`)."""
        channel = self._channels.get(channel_id)
        if not isinstance(channel, WebSocketChannel):
            raise ChannelNotRegisteredError(
                f"Channel {channel_id} is not a registered WebSocket channel"
            )
        channel.unsubscribe(connection_id, room_id, organization_id=organization_id)

    async def disconnect_websocket(
        self, channel_id: str, connection_id: str, *, organization_id: str | None = None
    ) -> None:
        """Unregister a WebSocket connection and emit framework event.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): only a connection registered for it is unregistered
        (:meth:`WebSocketChannel.unregister_connection`).

        Raises:
            ValueError: *organization_id* is set and the connection is not
                registered for it; nothing changes and no event is emitted.
        """
        channel = self._channels.get(channel_id)
        if isinstance(channel, WebSocketChannel):
            channel.unregister_connection(connection_id, organization_id=organization_id)
        await self._emit_framework_event(
            "channel_disconnected",
            channel_id=channel_id,
            data={"connection_id": connection_id},
        )

    # -- Read tracking --

    async def mark_read(
        self,
        room_id: str,
        channel_id: str,
        event_id: str,
        *,
        organization_id: str | None = None,
    ) -> None:
        """Mark an event as read for a channel."""
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        await self._store.mark_read(room_id, channel_id, event_id)

    async def mark_all_read(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> None:
        """Mark all events as read for a channel."""
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        await self._store.mark_all_read(room_id, channel_id)

    async def list_read_markers(
        self, room_id: str, *, organization_id: str | None = None
    ) -> dict[str, int]:
        """Return every channel's read high-water-mark (event index) in a room.

        Maps ``channel_id`` -> the highest read event ``index``. With one
        channel per member, this is the per-member read position used to
        aggregate "seen by" receipts; resolve channels to members via the
        bindings/participants (see :meth:`list_members`).
        """
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        return await self._store.list_read_markers(room_id)


# -- Admission (RFC §12.10.4) --
#
# Once close() starts, the kit's calls fall in three classes, coroutines and
# synchronous calls alike: the reads work (the store the kit closed refuses
# them once released), the teardown calls return without doing anything (the
# close tears everything down), and every other call is refused, whoever makes
# it: an outside caller, or the kit's own work still running (a hook, a tool,
# a channel's close). A call taken before runs as a call the close cuts
# (core._closing.DoorRuns). The classes are put on each public call here, in
# one place, rather than repeated at the top of each; the framework's own
# steps call the private bodies.

_READS = frozenset(
    {
        "get_room",
        "get_timeline",
        "get_binding",
        "list_bindings",
        "list_members",
        "is_member",
        "list_observations",
        "list_read_markers",
        "list_tasks",
        "regenerate_target",
        "room_recordings",
        "room_strategy_is_current",
        "sender_verification",
        "source_health",
        "verification_of",
    }
)
"""The reads: they write and deliver nothing, taken during the close and
after it while the store is open."""

_TEARDOWN: dict[str, Any] = {
    "cancel_delivery": 0,
    "cancel_task": False,
    "disconnect_websocket": None,
    "disconnect_voice": None,
    "disconnect_video": None,
    "leave": None,
    "unsubscribe_room": False,
}
"""The teardown calls, each with what it returns, doing nothing, once the
close started: the close tears down what they would."""

_HANDING_OFF = frozenset(
    {
        "answer_input",
        "bind_voice_session",
        "channel_call",
        "connect_video",
        "connect_voice",
        "connect_websocket",
        "delegate",
        "deliver",
        "join",
        "process_webhook",
        "reject_input",
        "send_greeting",
        "start_task",
        "steer",
        "synthesize",
        "transcribe",
    }
)
"""The calls that hand their work to the kit (a delegation, a turn, a session,
a call into the room they make) and wait for it, refused once the close
started like any other: they are not cut as running work, since what they
wait for is cut, and ends them; one waiting for the very work that closes
the kit (a delegation whose tool closes it) would hold the close its grace."""

_SYNC_READS = frozenset(
    {
        "get_channel",
        "hooks_for",
        "identity_enabled_for",
        "list_channels",
        "list_sources",
        "room_strategy",
        "speak_queue",
    }
)
"""The synchronous reads: they answer during the close and after it."""

_SYNC_TEARDOWN: dict[str, Any] = {"unsubscribe_websocket": None}
"""The synchronous teardown calls, each with what it returns, doing nothing,
once the close started."""

_TAKEN_WHILE_CLOSING = frozenset({"close", *_READS, *_TEARDOWN, *_SYNC_READS, *_SYNC_TEARDOWN})
"""The calls a closing kit does not refuse: a close joins the shutdown."""

_CUT_RESULTS: dict[str, Callable[[KitClosedError], Any]] = {
    "process_inbound": inbound_cut_result,
    "regenerate_response": inbound_cut_result,
    "send_event": event_cut_result,
    "post": event_cut_result,
    "commit_event": event_cut_result,
}
"""What a call the close cut returns, by the result it has a place for; any
other raises :class:`KitClosedError`."""


def _work_refused_once_closed(
    name: str, method: Callable[..., Awaitable[Any]]
) -> Callable[..., Awaitable[Any]]:
    """The call *name* (*method*), refused with :class:`KitClosedError` once
    the kit's close started, and run as a call the close cuts: at once while
    it waits for a room's lock, once the grace ran out while it runs; a call
    that hands its work off is not cut (:data:`_HANDING_OFF`)."""
    cut_result = _CUT_RESULTS.get(name)
    handing_off = name in _HANDING_OFF

    @functools.wraps(method)
    async def call(kit: RoomKit, /, *args: Any, **kwargs: Any) -> Any:
        if kit._closed:
            raise KitClosedError(f"The kit is closing: {name}() is refused")
        if handing_off:
            return await method(kit, *args, **kwargs)
        try:
            async with kit._door_runs.run(name, own=True):
                return await method(kit, *args, **kwargs)
        except KitClosedError as cut:
            if cut_result is None:
                raise
            return cut_result(cut)

    return call


def _read_while_open(
    name: str, method: Callable[..., Awaitable[Any]]
) -> Callable[..., Awaitable[Any]]:
    """The read *name* (*method*), taken during the close and after it, but
    against a store the kit closed: :class:`KitClosedError`, not the
    store's own error."""

    @functools.wraps(method)
    async def call(kit: RoomKit, /, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(kit, *args, **kwargs)
        except RuntimeError as error:
            if not kit._store_released:
                raise
            raise KitClosedError(f"The kit closed its store: {name}() reads nothing") from error

    return call


def _quiet_once_closed(
    name: str, method: Callable[..., Awaitable[Any]], value: Any
) -> Callable[..., Awaitable[Any]]:
    """The teardown call *name* (*method*), returning *value* without doing
    anything once the kit's close started."""

    @functools.wraps(method)
    async def call(kit: RoomKit, /, *args: Any, **kwargs: Any) -> Any:
        if kit._closed:
            return value
        return await method(kit, *args, **kwargs)

    return call


def _quiet_once_closed_sync(
    name: str, method: Callable[..., Any], value: Any
) -> Callable[..., Any]:
    """The synchronous teardown call *name* (*method*), returning *value*
    without doing anything once the kit's close started."""

    @functools.wraps(method)
    def call(kit: RoomKit, /, *args: Any, **kwargs: Any) -> Any:
        if kit._closed:
            return value
        return method(kit, *args, **kwargs)

    return call


def _refused_once_closed_sync(name: str, method: Callable[..., Any]) -> Callable[..., Any]:
    """The synchronous call *name* (*method*) (a channel's registration, a
    hook's, a WebSocket's subscription), refused with
    :class:`KitClosedError` once the kit's close started."""

    @functools.wraps(method)
    def call(kit: RoomKit, /, *args: Any, **kwargs: Any) -> Any:
        if kit._closed:
            raise KitClosedError(f"The kit is closing: {name}() is refused")
        return method(kit, *args, **kwargs)

    return call


for _name, _call in inspect.getmembers(RoomKit, inspect.iscoroutinefunction):
    if _name.startswith("_") or _name == "close":
        continue
    if _name in _READS:
        setattr(RoomKit, _name, _read_while_open(_name, _call))
    elif _name in _TEARDOWN:
        setattr(RoomKit, _name, _quiet_once_closed(_name, _call, _TEARDOWN[_name]))
    else:
        setattr(RoomKit, _name, _work_refused_once_closed(_name, _call))
for _name, _call in inspect.getmembers(RoomKit, inspect.isfunction):
    if _name.startswith("_") or inspect.iscoroutinefunction(_call) or _name in _SYNC_READS:
        continue
    if _name in _SYNC_TEARDOWN:
        setattr(RoomKit, _name, _quiet_once_closed_sync(_name, _call, _SYNC_TEARDOWN[_name]))
    else:
        setattr(RoomKit, _name, _refused_once_closed_sync(_name, _call))
del _name, _call
