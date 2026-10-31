"""HooksApiMixin — hook decorators, webhook processing, and delivery status."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.hooks import (
    AsyncHookFn,
    HookRegistration,
    SyncHookFn,
    hook_decorator,
)
from roomkit.core.inbound_router import binding_admits, routed_recipient
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.context import RoomContext
from roomkit.models.delivery import DeliveryStatus
from roomkit.models.enums import (
    ChannelType,
    EventDirection,
    EventType,
    HookExecution,
    HookTrigger,
    RoomStatus,
)
from roomkit.models.store_filter import RoomFilter

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine, IdentityHookRegistration
    from roomkit.core.mixins.helpers import FrameworkEventHandler, IdentityHookFn
    from roomkit.models.delivery import InboundMessage, InboundResult
    from roomkit.providers.sms.meta import WebhookMeta
    from roomkit.store.base import ConversationStore


logger = logging.getLogger("roomkit.framework")


@runtime_checkable
class HooksApiHost(Protocol):
    """Contract: capabilities a host class must provide for HooksApiMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store for room lookups and delivery status.
        _channels: Registry of channel-id to :class:`Channel` instances.
        _hook_engine: Hook engine for registration and dispatch.
        _event_handlers: Framework event handler registry.
        _identity_hooks: Identity hook registry.

    Methods provided by InboundMixin (or equivalent):
        process_inbound: Feed an inbound message into the framework pipeline.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _event_handlers: list[tuple[str, FrameworkEventHandler]]
    _identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]]

    async def process_inbound(
        self,
        message: InboundMessage,
        *,
        room_id: str | None = None,
        defer_delivery: bool = False,
        organization_id: str | None = None,
    ) -> InboundResult: ...


class HooksApiMixin(HelpersMixin):
    """Hook decorators, webhook processing, and delivery status.

    Host contract: :class:`HooksApiHost`.
    """

    _store: ConversationStore
    _hook_engine: HookEngine
    _turns: Any  # RoomKit's TurnClient
    turn_executor: Any  # RoomKit's own TurnExecutor
    _event_handlers: list[tuple[str, FrameworkEventHandler]]
    _identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]]

    # Stub for cross-mixin call — implemented by InboundMixin.
    async def process_inbound(
        self,
        message: InboundMessage,
        *,
        room_id: str | None = None,
        defer_delivery: bool = False,
        organization_id: str | None = None,
    ) -> InboundResult: ...

    def hook(
        self,
        trigger: HookTrigger,
        execution: HookExecution = HookExecution.SYNC,
        priority: int = 0,
        name: str = "",
        timeout: float = 30.0,
        channel_types: set[ChannelType] | None = None,
        channel_ids: set[str] | None = None,
        directions: set[EventDirection] | None = None,
        event_types: set[EventType] | None = None,
        fail_closed: bool = False,
        needs_lock: bool = True,
    ) -> Callable[..., Any]:
        """Decorator to register a global hook.

        A filtered-out hook is not called at all — neither its body nor its
        timeout budget. A trigger that fires for every event a turn produces
        (``BEFORE_BROADCAST`` runs once per text segment and once per
        tool-call event) is where ``event_types`` pays: a hook that only
        shapes tool calls declares it instead of guarding on ``event.type``
        and being invoked on every segment of every turn.

        Args:
            trigger: When the hook fires (BEFORE_BROADCAST, AFTER_BROADCAST, etc.)
            execution: SYNC (can block/modify) or ASYNC (fire-and-forget)
            priority: Lower numbers run first (default: 0)
            name: Optional name for logging and removal
            timeout: Max execution time in seconds (default: 30.0)
            channel_types: Only run for events from these channel types (None = all)
            channel_ids: Only run for events from these channel IDs (None = all)
            directions: Only run for events with these directions (None = all)
            event_types: Only run for events of these types (None = all)
            fail_closed: SYNC only (``ValueError`` on an ASYNC hook, which
                cannot block) — a timeout, an exception or an unusable
                result blocks the payload instead of letting it through, with
                ``reason="hook_timeout:<name>"`` / ``hook_error:<name>`` /
                ``hook_invalid_result:<name>`` (RFC §9.3). For content checks.
            needs_lock: SYNC ``BEFORE_BROADCAST`` only — ``False`` runs the
                hook before the room lock is taken, so its I/O does not hold
                the room; arrival order is kept by a per-room admission ticket
                (RFC §9.5.1). Only for a hook that reads the event, not the
                room's state. Raises ``ValueError`` on another trigger, or when
                a locked hook would be ordered before it by priority.

        A hook on a trigger a turn fires (``ON_TOOL_CALL``,
        ``BEFORE_TOOL_USE``..., RFC §10.6.6) is the kit's own executor's: it
        fires for the agents the kit runs. One that a channel the kit holds
        by its declaration alone could fire (its turns run on another
        executor) raises ``ValueError``: register it on that executor,
        ``executor.hook(...)``, or leave the channel out of ``channel_ids``.
        """

        return hook_decorator(
            self._hook_engine.register,
            trigger,
            execution=execution,
            priority=priority,
            name=name,
            timeout=timeout,
            channel_types=channel_types,
            channel_ids=channel_ids,
            directions=directions,
            event_types=event_types,
            fail_closed=fail_closed,
            needs_lock=needs_lock,
        )

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
        directions: set[EventDirection] | None = None,
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
        from roomkit.core.hooks import IdentityHookRegistration

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
        if not inspect.iscoroutinefunction(fn):
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
                # Named after the handler by the engine, unique (RFC §9.1).
                fn=_hook_fn,
            )
        )
        return fn

    async def process_webhook(
        self,
        meta: WebhookMeta,
        channel_id: str,
        organization_id: str | None = None,
    ) -> None:
        """Process any SMS provider webhook automatically.

        This is the simplest integration method. It handles:
        - Inbound messages → process_inbound() with all hooks
        - Delivery status → process_delivery_status() with ON_DELIVERY_STATUS hooks
        - Unknown webhooks → silently ignored (acknowledged)

        It does NOT authenticate the request. Verifying that the payload came
        from the provider is the endpoint's job and has to happen before this
        is called — nothing downstream can tell a forged webhook from a real
        one, and a webhook URL is public by construction (RFC §17.1).

        Args:
            meta: WebhookMeta from extract_sms_meta().
            channel_id: The channel ID for inbound messages.

        Example:
            @app.post("/webhooks/sms/{provider}/inbound")
            async def sms_webhook(provider: str, request: Request):
                raw = await request.body()
                if not sms_provider.verify_signature(
                    raw,
                    request.headers.get("X-Twilio-Signature", ""),
                    url=str(request.url),
                ):
                    raise HTTPException(status_code=403)

                meta = extract_sms_meta(provider, await request.json())
                await kit.process_webhook(meta, channel_id=f"sms-{provider}")
                return {"ok": True}

        ``organization_id`` scopes the room an inbound message lands in, as
        on :meth:`process_inbound`, and the room a delivery status is
        dispatched in (RFC §17.2): a host that resolved its tenant from the
        webhook names it here.
        """
        if meta.is_inbound:
            inbound = meta.to_inbound(channel_id)
            await self.process_inbound(inbound, organization_id=organization_id)
        elif meta.is_status:
            status = meta.to_status()
            status.channel_id = channel_id
            await self.process_delivery_status(status, organization_id=organization_id)
        # else: unknown webhook type, silently acknowledge

    async def process_delivery_status(
        self, status: DeliveryStatus, *, organization_id: str | None = None
    ) -> None:
        """Process a delivery status through the hook engine.

        Resolves the room from ``status.room_id`` or ``status.channel_id``
        (via the store) and dispatches ON_DELIVERY_STATUS hooks with full
        room context.

        Args:
            status: The DeliveryStatus from meta.to_status().
            organization_id: The organization the caller acts for (RFC
                §17.2): a status that names no room is routed among that
                organization's rooms only, every room when unset, as an
                inbound message is.
        """
        room_id = status.room_id
        if not room_id and status.channel_id:
            room_id = await self._room_of_status(status, status.channel_id, organization_id)

        if not room_id:
            logger.warning(
                "Cannot dispatch ON_DELIVERY_STATUS for message %s: no room_id resolved",
                status.message_id,
            )
            return

        try:
            context = await self._build_context(room_id)
        except (RoomNotFoundError, KeyError, ValueError):
            room = await self._store.get_room(room_id)
            if room is None:
                return
            context = RoomContext(room=room, bindings=[])
        if organization_id is not None and context.room.organization_id != organization_id:
            # A room the status names that is another organization's is not
            # found to the caller (RFC §17.2).
            logger.warning(
                "Cannot dispatch ON_DELIVERY_STATUS for message %s: room %s is not found",
                status.message_id,
                room_id,
            )
            return

        self._hook_engine._start_async_hooks(
            room_id,
            HookTrigger.ON_DELIVERY_STATUS,
            status,
            context,
            skip_event_filter=True,
        )

    async def _room_of_status(
        self, status: DeliveryStatus, channel_id: str, organization_id: str | None
    ) -> str | None:
        """The room a delivery status belongs to, or ``None`` (RFC §10.4).

        The newest ACTIVE or PAUSED room whose binding of the channel names
        the status's recipient, as step 1 finds a sender's, else the one
        ACTIVE room bound to the channel, unless its binding names another
        correspondent. Never one of several: on a number shared by many
        correspondents, the oldest room is another customer's, and the
        status (its recipient's number among it) would be dispatched with
        that room's context. A status is no activity, and resumes no room
        (§5.1). Among the rooms of *organization_id* only, when one is named
        (§17.2).
        """
        active = RoomFilter(statuses=[RoomStatus.ACTIVE], organization_id=organization_id)
        channel = self._channels.get(channel_id)
        # In the channel's form, as step 1 reads a sender (RFC §10.4).
        recipient = (
            channel.normalize_address(status.recipient)
            if channel is not None and status.recipient
            else status.recipient
        )
        if recipient:
            room_id = await self._store.find_room_id_by_binding(
                channel_id,
                recipient,
                None,
                RoomFilter(
                    statuses=[RoomStatus.ACTIVE, RoomStatus.PAUSED],
                    organization_id=organization_id,
                ),
                recipient_key=channel.recipient_key if channel is not None else None,
            )
            if room_id is not None:
                return room_id
        candidates = await self._store.find_room_ids_by_channel(channel_id, active, limit=2)
        if len(candidates) != 1:
            return None
        if not recipient:
            return candidates[0]
        # The one room is the recipient's only if its binding could name them:
        # a binding naming another correspondent, its participant or else its
        # recipient in the channel's form (RFC §5.7), is that person's room. A
        # value that is not a recipient is no one's this status can name.
        binding = await self._store.get_binding(candidates[0], channel_id)
        if binding is None or not binding_admits(binding, frozenset({recipient})):
            return None
        delivers_to = routed_recipient(channel, binding)
        named_by_recipient = not binding.group and binding.participant_id is None
        if named_by_recipient and delivers_to not in (None, recipient):
            return None
        return candidates[0]

    async def add_room_hook(
        self,
        room_id: str,
        trigger: HookTrigger,
        execution: HookExecution,
        fn: SyncHookFn | AsyncHookFn,
        priority: int = 0,
        name: str = "",
        *,
        fail_closed: bool = False,
        needs_lock: bool = True,
        organization_id: str | None = None,
    ) -> HookRegistration:
        """Add a hook for a specific room; the registration held.

        ``fail_closed`` and ``needs_lock`` have the meaning they have on
        :meth:`hook`. A hook on a trigger a turn fires (RFC §10.6.6) is the
        kit's own executor's for the room; while the kit holds a channel by
        its declaration alone it raises ``ValueError``: define it by name on
        the executors, ``executor.room_hook(name, trigger)(fn)``, and enable
        the name for the room with :meth:`enable_room_hook`.

        *name* is unique within the room's hooks (RFC §9.1): one a hook of
        the room holds, a name the room enables on the executors
        (:meth:`enable_room_hook`) included, or a value the framework writes
        in ``blocked_by``, raises ``ValueError``. Left empty, the hook is named after *fn*, made
        unique; the registration returned holds that name.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): the room is read with it first, and another organization's
        room raises :class:`RoomNotFoundError` with no hook registered.
        Unset, no room is read, and a room id the store does not hold takes
        the hook as before. :meth:`HookEngine.add_room_hook` is the
        synchronous engine call, which reads no room.
        """
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        return self._hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=trigger,
                execution=execution,
                fn=fn,
                priority=priority,
                name=name,
                fail_closed=fail_closed,
                needs_lock=needs_lock,
            ),
        )

    async def remove_room_hook(
        self, room_id: str, name: str, *, organization_id: str | None = None
    ) -> bool:
        """Remove a room hook the kit registered, by name; one defined by
        name on the executors is disabled for the room with
        :meth:`disable_room_hook`.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): another organization's room raises :class:`RoomNotFoundError`
        with no hook removed.
        """
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        return self._hook_engine.remove_room_hook(room_id, name)

    async def enable_room_hook(
        self, room_id: str, name: str, *, organization_id: str | None = None
    ) -> None:
        """Run the room hook *name* the executors define on a trigger a turn
        fires (``executor.room_hook(name, trigger)(fn)``) for *room_id*'s
        turns (RFC §10.6.6): the turns' requests name it, and only an
        executor that defines it takes them (RFC §10.6.7).

        The name joins the room's hooks, where a name is unique (RFC §9.1):
        a room hook the kit registered for *room_id* under *name* raises
        ``ValueError``, and so does registering one under a name the room
        enables. Enabling a name the room enables already changes nothing.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): another organization's room raises :class:`RoomNotFoundError`
        with nothing enabled.
        """
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        if name in self._hook_engine.room_hook_names(room_id):
            raise ValueError(
                f"Room {room_id} already holds a hook named {name!r}: a name the room "
                "enables joins its hooks, where a name is unique (RFC §9.1, §10.6.6)"
            )
        if not self.turn_executor.hooks.defines_room_hooks([name]):
            logger.warning(
                "Room hook %r enabled for room %s is not defined on the kit's own "
                "executor: the room's turns of the agents the kit runs find no "
                "executor, and its calls outside a turn (realtime voice, conference) "
                "run without it",
                name,
                room_id,
            )
        self._turns.enable_room_hook(room_id, name)

    async def disable_room_hook(
        self, room_id: str, name: str, *, organization_id: str | None = None
    ) -> bool:
        """Stop running the room hook *name* for *room_id*'s turns; whether
        it ran. *organization_id* is the organization the caller acts for
        (RFC §17.2): another organization's room raises
        :class:`RoomNotFoundError` with nothing disabled."""
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        return self._turns.disable_room_hook(room_id, name)
