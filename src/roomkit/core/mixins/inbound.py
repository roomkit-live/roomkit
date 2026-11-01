"""InboundMixin — inbound message processing entry point and routing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._runtime_record import RUNTIME_RECORD
from roomkit.channels.addressing import read_recipient
from roomkit.core._changes import for_channels
from roomkit.core._closing import Cascades, DoorRuns
from roomkit.core._delivery_keys import DeliveryKeys, NamedDelivery
from roomkit.core._instructions import FINGERPRINT, names_an_agent
from roomkit.core._publication_keys import admitted_key
from roomkit.core._reentrance import (
    REENTRY_LOOP_CAP,
    Entry,
    Placement,
    RoomWork,
    admission_order,
    chain_floor,
)
from roomkit.core.exceptions import (
    ChannelNotRegisteredError,
    RoomAlreadyExistsError,
    RoomNotFoundError,
)
from roomkit.core.inbound_router import (
    binding_names_no_one,
    known_as,
    recordable_sender,
    routed_recipient,
    still_the_messages,
)
from roomkit.core.locks import collecting_release_failures, failed_release
from roomkit.core.mixins.channel_ops import is_channel_detached
from roomkit.core.mixins.helpers import _REFUSING_STATUSES, HelpersMixin
from roomkit.core.mixins.inbound_identity import _IdentityBlockedError
from roomkit.core.mixins.verification import HELD_KEY
from roomkit.core.task_utils import run_to_its_end
from roomkit.models.delivery import (
    STANDALONE,
    DeliveryHandle,
    InboundMessage,
    InboundResult,
)
from roomkit.models.enums import (
    ChannelType,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import refuse_too_deep, with_pipeline_id
from roomkit.models.response_metadata import merge_caller_record

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.admission import RoomAdmission
    from roomkit.core.inbound_router import InboundRoomRouter
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.core.locks import RoomLockManager
    from roomkit.identity.base import IdentityResolver
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.identity import Identity, IdentityResult
    from roomkit.models.room import Room
    from roomkit.store.base import ConversationStore
    from roomkit.telemetry.base import TelemetryProvider

logger = logging.getLogger("roomkit.framework")

# How often a message is routed before it gets a room of its own: once, and
# once more when the room the router chose is no longer the message's under
# its lock (RFC §10.1 step 2).
_ROUTES = 2


@runtime_checkable
class InboundHost(Protocol):
    """Contract: capabilities a host class must provide for InboundMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _channels: Registry of channel-id to :class:`Channel` instances.
        _lock_manager: Per-room lock for serialised mutation.
        _admission: Admission tickets for off-lock checks (RFC §9.5.1).
        _identity_resolver: Optional identity resolver for RFC 7 pipeline.
        _identity_channel_types: Channel types eligible for identity resolution.
        _identity_timeout: Timeout in seconds for identity resolution.
        _process_timeout: Timeout in seconds for locked inbound processing.
        _inbound_router: Router that maps inbound messages to room IDs.
        _max_chain_depth: Maximum AI-to-AI chain depth (RFC 10).
        _inbound_rate_limiter: Token-bucket rate limiter (or ``None``).
        _inbound_rate_limit: Rate-limit configuration (or ``None``).
        _telemetry: Telemetry / tracing provider.

    Cross-mixin methods (provided by other mixins in the MRO):
        _resolve_identity: From :class:`InboundIdentityMixin`.
        _process_locked: From :class:`InboundLockedMixin`.
        _finish_cascade: From :class:`LaneExecutionMixin`.
        create_room: From :class:`RoomLifecycleMixin`.
        get_room: From :class:`RoomLifecycleMixin` (the scoped read, RFC §17.2).
        attach_channel: From :class:`ChannelOpsMixin`.
        _notify_binding_updated: From :class:`ChannelOpsMixin`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _lock_manager: RoomLockManager
    _admission: RoomAdmission
    _identity_resolver: IdentityResolver | None
    _identity_channel_types: set[ChannelType] | None
    _identity_timeout: float
    _process_timeout: float
    _inbound_router: InboundRoomRouter
    _max_chain_depth: int
    _inbound_rate_limiter: Any  # TokenBucketRateLimiter | None
    _inbound_rate_limit: Any  # RateLimit | None
    _telemetry: TelemetryProvider


class InboundMixin(HelpersMixin):
    """Inbound message processing pipeline — entry point and routing.

    Host contract: :class:`InboundHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _lock_manager: RoomLockManager
    _admission: RoomAdmission
    _identity_resolver: IdentityResolver | None
    _identity_channel_types: set[ChannelType] | None
    _identity_timeout: float
    _process_timeout: float
    _inbound_router: InboundRoomRouter
    _max_chain_depth: int
    _inbound_rate_limiter: Any  # TokenBucketRateLimiter | None
    _inbound_rate_limit: Any  # RateLimit | None
    _telemetry: TelemetryProvider
    _verification_inbound: bool
    _door_runs: DoorRuns
    _cascades: Cascades

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _resolve_identity: Any  # see InboundHost
    _apply_verification: Any  # see VerificationMixin
    _name_held: Any  # see VerificationMixin
    _record_held_cancellation: Any  # see VerificationMixin
    _process_locked: Any  # see InboundHost
    _queue_of: Any  # see InjectedEventsMixin
    _hand_off_decided: Any  # see InjectedEventsMixin
    _finish_queued: Any  # see InjectedEventsMixin
    _room_work: RoomWork
    _consume_streams_when_cascade_completes: Any  # see LaneExecutionMixin
    _finish_cascade: Any  # see LaneExecutionMixin
    create_room: Any  # see InboundHost
    get_room: Any  # see InboundHost
    attach_channel: Any  # see InboundHost
    _notify_binding_updated: Any  # see InboundHost

    async def process_inbound(
        self,
        message: InboundMessage,
        *,
        room_id: str | None = None,
        defer_delivery: bool = False,
        organization_id: str | None = None,
    ) -> InboundResult:
        """Process an inbound message through the full pipeline.

        Args:
            message: The inbound message to process.
            room_id: Explicit room to route to, bypassing the inbound router.
                Useful for shared channels attached to multiple rooms.
            defer_delivery: Return at the commit instead of waiting for the
                delivery set (RFC §10.1 step 18 detached completion). The
                result carries the committed event — a hook refusal is still
                decided under the room lock and reported synchronously — and
                ``result.delivery`` holds a :class:`DeliveryHandle` on the
                rest of the turn: delivery execution, reentry passes (an AI
                reply included) and streamed responses, all following in the
                room's lane. For a caller that must answer with the committed
                event while the agent's turn runs on (an HTTP route returning
                200, say); ``delivery_results`` is backfilled by
                ``delivery.wait()``.
            organization_id: The organization the caller acts for (RFC
                §17.2). The router is given it, with the sender's identity
                resolved within it, and the default router searches that
                organization's rooms only (every room when it is unset). The
                room the message would land in is checked against it before
                any event is committed or any channel auto-attached: a room
                belonging to another organization is reported as **not
                found**, whether the caller named it or a custom router
                answered it; a room the caller names, or a router answers,
                that does not exist yet is created under it. That auto-create
                makes a miss observable, so a scoped caller passes room ids
                it resolved itself, never one it was handed. Left unset, the
                read is unscoped and behaves as it always has. A library has
                no caller or auth context of its own, so the scope has to
                come from the caller: the row that mapped an external
                identity (a chat, a number) to a room is where it usually
                lives.

        Cancelling an awaited call stops only its turns (the message is
        still delivered to the room's other channels) and drains owned
        generation, tools and streams before propagating ``CancelledError``.
        Cleanup has a five-second budget; ``TimeoutError`` reports that work
        is still unwinding and its resources must be kept. Deferred callers
        explicitly use ``result.delivery.cancel()``, and any code by the
        message's ``delivery_key`` (:meth:`cancel_delivery`); cancelling a
        handle's ``wait()`` cancels only that wait. Committed timeline events
        are retained in every case.

        Called from the room's own work, the message is part of it (RFC
        §9.5.1): sent into its room under the room lock, from an off-lock
        hook's body or from a channel's parse, it is adopted by the event
        being processed and takes no ticket; from a turn or a delivery it is
        a new arrival. It draws on the reentry budget of that event, or of the
        cascade that delivers, as it starts: past it, it is refused
        ``reentry_loop_cap`` with no event, before anything else when it names
        its room (its channel not looked up, nothing attached), once routing
        knows its room when it is routed. Named with an *organization_id*,
        the room is read with it before the budget: another organization's
        room is not found, and nothing is drawn (RFC §17.2). A message sent
        within a turn is at least as deep as the turn's response (RFC §8.3).

        Raises:
            ChannelNotRegisteredError: the message's channel is not registered
                (a message refused past its room's work budget first excepted).
            ReentrantCallError: the message is routed (no *room_id*) into the
                room whose lock the caller holds or in whose admission order it
                stands (an off-lock hook's body, a parse); nothing is written.
            ValueError: its content, or the event its channel parses, holds
                more than five composite levels (RFC §5.3); nothing is stored.
                Its channel is a host channel, which receives no external
                payload (RFC §10.5); nothing is read or written.
        """
        # A sixth composite level is refused where it is built (RFC §5.3):
        # the message fails before anything is read or written.
        refuse_too_deep(message.content)
        # The message's key names its delivery from the call's start, before
        # anything is awaited (RFC §10.1): the read of its room's scope too.
        with self._delivery_keys.naming(message.delivery_key, organization_id) as named:
            return await self._process_named_inbound(
                message, room_id, defer_delivery, organization_id, named
            )

    async def _process_named_inbound(
        self,
        message: InboundMessage,
        room_id: str | None,
        defer_delivery: bool,
        organization_id: str | None,
        named: NamedDelivery,
    ) -> InboundResult:
        """:meth:`process_inbound` for *message*, whose delivery *named*
        names, from RFC §10.1 step 1."""
        from roomkit.telemetry.base import SpanKind
        from roomkit.telemetry.context import get_current_span, reset_span, set_current_span

        telemetry = self._telemetry

        def open_span() -> str:
            """The call's ``framework.inbound`` span, opened."""
            return telemetry.start_span(
                SpanKind.INBOUND_PIPELINE,
                "framework.inbound",
                parent_id=get_current_span(),
                channel_id=message.channel_id,
                room_id=room_id,
                attributes={"sender_id": message.sender_id or ""},
            )

        # One deadline from the call's start to its decision (RFC §13.6), as
        # an absolute instant shared by every wait before the commit rather
        # than re-applied at each: the scope's read of a room the call names
        # from its work, routing, with the identity it reads, the router and
        # step 2's waits for the room lock, then the context build, the
        # parse, the off-lock check and the room lock. A router or a lock
        # that hangs holds the caller no longer than the bound. A room step 2
        # creates is completed whatever the deadline: it is the room's state,
        # not the event's.
        deadline = asyncio.get_running_loop().time() + self._process_timeout
        # RFC §10.1 step 1, §9.5.1 Past the cap: a message that names its
        # room, sent by that room's work, draws on its budget as it starts,
        # before anything else but its scope's check; a routed one, and one
        # naming a room of no work, are charged at step 2 once their room is
        # known. The refusal writes nothing, and is traced as any call: a
        # loop the cap cut stays visible.
        charged = self._named_room_work(room_id)
        if charged is not None and room_id is not None:
            refused = await self._charge_named_room(
                charged, room_id, message, organization_id, deadline
            )
            if refused is not None:
                attributes = {"blocked": True, "deferred": defer_delivery}
                telemetry.end_span(open_span(), attributes=attributes)
                return refused

        # Inbound rate limiting — drop excess messages before any processing
        if (
            self._inbound_rate_limiter
            and self._inbound_rate_limit
            and not self._inbound_rate_limiter.acquire(
                message.channel_id, self._inbound_rate_limit
            )
        ):
            return InboundResult(blocked=True, reason="rate_limited")

        channel = self._channels.get(message.channel_id)
        if channel is None:
            raise ChannelNotRegisteredError(f"Channel {message.channel_id} not registered")
        if channel.channel_type is ChannelType.HOST:
            # RFC §10.5: the host speaks through post() or send_event(); its
            # channel receives no external payload, so nothing is routed,
            # opened or attached for one.
            raise ValueError(
                f"Channel {message.channel_id!r} is a host channel and receives no external "
                "payload: the host speaks through RoomKit.post() or send_event()"
            )
        # One form for the sender before anything reads it (RFC §10.4): the
        # router, the bindings and the participants compare and store it.
        # The provider's own spelling survives in raw_payload.
        sender = channel.normalize_address(message.sender_id)
        if sender != message.sender_id:
            message = message.model_copy(update={"sender_id": sender})

        inbound_span_id = open_span()
        # Propagate backend-specific context for robust parent linking
        token = set_current_span(
            inbound_span_id, telemetry_ctx=telemetry.get_span_context(inbound_span_id)
        )
        _inbound_result: InboundResult | None = None
        try:
            _inbound_result = await self._process_inbound_inner(
                message,
                channel,
                room_id,
                telemetry,
                inbound_span_id,
                defer_delivery,
                deadline,
                named,
                organization_id=organization_id,
                charged=charged,
            )
            return _inbound_result
        except Exception as exc:
            telemetry.end_span(
                inbound_span_id,
                status="error",
                error_message=str(exc),
                attributes={"deferred": defer_delivery},
            )
            raise
        finally:
            reset_span(token)
            if _inbound_result is not None:
                # ``deferred``: the turn's tail lives in a ``framework.detached``
                # child span; this span measures what the caller waited for.
                telemetry.end_span(
                    inbound_span_id,
                    attributes={"blocked": _inbound_result.blocked, "deferred": defer_delivery},
                )

    async def cancel_delivery(
        self,
        key: str,
        *,
        reason: str = "caller_cancelled",
        organization_id: str | None = None,
    ) -> int:
        """Cancel the deliveries the key names of the messages sent with
        ``InboundMessage(delivery_key=key)`` (RFC §10.1, *Explicit delivery
        cancellation*): how many it cancelled, 0 when the key names nothing
        (the deliveries are over, or were never sent).

        *organization_id* is the organization the caller acts for (RFC
        §17.2): only the deliveries of its rooms are cancelled, so two
        tenants that chose the same key never cancel each other's. Left
        unset, every delivery the key names in the process is. A delivery
        whose ``process_inbound`` named no organization belongs to none until
        the call reads its room, right after routing (step 3): a scoped
        cancellation reaching it earlier is applied once the room proves the
        organization's, and is not counted. Pass ``organization_id`` to
        ``process_inbound`` too, and the count is exact from the call's start.

        A key names its delivery from the start of ``process_inbound``.
        Each delivery's agents stop: a turn not started yet never starts, a
        running one is cut, and nothing answers it after. The message itself
        is still delivered to the room's other channels and its
        AFTER_BROADCAST hooks fire; one cancelled before its commit is
        committed all the same. A message held for verification stays named
        until its release is answered: cancelled while it waits, it is
        released with no answer, in any process. Returns at once, from
        anywhere in the process, a tool handler included; each delivery's
        caller reads its cancelled end on its result, as after
        ``result.delivery.cancel()``; a call refused before its commit
        reports its refusal instead.

        The key names, it does not authorize: check who may cancel before
        calling.
        """
        return self._delivery_keys.cancel(key, reason, organization_id=organization_id)

    @cached_property
    def _delivery_keys(self) -> DeliveryKeys:
        """The deliveries a caller named by a key (RFC §10.1)."""
        return DeliveryKeys(self._record_held_cancellation)

    def _named_room_work(self, room_id: str | None) -> Entry | None:
        """What a call naming *room_id* is to that room's work when it draws
        on the work's reentry budget, charged as it starts (RFC §10.1 step 1,
        §9.5.1); ``None`` for a call routed or into a room of no work,
        charged at step 2 if at all."""
        if room_id is None:
            return None
        entry = self._room_work.entry(room_id)
        return entry if entry.charged else None

    async def _charge_named_room(
        self,
        charged: Entry,
        room_id: str,
        message: InboundMessage,
        organization_id: str | None,
        deadline: float,
    ) -> InboundResult | None:
        """Draw the pass a message naming *room_id*, sent by that room's
        work, takes as it starts (RFC §10.1 step 1, §9.5.1): its refusal
        past the budget, ``None`` once drawn. A caller acting for an
        organization has the room read with it first, within the call's
        *deadline* (§13.6): another organization's room is not found before
        anything is drawn from the budget (§17.2). A room the store does not
        hold is left to step 2, which creates it under the organization.

        Raises:
            RoomNotFoundError: the room is another organization's.
        """
        if organization_id is not None:
            try:
                async with asyncio.timeout_at(deadline):
                    room = await self._store.get_room(room_id)
            except TimeoutError:
                return await self._refuse_on_timeout(room_id, channel_id=message.channel_id)
            if room is not None and room.organization_id != organization_id:
                raise RoomNotFoundError(f"Room {room_id} not found")
        if not charged.draw():
            return _past_the_cap(room_id, message)
        return None

    async def _process_inbound_inner(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str | None,
        telemetry: Any,
        inbound_span_id: str,
        defer_delivery: bool,
        deadline: float,
        named: NamedDelivery,
        organization_id: str | None = None,
        charged: Entry | None = None,
    ) -> InboundResult:
        """Inner inbound processing (extracted for telemetry wrapping), its
        delivery *named* by the message's key from the call's start (RFC
        §10.1), *deadline* the call's (§13.6). A failed release of the room
        lock is the result's error when no earlier error was met (§13.5)."""
        with collecting_release_failures() as failed_releases:
            result = await self._route_and_deliver(
                message,
                channel,
                room_id,
                telemetry,
                inbound_span_id,
                defer_delivery,
                named,
                deadline=deadline,
                organization_id=organization_id,
                charged=charged,
            )
        if result.error is None and failed_releases:
            # In place: a detached delivery's handle holds this result.
            result.error = failed_releases[0]
        return result

    async def _route_and_deliver(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str | None,
        telemetry: Any,
        inbound_span_id: str,
        defer_delivery: bool,
        named: NamedDelivery,
        *,
        deadline: float,
        organization_id: str | None,
        charged: Entry | None,
    ) -> InboundResult:
        """RFC §10.1 steps 2-18 for *message*, whose delivery *named* names,
        by the call's *deadline* (§13.6); *charged* the entry of a call
        naming its room, drawn as it started."""
        refused = self._refused_instruction(message)
        if refused is not None:
            return refused

        # A message the room's delivery work sends that creates a room,
        # routed or named, is that work's: charged once on its budget before
        # the room is created (RFC §9.5.1, gap G-070), unless it was charged
        # as it started.
        opener = self._room_work.opening() if charged is None else None
        opening = _Opening(opener) if opener is not None else None
        try:
            room_id, room_just_created = await self._route_to_room(
                message,
                channel,
                room_id,
                telemetry,
                inbound_span_id,
                organization_id=organization_id,
                deadline=deadline,
                opening=opening,
            )
        except _RoutingTimeoutError as exc:
            # Null while the message was routed (RFC §8.2), else the room the
            # call named or routing gave it.
            return await self._refuse_on_timeout(exc.room_id, channel_id=message.channel_id)
        except _PastTheCapError:
            return _past_the_cap(None, message)

        entry = self._entry_after_routing(room_id, room_just_created, opening, charged)
        if entry is None:
            return _past_the_cap(room_id, message)
        try:
            prepared = await self._prepare_event(
                message, channel, room_id, room_just_created, deadline, telemetry, named, entry
            )
        except TimeoutError:
            return await self._refuse_on_timeout(room_id, channel_id=message.channel_id)
        if isinstance(prepared, InboundResult):
            return prepared
        event, context, resolved_identity, pending_id_result = prepared
        return await self._commit_and_await_delivery(
            event,
            context,
            resolved_identity,
            pending_id_result,
            message,
            channel,
            room_id,
            deadline,
            named,
            entry,
            defer_delivery=defer_delivery,
        )

    def _entry_after_routing(
        self,
        room_id: str,
        room_just_created: bool,
        opening: _Opening | None,
        charged: Entry | None,
    ) -> Entry | None:
        """What the call is to its room's work once step 2 knows the room
        (RFC §9.5.1), charged one pass if it was not yet; ``None`` past the
        budget. A call naming its room was charged as it started, one that
        opened a room as the room was created; its cascade draws on that
        budget. A message is charged once (§8.3): one charged for a room
        another call created first, then routed into the room its work
        delivers in, is not charged there again."""
        if charged is not None:
            return charged
        if room_just_created and opening is not None:
            return Entry(Placement.ARRIVAL, opening.entry.budget, False)
        entry = self._room_work.entry(room_id)
        drawn = opening is not None and opening.charged
        if entry.charged and not drawn and not entry.draw():
            return None
        return entry

    def _refused_instruction(self, message: InboundMessage) -> InboundResult | None:
        """The refusal of an instruction that could only land as something
        else, before anything is read or written (RFC §10.1.1 steps 1 and 2):
        naming no agent the kit registers, it would ask nobody or make every
        agent in the room speak; with an idempotency key, it asks for a check
        that runs on the stored event it will never be. None for any other
        message."""
        if message.event_type != EventType.INSTRUCTION:
            return None
        if not names_an_agent(message.addressed_to, self._channels):
            return InboundResult(blocked=True, reason="instruction_unaddressed")
        if message.idempotency_key:
            return InboundResult(blocked=True, reason="instruction_not_idempotent")
        return None

    async def _prepare_event(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        room_just_created: bool,
        deadline: float,
        telemetry: Any,
        named: NamedDelivery,
        entry: Entry,
    ) -> tuple[Any, RoomContext, Identity | None, IdentityResult | None] | InboundResult:
        """RFC §10.1 steps 3-5, under the caller's pre-commit budget; *entry*
        is what the message is to the room's work (RFC §9.5.1).

        Everything here was previously unbounded: a store read that never
        returns during the context build, or a channel's own
        ``handle_inbound`` reaching a provider without a timeout of its own,
        held the caller forever while ``process_timeout`` sat configured and
        inert.

        Raises:
            TimeoutError: The budget ran out. Nothing of the event has been
                written at this point — a room routing created stays, as the
                room's state, which its creation completes whatever the
                deadline (RFC §13.6).
        """
        async with asyncio.timeout_at(deadline):
            return await self._prepare_event_inner(
                message, channel, room_id, room_just_created, telemetry, named, entry
            )

    async def _prepare_event_inner(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        room_just_created: bool,
        telemetry: Any,
        named: NamedDelivery,
        entry: Entry,
    ) -> tuple[Any, RoomContext, Identity | None, IdentityResult | None] | InboundResult:
        context = await self._build_context(room_id)
        # The room is read: its organization is the delivery's (RFC §10.1).
        named.locate(context.room.organization_id if context.room else None)

        # Fire ON_SESSION_STARTED for text channels when a new room is created
        if room_just_created and channel.channel_type not in (
            ChannelType.VOICE,
            ChannelType.REALTIME_VOICE,
        ):
            await self._fire_text_session_started(
                room_id,
                message.channel_id,
                channel.channel_type,
                message.sender_id or "",
            )

        # Let channel process inbound. A channel with no binding in the room
        # (one the integrator detached, RFC §7.5 rule 7) parses its message
        # seeing no event; its commit gate stores it ``source_unbound``.
        parse_context = (
            # A deleted event's content is withheld from it (RFC §10.3).
            for_channels(context)
            if context.get_binding(message.channel_id) is not None
            else context.model_copy(update={"recent_events": []})
        )
        # The parse stands in its message's admission order (RFC §9.5.1): a
        # call it makes into the room is adopted beside, on its budget.
        with admission_order(room_id, entry.budget):
            event = await channel.handle_inbound(message, parse_context)
        # The event its channel built is refused as the message is (RFC §5.3).
        refuse_too_deep(event.content)

        event = _at_least_as_deep(_apply_message_fields(event, message))
        event = with_pipeline_id(event)

        # Identity resolution pipeline (RFC §11)
        try:
            event, resolved_identity, pending_id_result = await self._resolve_identity(
                event, message, channel, room_id, context, telemetry
            )
        except _IdentityBlockedError as exc:
            return InboundResult(blocked=True, reason=exc.reason)

        # Identity verification (RFC §10.1 step 5b, §11.7): label, hold, codes.
        if self._verification_inbound:
            verified = await self._apply_verification(event, room_id, context, resolved_identity)
            if isinstance(verified, InboundResult):
                return verified
            event = verified

        return event, context, resolved_identity, pending_id_result

    async def _commit_and_await_delivery(
        self,
        event: Any,
        context: RoomContext,
        resolved_identity: Identity | None,
        pending_id_result: IdentityResult | None,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        deadline: float,
        named: NamedDelivery,
        entry: Entry,
        *,
        defer_delivery: bool = False,
    ) -> InboundResult:
        """RFC §10.1 steps 6-18: the locked region, then the delivery set.

        The room lock is acquired under the same budget as the phase before it
        (§13.6): a room whose lock is held by a stuck event would otherwise
        queue every later message with nothing to stop it, which is the pile-up
        the setting exists for. Past the lock, ``_process_locked`` spends what
        remains on the pre-commit gates and stops there. That timeout never
        invalidates a committed event. Explicit caller cancellation instead
        stops the message's turns, retaining the committed timeline record.
        The cascade becomes the delivery *named* since the call's start,
        cancelled before its commit when a cancellation reached it earlier;
        an event held for verification keeps it named until its release.
        """

        cascade = self._cascades.new(room_id, reentry_budget=entry.budget)
        named.attach(cascade)
        try:
            result = await self._complete_inbound_delivery(
                event,
                context,
                resolved_identity,
                pending_id_result,
                message,
                channel,
                room_id,
                deadline,
                cascade,
                named,
                entry,
                defer_delivery=defer_delivery,
            )
        except BaseException:
            # The caller owns this cascade even if setup failed after commit,
            # before it could receive a deferred handle. Drain off the lock.
            cascade.over()
            await cascade.abandon("caller_cancelled")
            raise
        cascade.call_returned()
        return result

    async def _complete_inbound_delivery(
        self,
        event: Any,
        context: RoomContext,
        resolved_identity: Identity | None,
        pending_id_result: IdentityResult | None,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        deadline: float,
        cascade: DeliveryCascade,
        named: NamedDelivery,
        entry: Entry,
        *,
        defer_delivery: bool,
    ) -> InboundResult:
        # A message nested in a pass's section (code under the lock) queues
        # what its hooks inject on that section's queue (RFC §9.4).
        queued = self._queue_of(entry)
        handed = None
        async with AsyncExitStack() as stack:
            try:
                async with asyncio.timeout_at(deadline):
                    # Off-lock check (RFC §9.5.1): runs before the lock under
                    # the same pre-commit budget, and holds the room's
                    # admission ticket until the stack unwinds — after the
                    # lock is released, so after the commit. A message the
                    # room's work adopted takes none.
                    precheck = await stack.enter_async_context(
                        self._admission.admitted(
                            room_id, event, context, budget=entry.budget, adopted=entry.adopted
                        )
                    )
                    await stack.enter_async_context(self._lock_manager.locked(room_id))
            except TimeoutError:
                return await self._refuse_on_timeout(room_id, channel_id=message.channel_id)
            try:
                result: InboundResult = await self._process_locked(
                    event,
                    room_id,
                    context,
                    cascade,
                    queued=queued,
                    resolved_identity=resolved_identity,
                    pending_id_result=pending_id_result,
                    deadline=deadline,
                    precheck=precheck,
                )
            finally:
                # Decided, or its caller gone: what its hooks injected, and
                # what the hooks of a call nested in its section injected, no
                # longer depends on this caller (RFC §9.4, §13.6), and commits
                # once the lock is released.
                handed = self._hand_off_decided(entry, queued, room_id, cascade)
            self._name_held(named, result)
        # Past its commit: the kit's close no longer cuts the call, but its
        # delivery (RFC §12.10.4).
        self._door_runs.committed()
        if result.error is None:
            # The lock's release comes before anything the delivery set
            # meets, so its failure is the earlier error (RFC §13.5).
            result.error = failed_release()
        # Each in a section of its own, once the event's section released the
        # lock (RFC §9.4, §10.1 steps 10 to 14): another event may commit
        # between them. Committed before this call returns, unless the call
        # is nested under a lock its caller holds.
        await self._finish_queued(handed)

        if defer_delivery:
            # Every result that reached this locked region gets a handle — a
            # hook refusal included, whose near-empty cascade resolves at
            # once. A refusal shed before it (rate limited, pre-commit
            # timeout, identity block) returned above with delivery=None:
            # there is no delivery to follow.
            return await self._defer_inbound_delivery(result, cascade, message, channel, room_id)

        await self._await_inbound_delivery(result, cascade, room_id, adopted=entry.adopted)
        await self._connect_session_if_ready(message, channel, room_id, result)
        return result

    async def _await_inbound_delivery(
        self, result: InboundResult, cascade: DeliveryCascade, room_id: str, *, adopted: bool
    ) -> None:
        """RFC §10.1 step 18 on the waiting path: the caller observes its
        event's delivery-set completion, reported on *result*.

        The cascade — the trigger's delivery set plus every reentry pass it
        transitively spawned — then the streams it started, read outside the
        lane (TTS delivery can take seconds). AFTER_BROADCAST, mutation and
        ON_ERROR hooks fire from the lane executor, off this room's lock. A
        stream failure is surfaced on the result so a headless caller can
        react (interactive callers ignore it — the ON_ERROR hooks already
        fired an error card).

        A call the room's work *adopted* answers at its commit instead, with
        no delivery results: its delivery waits in the lane behind the work
        that made it (RFC §9.5.1).
        """
        if adopted:
            await self._finish_cascade(cascade, room_id, detached=True)
            return
        stream_error, record = await self._finish_cascade(cascade, room_id, caller_logs=True)
        # Step 18 reports the delivery set the caller waited for, read after
        # the streams: what the other agents answered to a streamed segment
        # re-entered while it was read.
        result.report_cascade(cascade)
        if stream_error is not None and result.error is None:
            result.error = stream_error
        merge_caller_record(result.response_metadata, record)

    async def _defer_inbound_delivery(
        self,
        result: InboundResult,
        cascade: DeliveryCascade,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
    ) -> InboundResult:
        """Hand the caller the committed event now, with a handle on the rest
        of its turn (RFC §10.1 step 18, deferred completion).

        Blocked was decided under the lock, so a refusal is still synchronous;
        the delivery set, reentry passes and streamed responses follow in the
        room's lane, its streams consumed on the same background task a
        detached caller uses. The handle is the caller's grip on that tail: it
        backfills delivery results, errors and response metadata on
        completion.
        """
        consumer = self._consume_streams_when_cascade_completes(cascade, room_id)
        result.delivery = DeliveryHandle(cascade, consumer, result)
        await self._connect_session_if_ready(message, channel, room_id, result)
        return result

    async def _connect_session_if_ready(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        result: InboundResult,
    ) -> None:
        """Bind the session for stateful channels (voice, persistent WS, etc.).

        Runs AFTER hooks passed and the event was stored — a blocked event
        never reaches ``connect_session``. On the deferred path this still
        happens before ``process_inbound`` returns: the session-connected
        invariant does not depend on the caller waiting for delivery.
        """
        if message.session is not None and not result.blocked:
            binding = await self._store.get_binding(room_id, message.channel_id)
            if binding is not None:
                await channel.connect_session(message.session, room_id, binding)

    async def _route_to_room(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str | None,
        telemetry: Any,
        inbound_span_id: str,
        *,
        organization_id: str | None,
        deadline: float,
        opening: _Opening | None = None,
    ) -> tuple[str, bool]:
        """Route inbound message to a room, auto-creating if needed.

        ``organization_id`` scopes the room the message lands in (RFC §17.2):
        a room of another organization is not found, a room created here is
        created under it. *deadline* is the call's (RFC §13.6): past it,
        :class:`_RoutingTimeoutError`. *opening* is the budget a message of
        the room's delivery work draws on when it creates a room, routed or
        named, charged once before the room is created: spent,
        :class:`_PastTheCapError` with nothing written.

        Returns:
            A tuple of (room_id, room_just_created).
        """
        from roomkit.telemetry.base import SpanKind
        from roomkit.telemetry.context import get_current_span

        route_span = telemetry.start_span(
            SpanKind.INBOUND_PIPELINE,
            "framework.route",
            parent_id=get_current_span(),
            channel_id=message.channel_id,
            attributes={"sender_id": message.sender_id or ""},
        )
        try:
            if room_id is None:
                room_id, room_just_created = await self._route_and_claim(
                    message, channel, organization_id, deadline, opening
                )
            else:
                room_id, room_just_created = await self._find_or_open_room(
                    message, channel, room_id, organization_id, deadline=deadline, opening=opening
                )
            telemetry.end_span(route_span, attributes={"room_id": room_id or ""})
        except Exception as exc:
            telemetry.end_span(route_span, status="error", error_message=str(exc))
            raise

        # Backfill room_id on the INBOUND_PIPELINE span now that routing is done
        telemetry.set_attribute(inbound_span_id, "room_id", room_id)
        # Extract session_id from voice metadata if present
        voice_session_id = (message.metadata or {}).get("voice_session_id")
        if voice_session_id:
            telemetry.set_attribute(inbound_span_id, "session_id", voice_session_id)

        return room_id, room_just_created

    async def _find_or_open_room(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str | None,
        organization_id: str | None,
        identity: Identity | None = None,
        *,
        deadline: float,
        opening: _Opening | None = None,
    ) -> tuple[str, bool]:
        """The room *message* goes to, opened or attached as needed (RFC §10.1 step 2).

        The caller's *room_id* when it named one, the router's answer
        otherwise: the router is given the caller's organization and the
        sender's *identity* in it (§10.4), and its answer is held to the
        rules of an id the caller named. The router's answer is bounded by
        the call's *deadline* (RFC §13.6). Returns the room id and whether
        the room was just created.
        """
        routed = room_id is None
        if room_id is None:
            room_id = await _by(
                deadline,
                self._inbound_router.route(
                    channel_id=message.channel_id,
                    channel_type=channel.channel_type,
                    participant_id=_conversation(message, channel),
                    channel_data=message.metadata,
                    organization_id=organization_id,
                    identity=identity,
                ),
                room_id=None,
            )
        if room_id is None:
            _charge_opening(opening)
            return await self._open_room_for(message, None, organization_id), True
        created = await self._ensure_room_and_binding(
            message,
            room_id,
            organization_id,
            attach=not routed,
            deadline=deadline,
            opening=opening,
        )
        return room_id, created

    async def _ensure_room_and_binding(
        self,
        message: InboundMessage,
        room_id: str,
        organization_id: str | None,
        *,
        attach: bool = True,
        deadline: float,
        opening: _Opening | None = None,
    ) -> bool:
        """Create the named room, or attach the channel to it, when either is missing.

        A routed room is not attached here (*attach* false): it is attached
        under the lock that checks it is still the message's (``_claim_room``),
        so a room that is not gets no binding. The reads of a room that exists
        and the wait for its lock are bounded by the call's *deadline* (RFC
        §13.6); a room created here is completed whatever the deadline, once
        *opening*, the budget a message of the room's delivery work creates a
        room on, is charged (:meth:`_Opening.charge`). Returns whether the
        room was just created.
        """
        # Ensure room exists; auto-create if needed (e.g. voice session
        # with a room_id from SIP headers that hasn't been created yet).
        room_present, binding_present = await _by(
            deadline,
            self._room_and_binding_present(message, room_id, organization_id, attach=attach),
            room_id=room_id,
        )
        if not room_present:
            # A room the store lacks, named by the call or by its router, is
            # charged as one routing answered none (gap G-070).
            _charge_opening(opening)
            try:
                await self._open_room_for(message, room_id, organization_id)
                return True
            except RoomAlreadyExistsError:
                # Another call created the room meanwhile, and the store
                # refused to replace it (RFC §10.1 step 2, §14.3): it is read
                # again as one that exists, for the caller's organization.
                binding_present = await _by(
                    deadline,
                    self._created_meanwhile(message, room_id, organization_id, attach=attach),
                    room_id=room_id,
                )
        if not binding_present:
            # Room exists but the channel is not attached — unless the
            # integrator detached it (RFC §7.5-7). The unlocked read
            # keeps the common case (already bound) free; the actual
            # decision runs under the room lock in _maybe_auto_attach.
            await self._maybe_auto_attach(room_id, message.channel_id, deadline)
        return False

    async def _created_meanwhile(
        self, message: InboundMessage, room_id: str, organization_id: str | None, *, attach: bool
    ) -> bool:
        """Whether the room another call created meanwhile binds the message's
        channel (``True`` when nothing is to be attached), read as a room
        that exists, for the caller's organization (RFC §10.1 step 2)."""
        room = await self._store.get_room(room_id)
        if room is None or (
            organization_id is not None and room.organization_id != organization_id
        ):
            raise RoomNotFoundError(f"Room {room_id} not found") from None
        return not attach or await self._store.binding_exists(room_id, message.channel_id)

    async def _room_and_binding_present(
        self,
        message: InboundMessage,
        room_id: str,
        organization_id: str | None,
        *,
        attach: bool,
    ) -> tuple[bool, bool]:
        """Whether the named room exists for the caller, and whether it binds
        the message's channel (``True`` when nothing is to be attached).

        Existence checks only — neither the room nor the binding is used
        beyond the yes/no, so materialising either would decode its JSONB
        columns and validate a whole model per message for nothing. Asked
        together on one connection; what they decide (create, attach, lock)
        happens outside it, because a pooled connection must never be held
        across a lock or a write path.
        """
        async with self._store.connection():
            if organization_id is None:
                room_present = await self._store.room_exists(room_id)
            else:
                # The caller acts for one organization, and a room of
                # another is not found to it (RFC §17.2): refused
                # here, before any auto-attach or commit. One read
                # answers both existence and scope; the decode the
                # unscoped branch avoids is the price of the check.
                room = await self._store.get_room(room_id)
                if room is not None and room.organization_id != organization_id:
                    raise RoomNotFoundError(f"Room {room_id} not found")
                room_present = room is not None
            binding_present = (
                not attach
                or room_present
                and await self._store.binding_exists(room_id, message.channel_id)
            )
        return room_present, binding_present

    async def _open_room_for(
        self, message: InboundMessage, room_id: str | None, organization_id: str | None
    ) -> str:
        """Create the room an inbound message opens, its binding naming the sender.

        *room_id* is the caller's when it named a room that does not exist yet.
        The binding names the sender (RFC §10.1 step 2), so the router finds
        them there on their next message and admits no one else through it
        (§10.4): on a shared number, the second correspondent gets a room of
        their own. Completed whatever becomes of the caller, as whatever the
        deadline (RFC §13.6): its creation, its ON_ROOM_CREATED hooks and its
        attach.
        """
        return await run_to_its_end(
            self._open_room(message, room_id, organization_id),
            held=self._decided_writes,
            name=f"roomkit-decided-write-{room_id or 'new-room'}",
            # Another call created the room first: step 2 reads it again.
            expected=(RoomAlreadyExistsError,),
        )

    async def _open_room(
        self, message: InboundMessage, room_id: str | None, organization_id: str | None
    ) -> str:
        """:meth:`_open_room_for`'s creation, hooks and attach."""
        room = await self.create_room(room_id=room_id, organization_id=organization_id)
        channel = self._channels.get(message.channel_id)
        sender = _conversation(message, channel) if channel else message.sender_id
        recorded = sender if recordable_sender(sender) else None
        # Once its ON_ROOM_CREATED hooks have run, under the room lock: a
        # binding of the channel one of them made is the room's (RFC §10.1
        # step 2, §7.5 rule 7), and a room one of them closed gets none, as
        # a room in hand whose status refuses events: step 6 refuses the
        # message there.
        async with self._lock_manager.locked(room.id):
            created = await self._store.get_room(room.id)
            if created is not None:
                created = await self._room_timers.apply_due(created)
            if created is not None and created.status not in _REFUSING_STATUSES:
                await self._attach_unbound(
                    created,
                    message.channel_id,
                    participant_id=recorded,
                    # Where the room replies: the sender's address, or the
                    # chat the message came from.
                    metadata=channel.reply_metadata(message) if recorded and channel else {},
                )
        return room.id

    async def _route_and_claim(
        self,
        message: InboundMessage,
        channel: Channel,
        organization_id: str | None,
        deadline: float,
        opening: _Opening | None = None,
    ) -> tuple[str, bool]:
        """The room the router gives *message*, its binding naming the sender.

        The router reads the store off the lock, so by the time the room is
        locked to record the sender it may no longer be the message's (RFC
        §10.1 step 2): closed, paused, its binding recorded for a concurrent
        sender, or someone else heard or reached on the channel since. The
        message is then routed once more, its identity not read again, and a
        second answer that is not the message's, like no answer, gives it a
        new room: a router is asked twice at most, and a routed message is
        never refused ``room_closed`` here. The identity read is bounded by
        the call's *deadline* (RFC §13.6), as the router is.
        """
        sender = _conversation(message, channel)
        identity = await _by(
            deadline, self._routing_identity(channel, sender, organization_id), room_id=None
        )
        own = known_as(sender, identity)
        for _ in range(_ROUTES):
            room_id, created = await self._find_or_open_room(
                message,
                channel,
                None,
                organization_id,
                identity,
                deadline=deadline,
                opening=opening,
            )
            if not created:
                # Its room known only now, a message its room's own work
                # routes back into it cannot be adopted (RFC §9.5.1).
                self._room_work.refuse_routed(room_id)
            if created or await self._claim_room(room_id, message, channel, own, deadline):
                return room_id, created
            logger.debug("Room %s is no longer the routed message's; routing it again", room_id)
        _charge_opening(opening)
        return await self._open_room_for(message, None, organization_id), True

    async def _routing_identity(
        self, channel: Channel, sender: str, organization_id: str | None
    ) -> Identity | None:
        """The sender's identity as routing names them (RFC §10.4, §14.1).

        Read once per message, before the first route, within the caller's
        organization (§17.2), so the router and the record's check know the
        sender alike. A message with no address, and the framework's own
        sender, read none.
        """
        if not recordable_sender(sender):
            return None
        return await self._store.resolve_identity(
            str(channel.channel_type), sender, organization_id
        )

    async def _claim_room(
        self,
        room_id: str,
        message: InboundMessage,
        channel: Channel,
        own: frozenset[str],
        deadline: float,
    ) -> bool:
        """Whether *room_id* is still the routed message's, recording the sender on it.

        Checked under the room lock, where every binding mutation and every
        status change runs, against the room as the lock finds it
        (``still_the_messages``, RFC §10.1 step 2); *own* holds the ids the
        sender is known by. A room no longer the message's is left as it was.
        Otherwise a room that does not bind the channel has it attached, and
        a binding that is not a group and names no one records the sender as
        its correspondent: the router then finds them there and admits no
        one else (§10.4). The wait for the lock is bounded by the call's
        *deadline*, and the section it opens is not (§13.6): past it the
        message is refused, never let in unrecorded, which would leave the
        room open to a concurrent stranger. Nor is it cut by its caller:
        once the lock is held, the section runs to its end whatever becomes
        of the caller, as a room step 2 opens does (:meth:`_open_room_for`).
        """
        async with AsyncExitStack() as stack:
            await self._enter_lock_by(stack, room_id, deadline)
            return await run_to_its_end(
                self._claim_locked(room_id, message, channel, own),
                held=self._decided_writes,
                name=f"roomkit-decided-write-{room_id}",
            )

    async def _claim_locked(
        self, room_id: str, message: InboundMessage, channel: Channel, own: frozenset[str]
    ) -> bool:
        """:meth:`_claim_room`'s section, under the room lock its caller holds."""
        channel_id, sender = message.channel_id, _conversation(message, channel)
        room = await self._store.get_room(room_id)
        if room is None:
            return False
        # RFC §5.1, §10.1 step 2: the room's due timers apply first, so a
        # room due to close is no longer the message's.
        room = await self._room_timers.apply_due(room)
        binding = await self._store.get_binding(room_id, channel_id)
        recipient = routed_recipient(channel, binding) if binding is not None else None
        if not await still_the_messages(
            self._store, room, channel_id, binding, sender, own, recipient
        ):
            return False
        if binding is None:
            binding = await self._attach_unbound(room, channel_id)
        if binding is not None and recordable_sender(sender):
            await self._record_sender(room_id, binding, message, channel)
        return True

    async def _attach_unbound(
        self,
        room: Room,
        channel_id: str,
        *,
        participant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelBinding | None:
        """Step 2's attach of *channel_id* to *room*, read under the room lock
        the caller holds (RFC §10.1 step 2, §7.5 rule 7): the room's binding
        of the channel; one made by then is kept as it is, never replaced.
        ``None`` when the integrator detached the channel from the room: it
        is not attached again, and the message goes on unbound.

        Otherwise the channel is attached at its declaration's defaults
        (read-write, unmuted, visible to all, its capabilities), naming
        *participant_id* and carrying *metadata*, and the binding is read
        again: its ``ON_CHANNEL_ATTACHED`` hooks run under the same lock and
        may have changed it. The caller decides whether the room's status
        lets anything be attached.
        """
        binding = await self._store.get_binding(room.id, channel_id)
        if binding is not None:
            return binding
        if is_channel_detached(room, channel_id):
            return None
        await self.attach_channel(
            room.id, channel_id, participant_id=participant_id, metadata=metadata or {}
        )
        return await self._store.get_binding(room.id, channel_id)

    async def _record_sender(
        self, room_id: str, binding: ChannelBinding, message: InboundMessage, channel: Channel
    ) -> None:
        """Write the sender on *binding*, read under the room lock, where it names no one.

        The binding that names the sender by their address also gets them as
        its recipient where it holds no value (``channel.reply_metadata``):
        the room then replies to the person it is the conversation of,
        including a binding recorded before it carried a recipient. A
        recipient already there, the host's above all, is never replaced, nor
        is a value that is not a recipient (RFC §5.7), whose deliveries stay
        refused ``no_recipient``. Pushed to the channel as every binding
        update is; a group binding is never written.
        """
        if binding.group:
            return
        update: dict[str, Any] = {}
        if binding_names_no_one(binding):
            # A binding stored with a recipient and no correspondent (before
            # RFC §5.7 named one) is its recipient's: the sender is recorded
            # only on a binding that delivers to no one, or to them.
            update["participant_id"] = channel.recipient_address(binding) or _conversation(
                message, channel
            )
        if _lacks_recipient(binding.model_copy(update=update), message, channel):
            missing = {
                key: value
                for key, value in channel.reply_metadata(message).items()
                if read_recipient(binding.metadata, key) is None
            }
            update["metadata"] = {**binding.metadata, **missing}
        if not update:
            return
        binding = binding.model_copy(update=update)
        await self._store.update_binding(binding)
        self._notify_binding_updated(room_id, message.channel_id, binding)

    async def _maybe_auto_attach(self, room_id: str, channel_id: str, deadline: float) -> None:
        """Attach *channel_id* to the named room if it was never bound — not
        if it was revoked, and not to a room that refuses events.

        Auto-attach is a convenience for a channel that was never bound;
        re-granting access that was explicitly revoked is not its job
        (RFC §7.5-7). The decision runs under the room lock against fresh
        reads, the room's due timers applied first (§5.1, §10.1 step 2): a
        room CLOSED or ARCHIVED by then gets no binding, and step 6 refuses
        the message ``room_closed``. detach_channel() writes its tombstone
        and removes the binding under the same lock, so this sees either the
        binding or the revocation — a detach landing concurrently with an
        in-flight message can never be undone by it. The tombstone lives in
        room metadata, so the check holds across restarts and across workers
        sharing the store. attach_channel() re-acquires the lock reentrantly.
        The wait for the lock is bounded by the call's *deadline*, and the
        section it opens is not (RFC §13.6): past it the message is refused,
        never let in unattached. Nor is the section cut by its caller: it
        runs to its end, the attach's record and its ON_CHANNEL_ATTACHED
        hooks included, as a room step 2 opens does.
        """
        async with AsyncExitStack() as stack:
            await self._enter_lock_by(stack, room_id, deadline)
            await run_to_its_end(
                self._auto_attach_locked(room_id, channel_id),
                held=self._decided_writes,
                name=f"roomkit-decided-write-{room_id}",
            )

    async def _auto_attach_locked(self, room_id: str, channel_id: str) -> None:
        """:meth:`_maybe_auto_attach`'s section, under the room lock its
        caller holds."""
        room = await self._store.get_room(room_id)
        if room is None:
            return
        room = await self._room_timers.apply_due(room)
        if room.status in _REFUSING_STATUSES:
            return
        await self._attach_unbound(room, channel_id)

    async def _enter_lock_by(self, stack: AsyncExitStack, room_id: str, deadline: float) -> None:
        """Enter *room_id*'s lock on *stack*, step 2's wait for it bounded by
        the call's *deadline* (RFC §13.6): past it,
        :class:`_RoutingTimeoutError` naming the room."""
        await _by(
            deadline,
            stack.enter_async_context(self._lock_manager.locked(room_id)),
            room_id=room_id,
        )

    async def _fire_text_session_started(
        self,
        room_id: str,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str,
    ) -> None:
        """Fire ON_SESSION_STARTED for text channel room auto-creation.

        Internal hooks (name starts with ``_``) are awaited so the greeting
        gate mechanism completes before the first inbound message is processed.
        User hooks are fired in the background to avoid blocking the pipeline.
        """
        try:
            from roomkit.models.session_event import SessionStartedEvent

            context = await self._build_context(room_id)
            event = SessionStartedEvent(
                room_id=room_id,
                channel_id=channel_id,
                channel_type=channel_type,
                participant_id=participant_id,
            )
            # Await internal hooks (auto-greet must complete for gate ordering)
            await self._hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_SESSION_STARTED,
                event,
                context,
                skip_event_filter=True,
                name_prefix="_",
            )
            # Fire-and-forget user hooks (slow hooks must not block inbound).
            # Track via _pending_hook_tasks to prevent GC and ensure graceful
            # cancellation in close().
            task = asyncio.get_running_loop().create_task(
                self._hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_SESSION_STARTED,
                    event,
                    context,
                    skip_event_filter=True,
                    exclude_name_prefix="_",
                )
            )
            task.add_done_callback(self._pending_hook_tasks.discard)
            self._pending_hook_tasks.add(task)
            await self._emit_framework_event(
                "session_started",
                room_id=room_id,
                data={
                    "channel_id": channel_id,
                    "channel_type": str(channel_type),
                },
            )
        except Exception:
            logger.exception("Error firing ON_SESSION_STARTED for text channel")


class _PastTheCapError(Exception):
    """A message the room's delivery work sent would create a room past the
    reentry budget it draws on (RFC §8.3, gap G-070): refused with nothing
    written."""


@dataclass(slots=True)
class _Opening:
    """What a message of the room's delivery work is to that work when it
    creates a room, routed or named (RFC §9.5.1, gap G-070), charged once
    whichever branch creates the room."""

    entry: Entry
    charged: bool = False

    def charge(self) -> None:
        """Take one pass, or one hook call for an ASYNC hook's body's
        message, before the room is created; nothing the second time.

        Raises:
            _PastTheCapError: the budget is spent.
        """
        if self.charged:
            return
        if not self.entry.draw():
            raise _PastTheCapError
        self.charged = True


def _charge_opening(opening: _Opening | None) -> None:
    """Charge *opening*, if any, before a room is created for the message."""
    if opening is not None:
        opening.charge()


class _RoutingTimeoutError(Exception):
    """The call's ``process_timeout`` passed during step 2 (RFC §13.6): in
    the router or the identity read, *room_id* ``None``, or in a read of the
    room routing gave it or the wait for its lock."""

    def __init__(self, room_id: str | None) -> None:
        super().__init__(room_id)
        self.room_id = room_id


async def _by[T](deadline: float, work: Awaitable[T], *, room_id: str | None) -> T:
    """*work* bounded by *deadline*, the call's ``process_timeout`` (RFC
    §13.6): past it, :class:`_RoutingTimeoutError` naming *room_id*."""
    timeout = asyncio.timeout_at(deadline)
    try:
        async with timeout:
            return await work
    except TimeoutError:
        if not timeout.expired():
            raise
        raise _RoutingTimeoutError(room_id) from None


def _past_the_cap(room_id: str | None, message: InboundMessage) -> InboundResult:
    """The refusal of a message the room's own work sent past the reentry
    budget it draws on (RFC §8.3, §9.5.1): charged as its arrival starts,
    before its parse, it is refused with no event. *room_id* is ``None`` for
    a message that would have opened a new room."""
    logger.warning(
        "A message sent into %s from its own work is past the reentry budget: refused",
        f"room {room_id}" if room_id is not None else "a new room",
        extra={"room_id": room_id, "channel_id": message.channel_id},
    )
    return InboundResult(blocked=True, reason=REENTRY_LOOP_CAP)


def _at_least_as_deep(event: RoomEvent) -> RoomEvent:
    """*event* at least as deep as the turn the call runs within (RFC §8.3):
    a message a turn sends continues its chain."""
    floor = chain_floor()
    if event.chain_depth >= floor:
        return event
    return event.model_copy(update={"chain_depth": floor})


def _apply_message_fields(event: RoomEvent, message: InboundMessage) -> RoomEvent:
    """The event a channel parsed, with the caller's fields it did not set.

    Applied centrally, so every channel's ``handle_inbound`` carries them
    (each builds its own RoomEvent and would otherwise have to remember to
    copy each one). A field the channel resolved itself keeps its value.
    """
    event = _from_its_channel(event, message)
    if message.event_type == EventType.INSTRUCTION:
        event = _as_instruction(event, message)
    elif event.type == EventType.INSTRUCTION:
        event = _not_an_instruction(event, message)

    # Caller-requested visibility (e.g. ``"transport"`` for a proactive
    # notification that must not wake the room's intelligence channel).
    if message.visibility != Visibility.ALL and event.visibility == Visibility.ALL:
        event = event.model_copy(update={"visibility": message.visibility})

    # In-app thread parent, applied centrally so every channel's
    # handle_inbound carries it (each builds its own RoomEvent and would
    # otherwise have to remember to copy it). The locked pipeline then
    # normalises it to the thread root.
    if message.parent_event_id is not None and event.parent_event_id is None:
        event = event.model_copy(update={"parent_event_id": message.parent_event_id})

    # Addressing (RFC §19.3), applied centrally for the same reason. A
    # channel that reads an address off its own wire format sets it in
    # handle_inbound and keeps it — the caller's address only fills a gap.
    if message.addressed_to is not None and event.addressed_to is None:
        event = event.model_copy(update={"addressed_to": list(message.addressed_to)})

    # Carry the caller's publication key even when the channel only parses
    # content. The locked pipeline owns deduplication for every transport.
    if message.idempotency_key is not None and event.idempotency_key is None:
        event = event.model_copy(update={"idempotency_key": message.idempotency_key})
    # A key in the framework's publication namespace is the delivery path's
    # alone (RFC §22.1): dropped from any other message, whoever wrote it.
    admitted = admitted_key(event.idempotency_key)
    if admitted != event.idempotency_key:
        event = event.model_copy(update={"idempotency_key": admitted})

    # The chain the message continues (RFC §8.3, §23.3): a transport parses
    # a person's words, which open one, so only the caller sets it.
    if message.chain_depth and not event.chain_depth:
        event = event.model_copy(update={"chain_depth": message.chain_depth})

    # Where this message's answer may go — same central application, same
    # rule: a channel that resolved one itself keeps it.
    if message.response_visibility is not None and event.response_visibility is None:
        event = event.model_copy(update={"response_visibility": message.response_visibility})
    return _without_runtime_keys(event)


def _from_its_channel(event: RoomEvent, message: InboundMessage) -> RoomEvent:
    """*event* sourced from the channel that received *message* (RFC §10.1
    step 4): a parse cannot pass the message off as another channel's,
    whose binding, permissions and delivery set it would take. What else
    the parse wrote of its source stands: an SMS with media is an MMS, and
    a channel's echo of its own output travelled outbound (RFC §10.3 rule
    3)."""
    if event.source.channel_id == message.channel_id:
        return event
    logger.warning(
        "Channel %s parsed a message sourced from %s: it is sourced from %s",
        message.channel_id,
        event.source.channel_id,
        message.channel_id,
        extra={"channel_id": message.channel_id},
    )
    received = event.source.model_copy(update={"channel_id": message.channel_id})
    return event.model_copy(update={"source": received})


_RUNTIME_KEYS = (RUNTIME_RECORD, HELD_KEY, FINGERPRINT)
"""Metadata keys only the runtime writes: which text is the runtime's (RFC
§6.4), which message it holds for verification (§11.7.3), whose release
would otherwise hand a sender's message to the agents as another's, and
which instruction produced a response (§5.2, §10.1.1 step 6)."""


def _without_runtime_keys(event: RoomEvent) -> RoomEvent:
    """*event* without a runtime key its sender supplied. Last of the fields
    applied, so that no merge of the caller's metadata (an instruction's)
    brings one back."""
    if not any(key in event.metadata for key in _RUNTIME_KEYS):
        return event
    metadata = {k: v for k, v in event.metadata.items() if k not in _RUNTIME_KEYS}
    return event.model_copy(update={"metadata": metadata})


def _as_instruction(event: RoomEvent, message: InboundMessage) -> RoomEvent:
    """The event as the instruction the caller sent (RFC §10.1.1).

    An instruction is the caller's decision, not the wire's: a channel parses
    content and need not know the type, so the pipeline applies it, and its
    address, the one checked before anything was read (step 1). The caller's
    metadata rides it (step 4); standalone is the typed field's word alone
    (step 7): a caller's metadata key of the same name is dropped, never
    honoured. The instruction is never stored, so nothing is lost.
    """
    event = event.model_copy(
        update={"type": EventType.INSTRUCTION, "addressed_to": list(message.addressed_to or ())}
    )
    # The caller's metadata rides the instruction, whatever the transport's
    # parser kept of it; a key the parser set keeps its value (RFC §10.1.1
    # step 4): a delegation's hand-back names its task (§23.3 step 8).
    merged = {**message.metadata, **event.metadata}
    metadata = {k: v for k, v in merged.items() if k != STANDALONE}
    if message.standalone:
        metadata[STANDALONE] = True
    return event.model_copy(update={"metadata": metadata})


def _not_an_instruction(event: RoomEvent, message: InboundMessage) -> RoomEvent:
    """*event*, which its channel's parser typed INSTRUCTION, as the message
    the caller sent: an instruction is the caller's decision (RFC §10.1.1),
    and the wire would otherwise issue the application's directive past the
    checks the caller's takes before anything is read. The parser's
    ``standalone`` word goes with the type."""
    logger.warning(
        "Channel %s parsed a %s as an INSTRUCTION: only the caller issues one, it stays a %s",
        message.channel_id,
        message.event_type,
        message.event_type,
        extra={"channel_id": message.channel_id},
    )
    metadata = {k: v for k, v in event.metadata.items() if k != STANDALONE}
    return event.model_copy(update={"type": message.event_type, "metadata": metadata})


def _conversation(message: InboundMessage, channel: Channel) -> str:
    """The address *message* is routed and recorded by (RFC §10.4).

    The channel's conversation address (the chat, on a chat channel), except
    for a message the framework writes itself: its sender stays ``system``
    whatever its metadata says, so it is never recorded on a binding.
    """
    if not recordable_sender(message.sender_id):
        return message.sender_id
    return channel.conversation_address(message)


def _lacks_recipient(binding: ChannelBinding, message: InboundMessage, channel: Channel) -> bool:
    """Whether *binding* names *message*'s conversation but holds no value
    where its reply address goes. A value that is not a recipient is one
    held (RFC §5.7): the record does not write over it."""
    if binding.participant_id != _conversation(message, channel):
        return False
    return any(
        read_recipient(binding.metadata, key) is None for key in channel.reply_metadata(message)
    )
