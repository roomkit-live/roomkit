"""InboundMixin — inbound message processing entry point and routing."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.core.mixins.channel_ops import is_channel_detached
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.mixins.inbound_identity import _IdentityBlockedError
from roomkit.models.delivery import DeliveryHandle, InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelType,
    HookTrigger,
    Visibility,
)

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.inbound_router import InboundRoomRouter
    from roomkit.core.locks import RoomLockManager
    from roomkit.identity.base import IdentityResolver
    from roomkit.models.context import RoomContext
    from roomkit.models.identity import Identity, IdentityResult
    from roomkit.store.base import ConversationStore
    from roomkit.telemetry.base import TelemetryProvider

logger = logging.getLogger("roomkit.framework")


@runtime_checkable
class InboundHost(Protocol):
    """Contract: capabilities a host class must provide for InboundMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _channels: Registry of channel-id to :class:`Channel` instances.
        _lock_manager: Per-room lock for serialised mutation.
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
        _process_streaming_responses: From :class:`InboundStreamingMixin`.
        create_room: From :class:`RoomLifecycleMixin`.
        attach_channel: From :class:`ChannelOpsMixin`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _lock_manager: RoomLockManager
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
    _identity_resolver: IdentityResolver | None
    _identity_channel_types: set[ChannelType] | None
    _identity_timeout: float
    _process_timeout: float
    _inbound_router: InboundRoomRouter
    _max_chain_depth: int
    _inbound_rate_limiter: Any  # TokenBucketRateLimiter | None
    _inbound_rate_limit: Any  # RateLimit | None
    _telemetry: TelemetryProvider

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _resolve_identity: Any  # see InboundHost
    _process_locked: Any  # see InboundHost
    _process_streaming_responses: Any  # see InboundHost
    _consume_streams_when_cascade_completes: Any  # see LaneExecutionMixin
    create_room: Any  # see InboundHost
    attach_channel: Any  # see InboundHost

    async def process_inbound(
        self,
        message: InboundMessage,
        *,
        room_id: str | None = None,
        defer_delivery: bool = False,
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
        """
        from roomkit.telemetry.base import SpanKind
        from roomkit.telemetry.context import get_current_span, reset_span, set_current_span

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

        telemetry = self._telemetry
        inbound_span_id = telemetry.start_span(
            SpanKind.INBOUND_PIPELINE,
            "framework.inbound",
            parent_id=get_current_span(),
            channel_id=message.channel_id,
            room_id=room_id,
            attributes={"sender_id": message.sender_id or ""},
        )
        # Propagate backend-specific context for robust parent linking
        token = set_current_span(
            inbound_span_id, telemetry_ctx=telemetry.get_span_context(inbound_span_id)
        )
        _inbound_result: InboundResult | None = None
        try:
            _inbound_result = await self._process_inbound_inner(
                message, channel, room_id, telemetry, inbound_span_id, defer_delivery
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

    async def _process_inbound_inner(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str | None,
        telemetry: Any,
        inbound_span_id: str,
        defer_delivery: bool,
    ) -> InboundResult:
        """Inner inbound processing (extracted for telemetry wrapping)."""

        # Route to room (or auto-create). Deliberately outside the pre-commit
        # window below: routing is RFC §10.1 steps 1-2, the phase
        # ``process_timeout`` bounds is steps 3-11 — and this is the one part
        # that can create a room, so a timeout here would leave an orphan
        # behind for a message that was refused.
        room_id, room_just_created = await self._route_to_room(
            message, channel, room_id, telemetry, inbound_span_id
        )

        # One budget for the whole pre-commit phase (RFC §13.6), as an absolute
        # deadline so it can be shared with the locked region rather than
        # re-applied there. It covers the wait for the room lock on purpose:
        # the pile-up this setting exists to stop is one stuck event holding a
        # room's lock while every later message queues behind it, and a window
        # that stopped at the lock would leave exactly that unbounded.
        deadline = asyncio.get_running_loop().time() + self._process_timeout
        try:
            prepared = await self._prepare_event(
                message, channel, room_id, room_just_created, deadline, telemetry
            )
        except TimeoutError:
            return await self._refuse_on_timeout(room_id, message.channel_id)
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
            defer_delivery=defer_delivery,
        )

    async def _refuse_on_timeout(self, room_id: str, channel_id: str) -> InboundResult:
        logger.error(
            "Inbound pre-commit timed out after %.1fs",
            self._process_timeout,
            extra={"room_id": room_id, "channel_id": channel_id},
        )
        await self._emit_framework_event(
            "process_timeout",
            room_id=room_id,
            channel_id=channel_id,
            data={"timeout": self._process_timeout},
        )
        return InboundResult(blocked=True, reason="process_timeout")

    async def _prepare_event(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        room_just_created: bool,
        deadline: float,
        telemetry: Any,
    ) -> tuple[Any, RoomContext, Identity | None, IdentityResult | None] | InboundResult:
        """RFC §10.1 steps 3-5, under the caller's pre-commit budget.

        Everything here was previously unbounded: a store read that never
        returns during the context build, or a channel's own
        ``handle_inbound`` reaching a provider without a timeout of its own,
        held the caller forever while ``process_timeout`` sat configured and
        inert.

        Raises:
            TimeoutError: The budget ran out. Nothing durable has been written
                at this point — the room may have been created by the routing
                step above, which is deliberately outside the window.
        """
        async with asyncio.timeout_at(deadline):
            return await self._prepare_event_inner(
                message, channel, room_id, room_just_created, telemetry
            )

    async def _prepare_event_inner(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        room_just_created: bool,
        telemetry: Any,
    ) -> tuple[Any, RoomContext, Identity | None, IdentityResult | None] | InboundResult:
        context = await self._build_context(room_id)

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

        # Let channel process inbound
        event = await channel.handle_inbound(message, context)

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

        # Where this message's answer may go — same central application, same
        # rule: a channel that resolved one itself keeps it.
        if message.response_visibility is not None and event.response_visibility is None:
            event = event.model_copy(update={"response_visibility": message.response_visibility})

        # Identity resolution pipeline (RFC §11)
        try:
            event, resolved_identity, pending_id_result = await self._resolve_identity(
                event, message, channel, room_id, context, telemetry
            )
        except _IdentityBlockedError as exc:
            return InboundResult(blocked=True, reason=exc.reason)

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
        *,
        defer_delivery: bool = False,
    ) -> InboundResult:
        """RFC §10.1 steps 6-18: the locked region, then the delivery set.

        The room lock is acquired under the same budget as the phase before it
        (§13.6): a room whose lock is held by a stuck event would otherwise
        queue every later message with nothing to stop it, which is the pile-up
        the setting exists for. Past the lock, ``_process_locked`` spends what
        remains on the pre-commit gates and stops there — the commit and the
        delivery set it hands to the lane are not cancellable, or a committed
        event would come back reported as blocked.
        """
        from roomkit.core.lanes import DeliveryCascade

        cascade = DeliveryCascade(room_id, reentry_budget=self._max_chain_depth * 10)
        async with AsyncExitStack() as stack:
            try:
                async with asyncio.timeout_at(deadline):
                    await stack.enter_async_context(self._lock_manager.locked(room_id))
            except TimeoutError:
                return await self._refuse_on_timeout(room_id, message.channel_id)
            result: InboundResult = await self._process_locked(
                event,
                room_id,
                context,
                cascade,
                resolved_identity=resolved_identity,
                pending_id_result=pending_id_result,
                deadline=deadline,
            )

        if defer_delivery:
            # Deferred completion (RFC §10.1 step 18): the caller takes the
            # committed event now — blocked was decided under the lock above,
            # so a refusal is still synchronous — and the rest of the turn
            # (delivery set, reentry passes, streamed responses) follows in
            # the room's lane, its streams consumed on the same background
            # task a detached caller uses. The handle is the caller's grip on
            # that tail; it backfills delivery_results/error on completion.
            # Every result that reached this locked region gets one — a hook
            # refusal included, whose near-empty cascade resolves at once. A
            # refusal shed before it (rate limited, pre-commit timeout,
            # identity block) returned above with delivery=None: there is no
            # delivery to follow.
            consumer = self._consume_streams_when_cascade_completes(cascade, room_id)
            result.delivery = DeliveryHandle(cascade, consumer, result)
            await self._connect_session_if_ready(message, channel, room_id, result)
            return result

        # The caller observes its event's delivery-set completion (RFC §10.1
        # step 18): wait for the cascade — the trigger's delivery set plus
        # every reentry pass it transitively spawned. AFTER_BROADCAST,
        # mutation and ON_ERROR hooks fire from the lane executor, off this
        # room's lock.
        completed = await cascade.wait()
        if cascade.error is not None and result.error is None:
            result.error = cascade.error
        # Step 18 reports the delivery set the caller waited for.
        result.delivery_results = cascade.delivery_results

        # Handle streaming responses outside the lane (TTS delivery can take
        # seconds; the lane must not stall behind it). A failure while
        # consuming the response stream is surfaced on the result so a
        # headless caller can react (interactive callers ignore it — the
        # ON_ERROR hooks already fired an error card). A detached caller (a
        # reentrant process_inbound issued from a hook or a tool handler)
        # hands the consumption to a background task instead — a streaming
        # reply is only generated when its stream is consumed.
        if not completed:
            self._consume_streams_when_cascade_completes(cascade, room_id)
        elif cascade.streams:
            stream_error = await self._process_streaming_responses(cascade.streams, room_id)
            if stream_error is not None and result.error is None:
                result.error = stream_error

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
    ) -> tuple[str, bool]:
        """Route inbound message to a room, auto-creating if needed.

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
        room_just_created = False
        try:
            if room_id is None:
                room_id = await self._inbound_router.route(
                    channel_id=message.channel_id,
                    channel_type=channel.channel_type,
                    participant_id=message.sender_id,
                )
            if room_id is None:
                # Auto-create room and attach channel
                room = await self.create_room()
                room_id = room.id
                await self.attach_channel(room_id, message.channel_id)
                room_just_created = True
            else:
                # Ensure room exists; auto-create if needed (e.g. voice session
                # with a room_id from SIP headers that hasn't been created yet).
                # Existence checks only — neither the room nor the binding is
                # used beyond the yes/no, so materialising either would decode
                # its JSONB columns and validate a whole model per message for
                # nothing. Asked together on one connection; what they decide
                # (create, attach, lock) happens outside it, because a pooled
                # connection must never be held across a lock or a write path.
                async with self._store.connection():
                    room_present = await self._store.room_exists(room_id)
                    binding_present = room_present and await self._store.binding_exists(
                        room_id, message.channel_id
                    )
                if not room_present:
                    await self.create_room(room_id=room_id)
                    await self.attach_channel(room_id, message.channel_id)
                    room_just_created = True
                elif not binding_present:
                    # Room exists but the channel is not attached — unless the
                    # integrator detached it (RFC §7.5-7). The unlocked read
                    # keeps the common case (already bound) free; the actual
                    # decision runs under the room lock in _maybe_auto_attach.
                    await self._maybe_auto_attach(room_id, message.channel_id)
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

    async def _maybe_auto_attach(self, room_id: str, channel_id: str) -> None:
        """Attach *channel_id* if it was never bound — not if it was revoked.

        Auto-attach is a convenience for a channel that was never bound;
        re-granting access that was explicitly revoked is not its job
        (RFC §7.5-7). The decision runs under the room lock against fresh
        reads: detach_channel() writes its tombstone and removes the binding
        under the same lock, so this sees either the binding or the
        revocation — a detach landing concurrently with an in-flight message
        can never be undone by it. The tombstone lives in room metadata, so
        the check holds across restarts and across workers sharing the
        store. attach_channel() re-acquires the lock reentrantly.
        """
        async with self._lock_manager.locked(room_id):
            room = await self._store.get_room(room_id)
            if room is None:
                return
            binding = await self._store.get_binding(room_id, channel_id)
            if binding is None and not is_channel_detached(room, channel_id):
                await self.attach_channel(room_id, channel_id)

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
