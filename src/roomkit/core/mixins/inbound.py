"""InboundMixin — inbound message processing entry point and routing."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.mixins.inbound_identity import _IdentityBlockedError
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelType,
    HookTrigger,
)

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.inbound_router import InboundRoomRouter
    from roomkit.core.locks import RoomLockManager
    from roomkit.identity.base import IdentityResolver
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
    create_room: Any  # see InboundHost
    attach_channel: Any  # see InboundHost

    async def process_inbound(
        self, message: InboundMessage, *, room_id: str | None = None
    ) -> InboundResult:
        """Process an inbound message through the full pipeline.

        Args:
            message: The inbound message to process.
            room_id: Explicit room to route to, bypassing the inbound router.
                Useful for shared channels attached to multiple rooms.
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
                message, channel, room_id, telemetry, inbound_span_id
            )
            return _inbound_result
        except Exception as exc:
            telemetry.end_span(inbound_span_id, status="error", error_message=str(exc))
            raise
        finally:
            reset_span(token)
            if _inbound_result is not None:
                telemetry.end_span(
                    inbound_span_id,
                    attributes={"blocked": _inbound_result.blocked},
                )

    async def _process_inbound_inner(
        self,
        message: InboundMessage,
        channel: Channel,
        room_id: str | None,
        telemetry: Any,
        inbound_span_id: str,
    ) -> InboundResult:
        """Inner inbound processing (extracted for telemetry wrapping)."""

        # Route to room (or auto-create)
        room_id, room_just_created = await self._route_to_room(
            message, channel, room_id, telemetry, inbound_span_id
        )

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

        # Identity resolution pipeline (RFC §7)
        try:
            event, resolved_identity, pending_id_result = await self._resolve_identity(
                event, message, channel, room_id, context, telemetry
            )
        except _IdentityBlockedError as exc:
            return InboundResult(blocked=True, reason=exc.reason)

        # Process under room lock
        pending_streams: list[Any] = []
        async with self._lock_manager.locked(room_id):
            try:
                result: InboundResult = await asyncio.wait_for(
                    self._process_locked(
                        event,
                        room_id,
                        context,
                        resolved_identity=resolved_identity,
                        pending_id_result=pending_id_result,
                        pending_streams_out=pending_streams,
                    ),
                    timeout=self._process_timeout,
                )
            except TimeoutError:
                logger.error(
                    "Process locked timed out after %.1fs",
                    self._process_timeout,
                    extra={"room_id": room_id, "event_id": event.id},
                )
                await self._emit_framework_event(
                    "process_timeout",
                    room_id=room_id,
                    event_id=event.id,
                    data={"timeout": self._process_timeout},
                )
                return InboundResult(blocked=True, reason="process_timeout")

        # Handle streaming responses outside lock (TTS delivery can take seconds;
        # holding the lock would block concurrent process_inbound calls)
        if pending_streams:
            await self._process_streaming_responses(pending_streams, room_id)

        # Bind session for stateful channels (voice, persistent WS, etc.)
        # Runs AFTER hooks passed and the event was stored — a blocked
        # event never reaches connect_session.
        if message.session is not None and not result.blocked:
            binding = await self._store.get_binding(room_id, message.channel_id)
            if binding is not None:
                await channel.connect_session(message.session, room_id, binding)

        return result

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
                room = await self._store.get_room(room_id)
                if room is None:
                    room = await self.create_room(room_id=room_id)
                    await self.attach_channel(room_id, message.channel_id)
                    room_just_created = True
                else:
                    # Room exists — ensure channel is attached
                    binding = await self._store.get_binding(room_id, message.channel_id)
                    if binding is None:
                        await self.attach_channel(room_id, message.channel_id)
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
