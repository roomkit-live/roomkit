"""HooksApiMixin — hook decorators, webhook processing, and delivery status."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.hooks import (
    AsyncHookFn,
    HookRegistration,
    SyncHookFn,
)
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.context import RoomContext
from roomkit.models.delivery import DeliveryStatus
from roomkit.models.enums import (
    ChannelDirection,
    ChannelType,
    HookExecution,
    HookTrigger,
    RoomStatus,
)

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine, IdentityHookRegistration
    from roomkit.core.mixins.helpers import FrameworkEventHandler, IdentityHookFn
    from roomkit.providers.sms.meta import WebhookMeta
    from roomkit.store.base import ConversationStore


logger = logging.getLogger("roomkit.framework")


class HooksApiMixin(HelpersMixin):
    """Hook decorators, webhook processing, and delivery status."""

    _store: ConversationStore
    _hook_engine: HookEngine
    _event_handlers: list[tuple[str, FrameworkEventHandler]]
    _identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]]

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
            await self.process_inbound(inbound)  # type: ignore[attr-defined]
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
