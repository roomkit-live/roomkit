"""DeliverMixin — framework-level content delivery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.delivery import DeliveryContext, DeliveryStrategy, Immediate, resolve_strategy
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.serialization import serialize_strategy
from roomkit.delivery.worker import build_delivery_hook_event
from roomkit.models.enums import EventStatus, HookTrigger

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.delivery.base import DeliveryBackend
    from roomkit.models.context import RoomContext

logger = logging.getLogger("roomkit.delivery")


@runtime_checkable
class DeliveryHost(Protocol):
    """Contract: capabilities a host class must provide for DeliverMixin.

    Attributes provided by the host's ``__init__``:
        _delivery_strategy: Default delivery strategy (or ``None`` for immediate).
        _delivery_backend: Async delivery backend for the enqueue path.
        _hook_engine: Engine for BEFORE_DELIVER / AFTER_DELIVER hook execution.

    Methods provided by HelpersMixin (or equivalent):
        _build_context: Build a :class:`~roomkit.models.context.RoomContext`
            for hook invocation.
    """

    _delivery_strategy: DeliveryStrategy | None
    _delivery_backend: DeliveryBackend | None
    _hook_engine: HookEngine

    async def _build_context(self, room_id: str) -> RoomContext: ...


class DeliverMixin(HelpersMixin):
    """Adds ``deliver()`` to RoomKit for proactive content delivery.

    Host contract: :class:`DeliveryHost`.
    """

    _delivery_strategy: DeliveryStrategy | None
    _delivery_backend: DeliveryBackend | None

    async def deliver(
        self,
        room_id: str,
        content: str,
        *,
        channel_id: str | None = None,
        strategy: DeliveryStrategy | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Deliver content to a room/channel.

        Sends *content* to the target channel with awareness of channel
        state (voice playing, user speaking, idle).

        When a :class:`~roomkit.delivery.base.DeliveryBackend` is
        configured, the item is enqueued and the worker executes
        delivery asynchronously.  Otherwise delivery happens in-process
        with ``BEFORE_DELIVER`` / ``AFTER_DELIVER`` hooks.

        Args:
            room_id: Target room ID.
            content: Text content to deliver.
            channel_id: Target channel ID.  If ``None``, auto-detects
                the best transport channel (prefers voice).
            strategy: Delivery strategy — controls **when** to deliver.
                Accepts a :class:`DeliveryStrategy` instance or a string
                shorthand (``"immediate"``, ``"wait_for_idle"``,
                ``"queued"``).  Falls back to the framework default.
            metadata: Optional metadata attached to the delivery event.
        """
        resolved = resolve_strategy(strategy) or self._delivery_strategy
        if resolved is None:
            resolved = Immediate()

        # Backend path: enqueue instead of executing in-process
        if self._delivery_backend is not None:
            item = DeliveryItem(
                room_id=room_id,
                content=content,
                channel_id=channel_id,
                strategy=serialize_strategy(resolved),
                metadata=metadata or {},
            )
            await self._delivery_backend.enqueue(item)
            return

        # In-process path (no backend configured)
        ctx = DeliveryContext(
            kit=self,  # ty: ignore[invalid-argument-type]
            room_id=room_id,
            content=content,
            channel_id=channel_id,
            metadata=metadata,
        )

        strategy_name = serialize_strategy(resolved).get("type", "immediate")
        extra_meta = dict(metadata) if metadata else {}

        # BEFORE_DELIVER
        hook_event = build_delivery_hook_event(
            room_id,
            content,
            channel_id=channel_id,
            strategy_name=strategy_name,
            extra_meta=extra_meta,
        )
        try:
            room_context = await self._build_context(room_id)
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.BEFORE_DELIVER, hook_event, room_context
            )
        except Exception:
            logger.debug("BEFORE_DELIVER hook failed", exc_info=True)

        # Execute delivery strategy
        error: str | None = None
        try:
            await resolved.deliver(ctx)
        except Exception as exc:
            error = str(exc)
            logger.exception("Delivery failed in room %s", room_id)

        # AFTER_DELIVER
        after_extra = {**extra_meta, "error": error}
        after_event = build_delivery_hook_event(
            room_id,
            content,
            channel_id=channel_id,
            strategy_name=strategy_name,
            status=EventStatus.FAILED if error else EventStatus.DELIVERED,
            extra_meta=after_extra,
        )
        try:
            room_context = await self._build_context(room_id)
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.AFTER_DELIVER, after_event, room_context
            )
        except Exception:
            logger.debug("AFTER_DELIVER hook failed", exc_info=True)
