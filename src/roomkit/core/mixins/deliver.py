"""DeliverMixin — framework-level content delivery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core._reentrance import chain_floor
from roomkit.core.delivery import DeliveryStrategy, Immediate, resolve_strategy
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.serialization import serialize_strategy
from roomkit.delivery.worker import execute_delivery, reject_invalid_delivery
from roomkit.models.delivery import DeliveryError, DeliveryOutcome

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

    # Cross-mixin method — attribute annotation avoids MRO shadowing
    get_room: Any  # RoomLifecycleMixin

    async def deliver(
        self,
        room_id: str,
        content: str,
        *,
        channel_id: str | None = None,
        strategy: DeliveryStrategy | str | None = None,
        metadata: dict[str, Any] | None = None,
        addressed_to: list[str] | None = None,
        idempotency_key: str | None = None,
        session_id: str | None = None,
        chain_depth: int = 0,
        instruction: bool = False,
        organization_id: str | None = None,
    ) -> DeliveryOutcome:
        """Bring *content* into a room from outside it, for the room's agents.

        A background task's result, an external event, a scheduled nudge,
        what another room produced: *content* reaches the room's agents, who
        read it and act on it, at a moment the target channel allows (voice
        playing, user speaking, idle).

        It is not a way to text the room's correspondent. On a text transport
        (SMS, email, WhatsApp, WebSocket...) *content* enters the room as a
        message received on that channel from the framework's own sender
        (``system``): the correspondent receives the agents' answer, never
        *content* itself, and with no agent in the room nothing is sent. A
        message for the correspondent is the host's own words: write it with
        :meth:`post` (RFC §10.5).

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
            addressed_to: Intelligence channel ids asked to act. None keeps
                routing; [] solicits no agent. This does not change visibility.
            idempotency_key: Publication/injection key retained by the
                ConversationStore. Scoped to the room for text and to the
                room/channel/session for realtime voice. Uncertain voice
                submissions return unknown and are never automatically repeated.
            session_id: Exact realtime session on the selected channel. An ended
                or replaced session is unavailable; no substitute is selected.
                A keyed receipt can still be replayed after its session ends.
            chain_depth: The chain the delivered content continues (RFC §8.3).
                0 opens one, outside the room's work. Called while a turn or
                a tool call runs, the depth is at least the turn's response
                or the call's, and from an ASYNC hook's body at least one
                deeper than the hook's event, whatever is passed: a queued
                item carries it to the worker. Content delivered for a turn
                once it is over, such as a background task's result, passes
                that turn's depth, so a cycle of task, result and task again
                ends at ``max_chain_depth`` (§23.3).
            instruction: Deliver *content* as the application's direction to
                an agent, never as a participant's words (RFC §22.1): an
                ``INSTRUCTION`` event through the text pipeline, which needs
                *addressed_to* and no *idempotency_key* (§10.1.1), or the
                ``system`` intent in a realtime session (§12.4). The strategy
                and the delivery hooks apply unchanged.
            organization_id: The organization the caller acts for (RFC
                §17.2): the room is read with it before any hook runs or
                anything is queued. Unset, no room is read for the scope.

        Returns:
            Queue acceptance or the actual execution outcome. ``sent`` does not
            imply turn completion. Inspect ``inbound`` for text turn details.

        Raises:
            RoomNotFoundError: *organization_id* is set and the room is
                missing or another organization's; nothing is delivered,
                queued or fired.
        """
        return await self._deliver(
            room_id,
            content,
            channel_id=channel_id,
            strategy=strategy,
            metadata=metadata,
            addressed_to=addressed_to,
            idempotency_key=idempotency_key,
            session_id=session_id,
            chain_depth=chain_depth,
            instruction=instruction,
            organization_id=organization_id,
        )

    async def _deliver(
        self,
        room_id: str,
        content: str,
        *,
        channel_id: str | None = None,
        strategy: DeliveryStrategy | str | None = None,
        metadata: dict[str, Any] | None = None,
        addressed_to: list[str] | None = None,
        idempotency_key: str | None = None,
        session_id: str | None = None,
        chain_depth: int = 0,
        instruction: bool = False,
        organization_id: str | None = None,
        runtime_record: str | None = None,
    ) -> DeliveryOutcome:
        """:meth:`deliver`, and for the runtime's own text (a background
        result handed back) its *runtime_record*, which the delivered event
        carries as its provenance (RFC §6.4) and no caller of
        :meth:`deliver` sets."""
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        resolved = resolve_strategy(strategy) or self._delivery_strategy
        if resolved is None:
            resolved = Immediate()
        # Made within a turn or an ASYNC hook's body, the content continues
        # at least that work's chain (RFC §8.3), when it runs here as when a
        # worker runs it later, outside the work. A negative depth is left to
        # the item's validation.
        floor = chain_floor()
        if 0 <= chain_depth < floor:
            chain_depth = floor

        item = DeliveryItem(
            room_id=room_id,
            content=content,
            channel_id=channel_id,
            strategy=serialize_strategy(resolved),
            metadata=metadata or {},
            addressed_to=addressed_to,
            idempotency_key=idempotency_key,
            session_id=session_id,
            chain_depth=chain_depth,
            instruction=instruction,
            runtime_record=runtime_record,
        )
        if self._delivery_backend is not None:
            refusal = await reject_invalid_delivery(
                self,  # ty: ignore[invalid-argument-type]
                item,
                hook_engine=self._hook_engine,
            )
            if refusal is not None:
                return refusal
            try:
                await self._delivery_backend.enqueue(item)
            except Exception as exc:
                return DeliveryOutcome(
                    status="failed",
                    reason="enqueue_failed",
                    delivery_item_id=item.id,
                    error=DeliveryError(code=type(exc).__name__, message=str(exc)),
                )
            return DeliveryOutcome(status="queued", delivery_item_id=item.id)
        return await execute_delivery(
            self,  # ty: ignore[invalid-argument-type]
            item,
            strategy=resolved,
            hook_engine=self._hook_engine,
        )
