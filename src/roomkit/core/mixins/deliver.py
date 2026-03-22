"""DeliverMixin — framework-level content delivery."""

from __future__ import annotations

from typing import Any

from roomkit.core.delivery import DeliveryContext, DeliveryStrategy, resolve_strategy
from roomkit.core.mixins.helpers import HelpersMixin


class DeliverMixin(HelpersMixin):
    """Adds ``deliver()`` to RoomKit for proactive content delivery."""

    _delivery_strategy: DeliveryStrategy | None

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
            # No strategy — use Immediate as fallback
            from roomkit.core.delivery import Immediate

            resolved = Immediate()

        ctx = DeliveryContext(
            kit=self,  # type: ignore[arg-type]
            room_id=room_id,
            content=content,
            channel_id=channel_id,
            metadata=metadata,
        )
        await resolved.deliver(ctx)
