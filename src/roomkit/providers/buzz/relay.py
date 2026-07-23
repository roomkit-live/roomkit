"""Buzz relay provider backed by a shared BuzzClient source.

Outbound sends go over the HTTP bridge (``buzzkit.BuzzClient.send_message``),
reusing the connection and identity owned by the paired
:class:`~roomkit.sources.buzz.BuzzRelaySource`. This module never imports
``buzzkit`` at load time — the client is owned by the source.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.buzz.base import BuzzRelayProvider
from roomkit.providers.utils import extract_event_text

if TYPE_CHECKING:
    from roomkit.sources.buzz import BuzzRelaySource

logger = logging.getLogger("roomkit.providers.buzz")


class BuzzProvider(BuzzRelayProvider):
    """Outbound Buzz delivery via a shared :class:`BuzzRelaySource` client.

    The provider delegates every send to the ``buzzkit.BuzzClient`` owned by the
    paired source. It does **not** manage the connection lifecycle — that stays
    with the source. Sends use the HTTP bridge, so they succeed even when the
    inbound WebSocket is mid-reconnect.
    """

    def __init__(self, source: BuzzRelaySource) -> None:
        self._source = source

    @property
    def name(self) -> str:
        return "buzz"

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Post ``event`` to Buzz channel ``to`` (a relay channel UUID)."""
        client: Any = self._source.client
        if client is None:
            return ProviderResult(success=False, error="buzz_not_ready")
        text = extract_event_text(event)
        if not text:
            return ProviderResult(success=False, error="empty_message")
        try:
            result = await client.send_message(to, text)
        except Exception as exc:
            logger.warning("Buzz send failed: %s", exc)
            return ProviderResult(success=False, error=str(exc))
        return ProviderResult(
            success=bool(result.get("accepted", False)),
            provider_message_id=result.get("event_id"),
        )
