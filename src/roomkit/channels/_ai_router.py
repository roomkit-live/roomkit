"""Choosing the provider of each AI turn (RFC §6.4).

``AIChannel(router=...)`` asks an :class:`AIRouter` which provider serves a
turn, instead of serving every turn on one provider. The router is asked at
the turn's start, before the turn's context is built, so the context is built
for the provider it picks (whether that model sees images, its context window,
whether it holds tools unseen). It is asked again once that provider's retries
are spent with nothing delivered, and may name another provider for the
generation. Every later round of the turn stays on the provider that served
it: a tool round reads the reasoning and the cache of the round before.

A router is plain code: it may read the machine's load, the hour, the room, a
classifier's answer, or anything else the application knows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from roomkit._media import content_text
from roomkit.providers.ai.base import AITextPart

if TYPE_CHECKING:
    from roomkit.models.event import RoomEvent
    from roomkit.providers.ai.base import AIMessage, AIProvider, ProviderError

RouteReason = Literal["turn", "retry"]
"""Why a router is asked: ``turn`` at a turn's start, ``retry`` once the
provider a generation ran on spent its retries without delivering anything."""


@dataclass(frozen=True, slots=True)
class RouteRequest:
    """What a router decides on."""

    reason: RouteReason
    text: str
    """What the turn answers, as text: the text of the event that opened it,
    each media read as its label; for a turn that answers no room event (a
    voice delegation), its last user message."""
    event: RoomEvent | None = None
    """The event the turn answers; ``None`` for a turn that answers none."""
    room_id: str | None = None
    previous: AIProvider | None = None
    """For ``retry``, the provider whose retries are spent: the one the turn
    was on. ``None`` at a turn's start."""
    failed: ProviderError | None = None
    """For ``retry``, the last error that provider raised."""


class AIRouter(ABC):
    """Picks the provider of each turn of an :class:`~roomkit.channels.ai.AIChannel`.

    The channel closes its router when it closes; a router closes the
    providers it holds.
    """

    @abstractmethod
    async def route(self, request: RouteRequest) -> AIProvider | None:
        """The provider for *request*.

        For a turn's start (``reason == "turn"``) a router must name one. For a
        ``retry``, ``None`` names no other: the last error is the turn's. A
        provider named for a retry gets one attempt; when it fails before it
        delivers anything, the turn's error is the one ``previous`` raised.
        """

    @property
    def name(self) -> str:
        """The router's label, which the channel's ``provider_name`` reads."""
        return type(self).__name__

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release what the router holds: the providers it routes to."""


class FallbackRouter(AIRouter):
    """Every turn on *primary*; once a generation's provider spent its
    retries, the other provider."""

    def __init__(self, primary: AIProvider, fallback: AIProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def primary(self) -> AIProvider:
        return self._primary

    @property
    def fallback(self) -> AIProvider:
        return self._fallback

    async def route(self, request: RouteRequest) -> AIProvider:
        if request.reason == "turn":
            return self._primary
        return self._fallback if request.previous is self._primary else self._primary

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()


def turn_request(event: RoomEvent, room_id: str | None) -> RouteRequest:
    """What the router is asked at the start of a turn that answers *event*."""
    text = content_text(event.content) or ""
    return RouteRequest("turn", text, event=event, room_id=room_id or event.room_id)


def messages_request(messages: list[AIMessage], room_id: str | None) -> RouteRequest:
    """What the router is asked at the start of a turn that answers no room
    event, but its last user message (a voice delegation)."""
    last = next((m for m in reversed(messages) if m.role == "user"), None)
    return RouteRequest("turn", _message_text(last) if last else "", room_id=room_id)


def _message_text(message: AIMessage) -> str:
    """*message*'s text parts, joined."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(p.text for p in message.content if isinstance(p, AITextPart))
