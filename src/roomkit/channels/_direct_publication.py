"""The room audience of a text channel called without an owner-issued turn."""

from __future__ import annotations

import contextvars
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

from roomkit.core.visibility import answers_reach

if TYPE_CHECKING:
    from datetime import datetime

    from roomkit.channels.base import Channel
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.realtime.base import EphemeralEvent


@dataclass(frozen=True, slots=True)
class DirectAudience:
    room_id: str
    channel_id: str
    created_at: datetime
    public: bool

    def accepts(self, topic: str, event: EphemeralEvent) -> bool:
        return (
            self.public
            and topic == f"room:{self.room_id}"
            and event.room_id == self.room_id
            and event.channel_id in (None, self.channel_id)
            and event.room_created_at in (None, self.created_at)
        )


current_audience: contextvars.ContextVar[DirectAudience | None] = contextvars.ContextVar(
    "roomkit_direct_audience", default=None
)


@contextmanager
def _audience(audience: DirectAudience) -> Generator[None, None, None]:
    token = current_audience.set(audience)
    try:
        yield
    finally:
        current_audience.reset(token)


async def _scoped_stream[T](
    stream: AsyncIterator[T], audience: DirectAudience
) -> AsyncIterator[T]:
    """Scope each read and its cleanup. A yielded item belongs to the reader's context."""
    iterator = aiter(stream)
    try:
        while True:
            with _audience(audience):
                try:
                    item = await anext(iterator)
                except StopAsyncIteration:
                    return
            yield item
    finally:
        if (close := getattr(iterator, "aclose", None)) is not None:
            with _audience(audience):
                await close()


def scope_direct_prompt[C: Channel](
    call: Callable[[C, RoomEvent, ChannelBinding, RoomContext], Awaitable[ChannelOutput]],
) -> Callable[[C, RoomEvent, ChannelBinding, RoomContext], Awaitable[ChannelOutput]]:
    """Bind the trigger's audience during preparation, stream reads and their child work."""

    @wraps(call)
    async def scoped(
        channel: C, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        from roomkit.turns._outlet import TurnRealtime, current_turn_outlet

        if current_turn_outlet() is not None or not isinstance(
            getattr(channel, "_realtime", None), TurnRealtime
        ):
            return await call(channel, event, binding, context)
        room = context.room
        audience = DirectAudience(
            event.room_id,
            channel.channel_id,
            room.created_at,
            room.id == binding.room_id == event.room_id
            and channel.channel_id == binding.channel_id
            and answers_reach(event) == "all",
        )
        with _audience(audience):
            output = await call(channel, event, binding, context)
        if output.response_stream is not None:
            output.response_stream = _scoped_stream(output.response_stream, audience)
        return output

    return scoped
