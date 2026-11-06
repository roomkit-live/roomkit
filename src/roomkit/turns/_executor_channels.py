"""The channels one executor runs (RFC §10.6.2, §10.6.3).

The kit's own executor finds them in the kit's registry, which wires and
closes them. An executor a host builds builds each through the host's
factory once, wires it, and closes what it built when it closes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.telemetry.base import TelemetryProvider

logger = logging.getLogger("roomkit.turns")

ChannelFactory = Callable[[str], "Channel | None | Awaitable[Channel | None]"]
"""Builds, or finds, the channel a turn names by its id."""


async def _called(factory: ChannelFactory, channel_id: str) -> Channel | None:
    built: Any = factory(channel_id)
    return await built if inspect.isawaitable(built) else built


class ExecutorChannels:
    """The channels of one executor, by id.

    Args:
        factory: The host's factory, or the registry the executor reads.
        owns: Whether the executor builds its channels (a host's factory):
            each is built once, wired and closed with the executor; else
            the factory is a registry read every time, its owner wiring
            and closing what it holds.
        wire: Wires a channel the executor built.
        telemetry: The telemetry a built channel reports to.
    """

    def __init__(
        self,
        factory: ChannelFactory,
        *,
        owns: bool,
        wire: Callable[[Channel], None],
        telemetry: TelemetryProvider | None = None,
    ) -> None:
        self._factory = factory
        self._owns = owns
        self._wire = wire
        self._telemetry = telemetry
        self._built: dict[str, Channel] = {}
        self._building: dict[str, asyncio.Task[Channel | None]] = {}
        self._closed = False

    async def get(self, channel_id: str) -> Channel | None:
        """The channel for *channel_id*, ``None`` when the factory has none."""
        if not self._owns:
            return await _called(self._factory, channel_id)
        if self._closed:
            raise RuntimeError("The executor is closed: it builds no channel")
        built = self._built.get(channel_id)
        if built is not None:
            return built
        pending = self._building.get(channel_id)
        if pending is None:
            pending = asyncio.get_running_loop().create_task(self._build(channel_id))
            self._building[channel_id] = pending
            pending.add_done_callback(lambda task: self._built_or_failed(channel_id, task))
        # Shielded: a turn cut while the channel is built does not cut the
        # build another turn waits for.
        return await asyncio.shield(pending)

    def _built_or_failed(self, channel_id: str, task: asyncio.Task[Channel | None]) -> None:
        self._building.pop(channel_id, None)
        if not task.cancelled() and task.exception() is not None:
            # Read here: its waiters may all have been cut meanwhile.
            logger.debug("Building channel %s failed", channel_id, exc_info=task.exception())

    async def _build(self, channel_id: str) -> Channel | None:
        channel = await _called(self._factory, channel_id)
        if channel is None:
            return None
        try:
            self._wire(channel)
            if self._telemetry is not None:
                channel._telemetry = self._telemetry  # ty: ignore[unresolved-attribute]
                if hasattr(channel, "_propagate_telemetry"):
                    channel._propagate_telemetry()  # ty: ignore[call-non-callable]
            if self._closed:
                raise RuntimeError("The executor closed while the channel was built")
        except BaseException:
            await close_quietly(channel)
            raise
        self._built[channel_id] = channel
        return channel

    async def release(self, channel_id: str) -> Channel | None:
        """Let go of the channel built for *channel_id*, after a build of it
        still running: the next :meth:`get` builds it anew. The channel let
        go, for its caller to close; ``None`` when none was built, or when
        the channels are a registry's, whose owner replaces them."""
        if not self._owns:
            return None
        pending = self._building.get(channel_id)
        if pending is not None:
            await asyncio.wait({pending})
        return self._built.pop(channel_id, None)

    async def close(self) -> None:
        """Close what the executor built, a build still running included;
        what a registry holds is its owner's."""
        self._closed = True
        building = list(self._building.values())
        if building:
            await asyncio.gather(*building, return_exceptions=True)
        built, self._built = list(self._built.values()), {}
        for channel in built:
            await close_quietly(channel)


async def close_quietly(channel: Channel) -> None:
    try:
        await channel.close()
    except Exception:
        logger.exception("Closing channel %s failed", channel.channel_id)
