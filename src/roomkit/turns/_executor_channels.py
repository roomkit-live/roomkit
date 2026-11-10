"""The channels one executor runs (RFC §10.6.2, §10.6.3).

The kit's own executor finds them in the kit's registry, which wires and
closes them. An executor a host builds builds each through the host's
factory when a turn, a channel call or a committed event first needs it,
never for a room's attach or detach, wires it, and closes what it built when
it closes. Given a bound, it keeps that many built at most: past it, the one
used least recently that nothing runs on is let go, as a release lets it go.
A build made for a committed event alone is the first let go.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable, Generator
from typing import TYPE_CHECKING, Any

from roomkit.channels._mark_copies import compile_mark_patterns

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
            each is built when first needed, wired and closed with the
            executor; else the factory is a registry read every time, its
            owner wiring and closing what it holds, and *bound* is unused.
        wire: Wires a channel the executor built.
        telemetry: The telemetry a built channel reports to.
        bound: How many channels it keeps built at most; no bound when
            ``None``.
        in_use: Whether a turn runs on a built channel, or still may, which
            is then never let go for the bound (a channel control in
            progress counts too, :meth:`use`).
        let_go: Takes a channel let go for the bound, to close it once
            nothing runs on it.
    """

    def __init__(
        self,
        factory: ChannelFactory,
        *,
        owns: bool,
        wire: Callable[[Channel], None],
        telemetry: TelemetryProvider | None = None,
        bound: int | None = None,
        in_use: Callable[[Channel], bool] = lambda _channel: False,
        let_go: Callable[[Channel], None] = lambda _channel: None,
    ) -> None:
        self._factory = factory
        self._owns = owns
        self._wire = wire
        self._telemetry = telemetry
        self._bound = bound
        self._in_use = in_use
        self._let_go = let_go
        # Least recently used first.
        self._built: OrderedDict[str, Channel] = OrderedDict()
        self._building: dict[str, asyncio.Task[Channel | None]] = {}
        # The channel controls in progress, by channel id (:meth:`use`).
        self._using: Counter[str] = Counter()
        self._closed = False
        self._warming = _warm() if owns else None

    async def get(self, channel_id: str, *, recent: bool = True) -> Channel | None:
        """The channel for *channel_id*, built when it is not; ``None`` when
        the factory has none. With *recent* false (a committed event), the
        use moves nothing up, and a build begun for it is the first let go."""
        if not self._owns:
            return await _called(self._factory, channel_id)
        if self._closed:
            raise RuntimeError("The executor is closed: it builds no channel")
        built = self._built.get(channel_id)
        if built is None:
            pending = self._building.get(channel_id)
            if pending is None:
                pending = asyncio.get_running_loop().create_task(
                    self._build(channel_id, first_let_go=not recent)
                )
                self._building[channel_id] = pending
                pending.add_done_callback(lambda task: self._built_or_failed(channel_id, task))
            # Shielded: a turn cut while the channel is built does not cut
            # the build another turn waits for.
            built = await asyncio.shield(pending)
        if recent and built is not None and channel_id in self._built:
            self._built.move_to_end(channel_id)
        return built

    @contextlib.contextmanager
    def use(self, channel_id: str) -> Generator[None]:
        """A channel control in progress on *channel_id* (an attach, a
        detach, a call, a committed event, a drop): the bound lets go of
        nothing it runs on, and lets go of what is past it once the control
        ends."""
        self._using[channel_id] += 1
        try:
            yield
        finally:
            self._using[channel_id] -= 1
            if not self._using[channel_id]:
                del self._using[channel_id]
            self.shrink()

    async def built(self, channel_id: str) -> Channel | None:
        """The channel built for *channel_id*, after a build of it still
        running; ``None`` when none is: nothing is built for the asking (a
        room's attach or detach, RFC §10.6.3)."""
        if not self._owns:
            return await _called(self._factory, channel_id)
        pending = self._building.get(channel_id)
        if pending is not None:
            await asyncio.wait({pending})
        return self._built.get(channel_id)

    def shrink(self) -> None:
        """Let go of the channels used least recently past the bound, but
        those something runs on: the bound is then exceeded until they are
        free."""
        excess = 0 if self._bound is None else len(self._built) - self._bound
        if excess <= 0:
            return
        for channel_id, channel in list(self._built.items()):
            if excess <= 0:
                return
            if self._using[channel_id] or self._in_use(channel):
                continue
            del self._built[channel_id]
            excess -= 1
            self._let_go(channel)

    def _built_or_failed(self, channel_id: str, task: asyncio.Task[Channel | None]) -> None:
        self._building.pop(channel_id, None)
        if not task.cancelled() and task.exception() is not None:
            # Read here: its waiters may all have been cut meanwhile.
            logger.debug("Building channel %s failed", channel_id, exc_info=task.exception())

    async def _build(self, channel_id: str, *, first_let_go: bool) -> Channel | None:
        """Build *channel_id*'s channel and keep it: the most recent of the
        builds, past the bound letting go of the least recent, or, begun for
        a committed event (*first_let_go*), the least recent itself, which
        the event's end lets go (:meth:`use`)."""
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
            # What a room's attach compiled before an executor built only
            # for a turn: off the event loop, once per process (begun as the
            # executor was made, so a first turn rarely waits for it).
            await compile_mark_patterns()
        except BaseException:
            await close_quietly(channel)
            raise
        self._built[channel_id] = channel
        if first_let_go:
            self._built.move_to_end(channel_id, last=False)
        else:
            self.shrink()
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
        if self._warming is not None:
            self._warming.cancel()
            await asyncio.gather(self._warming, return_exceptions=True)
        building = list(self._building.values())
        if building:
            await asyncio.gather(*building, return_exceptions=True)
        built, self._built = list(self._built.values()), OrderedDict()
        for channel in built:
            await close_quietly(channel)


def _warm() -> asyncio.Task[None] | None:
    """Begin compiling the mark patterns a channel's turns read (about a
    second, once per process) when the executor is made in a running loop;
    ``None`` outside one, where the first build compiles them."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop.create_task(compile_mark_patterns(), name="roomkit-executor-warm")


async def close_quietly(channel: Channel) -> None:
    try:
        await channel.close()
    except Exception:
        logger.exception("Closing channel %s failed", channel.channel_id)
