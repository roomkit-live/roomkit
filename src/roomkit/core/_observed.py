"""The ASYNC observers of a pass's ``BEFORE_BROADCAST`` chain (RFC §9.2,
§10.1).

A pass runs the chain under its room's lock, before its commit, and its
observers start once the pass is over: its event committed and its delivery
set placed in the room's lane, whatever the store's commit took. An observer
that closes the kit then finds its event's set in the lane, running when the
lane was idle (RFC §12.10.4). A pass the kit's close cuts starts none:
nothing new starts once the close began.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent


@dataclass(slots=True)
class Observed:
    """What a pass's chain owes its observers: the event it allowed, as its
    SYNC hooks left it, in the context they were handed; nothing when it
    blocked."""

    owed: tuple[RoomEvent, RoomContext] | None = None

    def owe(self, event: RoomEvent, context: RoomContext) -> None:
        """The chain allowed *event*, read in *context*."""
        self.owed = (event, context)

    async def start(self, hooks: HookEngine, room_id: str) -> None:
        """Start what the chain owes, once: a pass whose end comes before its
        section's (a row stored ``BLOCKED`` whose injected events run their
        passes next) starts them there, as every pass does before those."""
        owed, self.owed = self.owed, None
        if owed is not None:
            event, context = owed
            await hooks.run_observers(room_id, HookTrigger.BEFORE_BROADCAST, event, context)


@contextlib.asynccontextmanager
async def observed_once_over(
    hooks: HookEngine, room_id: str, *, closing: Callable[[], bool]
) -> AsyncGenerator[Observed]:
    """Around one pass of *room_id*: what its chain owes is started as it
    ends, on every path, a refusal, a failure or a timeout included, in the
    pass's own context (the cascade's budget its section draws on); but for
    a pass cut while the kit is *closing*, which starts none (RFC §10.1 step
    13, §12.10.4)."""
    observed = Observed()
    try:
        yield observed
    except asyncio.CancelledError:
        if closing():
            observed.owed = None
        raise
    finally:
        await observed.start(hooks, room_id)
