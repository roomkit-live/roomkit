"""The run's kit killed as the owner of its turns (roomkit-specs
``conformance/README.md``, the ``owner.kill`` call; ``expect.killed_owner_frames``).

The turn backend the kit is built with hands the owner each turn through a
link that, once a step killed the owner, carries nothing either way: the
executors hear nothing more from the kit, and what they send it is never
read, only its type kept. A killed process releases nothing either, so the
executor of a turn learns of the death from its owner's silence alone (RFC
§10.6.8). The kit itself runs on: what it does of its turns afterwards, as
its own liveness checks fail them, reaches no executor and is not observed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns.base import TurnBackend, TurnLink
from roomkit.turns.wire import Control, Frame, TurnRequest


class OwnerKill:
    """Whether the run's kit was killed as its turns' owner, and what its
    turns' executors sent it since."""

    def __init__(self) -> None:
        self.killed = False
        self.frames: set[str] = set()
        """The type of each frame its turns' executors sent once it was killed."""
        self._drains: set[asyncio.Task[None]] = set()

    def wrap(self, backend: TurnBackend) -> None:
        """Has *backend* hand the owner each turn through a link a kill cuts:
        the backend stays the one the kit reads, whatever its transport."""
        submit = backend.submit

        async def submitting(request: TurnRequest) -> TurnLink:
            return _OwnerLink(await submit(request), self)

        target: Any = backend
        target.submit = submitting

    def kill(self) -> None:
        self.killed = True

    def drain(self, link: TurnLink) -> None:
        """Keeps reading what *link*'s executor sends the dead owner, once the
        owner stopped reading it."""
        task = asyncio.get_running_loop().create_task(self._read_on(link))
        self._drains.add(task)
        task.add_done_callback(self._drains.discard)

    async def _read_on(self, link: TurnLink) -> None:
        while True:
            try:
                _, frame = await link.receive()
            except TurnLinkError:
                return
            self.frames.add(frame.type)

    async def close(self) -> None:
        """Stops reading for the dead owner: the run is over."""
        for task in list(self._drains):
            task.cancel()
        await asyncio.gather(*self._drains, return_exceptions=True)


class _OwnerLink(TurnLink):
    """The owner's link to one turn, which carries nothing once it was killed."""

    def __init__(self, link: TurnLink, owner: OwnerKill) -> None:
        self.request = link.request
        self._link = link
        self._owner = owner

    @property
    def taken(self) -> bool:
        return self._link.taken

    def beat_heard(self) -> bool:
        return not self._owner.killed and self._link.beat_heard()

    def release(self) -> None:
        if not self._owner.killed:
            self._link.release()

    async def send(self, message: Control) -> None:
        if not self._owner.killed:
            await self._link.send(message)

    async def receive(self) -> tuple[int, Frame]:
        """The next frame while the owner lives; once killed, it reads on for
        the dead owner and hands it nothing, until the owner stops reading."""
        while True:
            try:
                seq, frame = await self._link.receive()
            except asyncio.CancelledError:
                if self._owner.killed:
                    self._owner.drain(self._link)
                raise
            except TurnLinkError:
                if not self._owner.killed:
                    raise
                await asyncio.Event().wait()
                raise
            if not self._owner.killed:
                return seq, frame
            self._owner.frames.add(frame.type)
