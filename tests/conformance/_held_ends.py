"""Turns whose end is owed once cut (roomkit-specs ``conformance/README.md``,
a channel's ``holds_end``; ``expect.turn_ends``).

The turn backend the kit is built with hands the owner each turn of such a
channel through a link that, once the owner sent the turn's executor a
``close`` or a ``cancel`` (RFC §10.6.8), holds the turn's ``end`` at the
channel's gate: the executor applies the cut as it does any, and its end
reaches the owner only once a step opens the gate. Each turn so cut is
recorded with whether the owner, still holding the turn when the gate opened,
read its end.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from roomkit.turns.base import TurnBackend, TurnLink
from roomkit.turns.wire import Cancel, Close, Control, End, Frame, TurnRequest


class HeldEnds:
    """The channels whose turns hold their end once cut, and each turn held."""

    def __init__(
        self, channels: Iterable[Mapping[str, Any]], hold: Callable[[str], Awaitable[None]]
    ) -> None:
        self._gates = {
            channel["id"]: channel["holds_end"] for channel in channels if "holds_end" in channel
        }
        self._hold = hold
        self.ends: dict[str, list[dict[str, Any]]] = {}
        """Per channel, each of its turns the owner cut (README, ``turn_ends``)."""

    def __bool__(self) -> bool:
        return bool(self._gates)

    def wrap(self, backend: TurnBackend) -> None:
        """Has *backend* hand the owner each held channel's turn through a
        link that holds its end once cut: the backend stays the one the kit
        reads, whatever its transport."""
        submit = backend.submit

        async def submitting(request: TurnRequest) -> TurnLink:
            link = await submit(request)
            gate = self._gates.get(request.channel_id)
            return link if gate is None else _HeldEndLink(link, gate, self)

        target: Any = backend
        target.submit = submitting

    def cut(self, request: TurnRequest, how: str) -> dict[str, Any]:
        """Records *request*'s turn cut by *how*, its end not read yet."""
        entry = {"room_id": request.room_id, "cut": how, "read": False}
        self.ends.setdefault(request.channel_id, []).append(entry)
        return entry

    async def hold(self, gate: str) -> None:
        await self._hold(gate)


class _HeldEndLink(TurnLink):
    """The owner's link to one turn, its end held at *gate* once cut."""

    def __init__(self, link: TurnLink, gate: str, held: HeldEnds) -> None:
        self.request = link.request
        self._link = link
        self._gate = gate
        self._held = held
        self._entry: dict[str, Any] | None = None
        self._released = False

    @property
    def taken(self) -> bool:
        return self._link.taken

    def beat_heard(self) -> bool:
        return self._link.beat_heard()

    def release(self) -> None:
        self._released = True
        self._link.release()

    async def send(self, message: Control) -> None:
        if self._entry is None and isinstance(message, Close | Cancel):
            how = "close" if isinstance(message, Close) else "cancel"
            self._entry = self._held.cut(self.request, how)
        await self._link.send(message)

    async def receive(self) -> tuple[int, Frame]:
        seq, frame = await self._link.receive()
        if isinstance(frame, End) and self._entry is not None:
            await self._held.hold(self._gate)
            # An owner that let the turn go drops this end unread.
            self._entry["read"] = not self._released
        return seq, frame
