"""State comparisons shared by overlapping calls of one channel and room."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.channels.base import Channel


class CallState:
    """Compare a call's outcome with state included in its peers' answers.

    A peer that reports a change advances this comparison too: a call that
    restores its initial state must undo that report, even when its own
    before and after are equal. A read without a change still reports nothing.
    """

    def __init__(self, channel: Channel, room_id: str, peers: list[CallState]) -> None:
        self._channel = channel
        self._room_id = room_id
        self._before = deepcopy(channel.room_state(room_id))
        self._peers = peers

    def changed(self) -> dict[str, object] | None:
        state = self._channel.room_state(self._room_id)
        if state == self._before:
            return None
        for peer in self._peers:
            peer._before = deepcopy(state)
        return state


class CallStates:
    """Only calls in flight keep comparisons; cancellation retires one too."""

    def __init__(self) -> None:
        self._pending: dict[tuple[int, str], list[CallState]] = {}

    @contextmanager
    def track(self, channel: Channel, room_id: str) -> Generator[CallState, None, None]:
        key = (id(channel), room_id)
        peers = self._pending.setdefault(key, [])
        try:
            state = CallState(channel, room_id, peers)
            peers.append(state)
            try:
                yield state
            finally:
                peers.remove(state)
        finally:
            if not peers:
                del self._pending[key]
