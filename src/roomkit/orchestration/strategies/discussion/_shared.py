"""The speak queue every process serving the room shares, and the lease (RFC §19.7.5 rule 16).

The queue lives with the room, in its metadata. Each change reads it and
writes it back under the room lock, a lock every process takes (Section 13.5,
``PostgresAdvisoryLockManager`` across processes). The lease names the one
process that gives the turns, until it expires. Its holder renews it while it
holds it, and another process that installed the discussion takes it once it
has expired.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

from ._config import CONFIG_KEY, DiscussionConfig
from ._queue import SpeakQueueState

STATE_KEY = "_speak_queue"
"""The room metadata key the speak queue is stored under."""

LEASE_SECONDS = 15.0
"""How long the lease lasts unless its holder renews it."""

_RENEW_BEFORE = LEASE_SECONDS * 2 / 3
_LEASE_FIELDS = frozenset({"lease_holder", "lease_expires", "version"})


class SharedQueue:
    """One room's speak queue as this process sees it: the last state read."""

    def __init__(self, kit: Any, room_id: str) -> None:
        self._kit = kit
        self._room_id = room_id
        self.me: str = kit._instance_id
        self.state = SpeakQueueState()
        self.config: DiscussionConfig | None = None
        """The configuration stored with the room when last read: gone once
        any process uninstalled the discussion."""

    async def read(self) -> SpeakQueueState:
        """The stored queue, read now, without the lock."""
        room = await self._kit.store.get_room(self._room_id)
        metadata = room.metadata if room is not None else {}
        stored = (metadata or {}).get(STATE_KEY)
        self.state = SpeakQueueState.model_validate(stored) if stored else SpeakQueueState()
        self.config = DiscussionConfig.stored(metadata)
        return self.state

    @contextlib.asynccontextmanager
    async def editing(self) -> AsyncIterator[SpeakQueueState]:
        """Change the stored queue: read under the room lock, written back when
        the block changed it, its version moved when the queue itself changed
        (a lease renewal alone does not)."""
        async with self._kit._lock_manager.locked(self._room_id):
            state = await self.read()
            before = state.model_dump(mode="json")
            yield state
            after = state.model_dump(mode="json")
            if after != before:
                if _queue_part(after) != _queue_part(before):
                    state.version += 1
                await self.write(state)

    async def write(self, state: SpeakQueueState) -> None:
        await self._kit.store.patch_room_metadata(
            self._room_id, {STATE_KEY: state.model_dump(mode="json")}
        )

    async def store_config(self, config: DiscussionConfig) -> None:
        self.config = config
        await self._kit.store.patch_room_metadata(
            self._room_id, {CONFIG_KEY: config.model_dump(mode="json")}
        )

    async def forget(self) -> None:
        """Remove the discussion from the room, for every process."""
        self.state, self.config = SpeakQueueState(), None
        await self._kit.store.patch_room_metadata(
            self._room_id, {STATE_KEY: None, CONFIG_KEY: None}
        )

    # -- The lease --

    def holds(self, state: SpeakQueueState) -> bool:
        return state.lease_holder == self.me and state.lease_expires > time.time()

    def held_elsewhere(self, state: SpeakQueueState) -> bool:
        return state.lease_holder not in (None, self.me) and state.lease_expires > time.time()

    def take(self, state: SpeakQueueState) -> bool:
        """Take or renew the lease, unless another process holds it; whether
        this process holds it now. Taking it over from a process whose lease
        expired ends that process's turn (rule 6)."""
        if self.held_elsewhere(state):
            return False
        if state.lease_holder != self.me:
            state.speaking = None
        state.lease_holder = self.me
        state.lease_expires = time.time() + LEASE_SECONDS
        return True

    def renew_due(self, state: SpeakQueueState) -> bool:
        return state.lease_holder == self.me and state.lease_expires - time.time() < _RENEW_BEFORE

    def grant(self, state: SpeakQueueState, holder: str) -> None:
        """Hand the lease to *holder* (the process holding an instruction's text)."""
        state.lease_holder = holder
        state.lease_expires = time.time() + LEASE_SECONDS

    def release(self, state: SpeakQueueState) -> None:
        if state.lease_holder == self.me:
            state.lease_holder, state.lease_expires, state.speaking = None, 0.0, None


def _queue_part(dumped: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dumped.items() if k not in _LEASE_FIELDS}
