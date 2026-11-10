"""The channels an executor let go of (RFC §10.6.2, §10.6.3).

A release (``channel_released``) or the executor's bound lets go of a build:
the next turn builds the channel anew, and the build let go is closed once
the turns that run on it, or still may, end. A turn running on it is never
cut for this.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from roomkit.turns._executor_channels import close_quietly

if TYPE_CHECKING:
    from roomkit.channels.base import Channel


class Retiring:
    """The builds an executor let go of, each closed once its turns end."""

    def __init__(self) -> None:
        # Each build let go with the turns that run on it or still may.
        self._waiting: dict[int, tuple[Channel, set[asyncio.Task[None]]]] = {}
        self._closing: set[asyncio.Task[None]] = set()

    def retire(self, channel: Channel, holders: set[asyncio.Task[None]]) -> None:
        """Close *channel* once the turns *holders* end, at once when none."""
        if holders:
            self._waiting[id(channel)] = (channel, holders)
        else:
            self._close(channel)

    def turn_ended(self, task: asyncio.Task[None]) -> None:
        """The turn *task* ended: a build it alone still held is closed."""
        for key, (channel, holders) in list(self._waiting.items()):
            holders.discard(task)
            if not holders:
                del self._waiting[key]
                self._close(channel)

    async def close(self) -> None:
        """Close every build let go of, whatever still holds it, and wait for
        the closes running."""
        waiting, self._waiting = list(self._waiting.values()), {}
        for channel, _holders in waiting:
            self._close(channel)
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)

    def _close(self, channel: Channel) -> None:
        task = asyncio.get_running_loop().create_task(
            close_quietly(channel), name=f"roomkit-release-{channel.channel_id}"
        )
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
