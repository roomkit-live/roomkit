"""The keys the Redis turn backend's processes write (RFC §10.6.7, §10.6.8)."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote


@dataclass(frozen=True, slots=True)
class Keys:
    """The keys of one backend's processes."""

    prefix: str

    @property
    def alive(self) -> str:
        return f"{self.prefix}:alive"

    @property
    def declarations(self) -> str:
        return f"{self.prefix}:executors"

    @property
    def owners(self) -> str:
        return f"{self.prefix}:owners"

    def running(self, node: str) -> str:
        return f"{self.prefix}:running:{node}"

    def agent_presence(self, channel_id: str, room_id: str) -> str:
        """The presence last published for *channel_id* in *room_id*."""
        return f"{self.prefix}:presence:{quote(room_id, safe='')}:{quote(channel_id, safe='')}"

    def holder(self, channel_id: str, room_id: str) -> str:
        """The executor holding *room_id*'s state of *channel_id*."""
        return f"{self.prefix}:holder:{quote(channel_id, safe='')}:{quote(room_id, safe='')}"

    def inbox(self, node: str) -> str:
        return f"{self.prefix}:inbox:{node}"

    def beat(self, node: str) -> list[str]:
        """The keys a beat of *node* writes, in the beat script's order."""
        return [
            self.alive,
            self.declarations,
            self.owners,
            self.inbox(node),
            self.running(node),
            f"{self.prefix}:generation:{node}",
        ]
