"""What the processes of the Redis turn backend write about themselves (RFC
§10.6.7, §10.6.8): the model, the scripts, and the reads and writes.

Each process beats from a thread of its own, on a connection of its own, so
its turns' CPU never stalls the beat. A beat scores, on the Redis server's
clock, each executor the process serves (as ``<node>:<executor id>``: the
same id restarted in another process is another executor), writes what each
declares again (so a lost declaration, or a room hook defined later, is
declared at the next beat), withdraws the executors that closed, writes the
set of turns the process still runs, and scores the process as a room owner
when it is one. A turn counts as alive while its executor beats and its
process still runs it: a turn whose task ended without its ``end`` is soon
lost, not waited for forever. Each owner asks only about the turns it reads,
so what every process reads of the others costs the same however many turns
they run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from roomkit.turns._hosting import Hosted
from roomkit.turns._organizations import RoomNeeds, serves_organization

logger = logging.getLogger("roomkit.turns")

# KEYS: alive executors, declarations, alive owners, this node's inbox, this
# node's running turns, this node's generation. ARGV: the ttl in ms, the beat
# as JSON. A beat built from an older set of executors than one written
# already writes nothing.
BEAT_SCRIPT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local beat = cjson.decode(ARGV[2])
local generation = tonumber(redis.call('GET', KEYS[6]) or '0')
if beat.generation < generation then
  return now
end
redis.call('SET', KEYS[6], beat.generation, 'PX', ARGV[1])
for member, declared in pairs(beat.declared) do
  redis.call('ZADD', KEYS[1], now, member)
  redis.call('HSET', KEYS[2], member, declared)
end
for _, member in ipairs(beat.undeclared) do
  redis.call('ZADD', KEYS[1], now, member)
  redis.call('HDEL', KEYS[2], member)
end
for _, member in ipairs(beat.gone) do
  redis.call('ZREM', KEYS[1], member)
  redis.call('HDEL', KEYS[2], member)
end
if beat.owner then
  redis.call('ZADD', KEYS[3], now, beat.node)
end
redis.call('DEL', KEYS[5])
for _, turn in ipairs(beat.running) do
  redis.call('SADD', KEYS[5], turn)
end
redis.call('PEXPIRE', KEYS[5], ARGV[1])
redis.call('PEXPIRE', KEYS[4], ARGV[1])
return now
"""

# KEYS: alive executors, declarations, alive owners. ARGV: the age past which
# a member is forgotten (ms). The view as JSON.
_VIEW = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local oldest = now - tonumber(ARGV[1])
for _, member in ipairs(redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', oldest)) do
  redis.call('ZREM', KEYS[1], member)
  redis.call('HDEL', KEYS[2], member)
end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', oldest)
local view = {now = now, executors = {}, owners = {}}
local alive = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
for i = 1, #alive, 2 do
  local member = alive[i]
  local declared = redis.call('HGET', KEYS[2], member)
  view.executors[member] = {tonumber(alive[i + 1]), declared or false}
end
local owners = redis.call('ZRANGE', KEYS[3], 0, -1, 'WITHSCORES')
for i = 1, #owners, 2 do
  view.owners[owners[i]] = tonumber(owners[i + 1])
end
return cjson.encode(view)
"""


@dataclass(frozen=True, slots=True)
class Declaration:
    """What an executor of a process declares to the owners (RFC §10.6.7):
    its channels, names and prefixes as it was given them, its room hooks,
    and the organizations it serves (``None``: the rooms without one; a
    beat that names none, an executor of a release before them, reads so)."""

    node: str
    executor_id: str
    channels: Hosted
    room_hooks: frozenset[str]
    organizations: frozenset[str] | None = None

    @property
    def member(self) -> str:
        """The executor, as one process runs it."""
        return member_of(self.node, self.executor_id)

    def takes(self, channel_id: str, needs: RoomNeeds) -> bool:
        return (
            self.channels.covers(channel_id)
            and serves_organization(self.organizations, needs.organization_id)
            and all(h in self.room_hooks for h in needs.room_hooks)
        )

    def dump(self) -> str:
        return json.dumps(
            {
                "node": self.node,
                "executor_id": self.executor_id,
                "channels": sorted(self.channels.entries),
                "room_hooks": sorted(self.room_hooks),
                "organizations": (
                    None if self.organizations is None else sorted(self.organizations)
                ),
            }
        )

    @classmethod
    def load(cls, raw: str) -> Declaration:
        data = json.loads(raw)
        return cls(
            data["node"],
            data["executor_id"],
            Hosted(frozenset(data["channels"])),
            frozenset(data["room_hooks"]),
            None if data.get("organizations") is None else frozenset(data["organizations"]),
        )


def member_of(node: str, executor_id: str) -> str:
    """An executor as one process runs it: ``<node>:<executor id>``."""
    return f"{node}:{executor_id}"


@dataclass(slots=True)
class View:
    """The processes alive as one read saw them, on the server's clock."""

    now: int = 0
    beats: dict[str, int] = field(default_factory=dict)
    declared: dict[str, Declaration] = field(default_factory=dict)
    owners: dict[str, int] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any) -> View:
        data = json.loads(raw)
        view = cls(now=int(data["now"]))
        for member, (beat, declared) in _entries(data.get("executors")):
            view.beats[member] = int(beat)
            if declared:
                try:
                    view.declared[member] = Declaration.load(declared)
                except (ValueError, KeyError, TypeError):
                    logger.warning("The declaration of executor %s is malformed", member)
        view.owners = {node: int(beat) for node, beat in _entries(data.get("owners"))}
        return view


def _entries(value: Any) -> Iterable[tuple[str, Any]]:
    """A Lua table cjson wrote: an empty one comes as ``{}`` or ``[]``."""
    return value.items() if isinstance(value, dict) else ()


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


# KEYS: the presence last published for an agent in a room. ARGV: the state
# (empty: forget it), the key's ttl in ms. 1 when the state changed (the
# caller publishes it).
CLAIM_PRESENCE_SCRIPT = """
if ARGV[1] == '' then
  redis.call('DEL', KEYS[1])
  return 0
end
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
return 1
"""

# How long a presence published is kept once no change renews it.
_PRESENCE_TTL_MS = 7 * 24 * 3600 * 1000


class Registry:
    """The processes' beats and declarations, on Redis."""

    def __init__(self, client: Any, keys: Keys) -> None:
        self._client = client
        self._keys = keys
        self._view = client.register_script(_VIEW)
        self._beat = client.register_script(BEAT_SCRIPT)
        self._claim = client.register_script(CLAIM_PRESENCE_SCRIPT)

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        """Whether *channel_id*'s presence in *room_id* changes to *online*:
        one process alone is told so (RFC §10.6.7); ``None`` forgets it."""
        key = self._keys.agent_presence(channel_id, room_id)
        state = "" if online is None else "online" if online else "offline"
        return bool(await self._claim(keys=[key], args=[state, _PRESENCE_TTL_MS]))

    async def view(self, forget_after_ms: int) -> View:
        """The processes alive, read now; members silent for
        *forget_after_ms* are forgotten."""
        keys = self._keys
        raw = await self._view(
            keys=[keys.alive, keys.declarations, keys.owners], args=[forget_after_ms]
        )
        return View.parse(raw)

    async def running(self, turns: dict[str, list[str]]) -> dict[str, frozenset[str]]:
        """Which of *turns* (by node) each node ran at its last beat."""
        nodes = [node for node, ids in turns.items() if ids]
        if not nodes:
            return {}
        pipe = self._client.pipeline(transaction=False)
        for node in nodes:
            pipe.smismember(self._keys.running(node), turns[node])
        found = await pipe.execute()
        return {
            node: frozenset(t for t, runs in zip(turns[node], flags, strict=True) if runs)
            for node, flags in zip(nodes, found, strict=True)
        }

    async def beat(self, node: str, payload: str, ttl_ms: int) -> None:
        """Beat once from this task: a member counts as alive from its first."""
        await self._beat(keys=self._keys.beat(node), args=[ttl_ms, payload])

    async def withdraw(self, members: Iterable[str], owner: str | None = None) -> None:
        """Withdraw *members*, and the owner process *owner* when given."""
        names = list(members)
        if not names and owner is None:
            return
        pipe = self._client.pipeline(transaction=False)
        if names:
            pipe.hdel(self._keys.declarations, *names)
            pipe.zrem(self._keys.alive, *names)
        if owner is not None:
            pipe.zrem(self._keys.owners, owner)
        await pipe.execute()
