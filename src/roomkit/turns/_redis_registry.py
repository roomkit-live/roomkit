"""What the processes of the Redis turn backend share, kept by Redis (RFC
§10.6.7, §10.6.8): their beats, each room's holder and the presence last
published.

A beat scores, on the Redis server's clock, each executor the process serves
(as ``<node>:<executor id>``: the same id restarted in another process is
another executor), writes what each declares again (so a lost declaration,
or a room hook defined later, is declared at the next beat), withdraws the
executors that closed, writes the set of turns the process still runs, and
scores the process as a room owner when it is one. Each owner asks only
about the turns it reads, so what every process reads of the others costs
the same however many turns they run.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from roomkit.turns._redis_clients import cut_as_timeout
from roomkit.turns.broker import Beat, Beats, BeatWriter, RegistryView, TurnRegistry

if TYPE_CHECKING:
    from roomkit.turns._redis_clients import RedisClients
    from roomkit.turns._redis_keys import Keys

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
# How long a room's holder is kept once no turn renews it.
_HOLDER_TTL_S = 7 * 24 * 3600


class RedisTurnRegistry(TurnRegistry):
    """The beats, the holders and the presences of one backend's processes,
    on Redis.

    Args:
        clients: The backend's connections: the registry's calls go through
            the pool, its beat writer's through a connection of its own.
        keys: The keys of the backend's processes.
        ttl_ms: How long a beat's keys outlive it.
    """

    def __init__(self, clients: RedisClients, keys: Keys, ttl_ms: int) -> None:
        client = clients.pooled
        self._client = client
        self._connect_beat = clients.beat
        self._cut = clients.timeout_errors
        self._keys = keys
        self._ttl_ms = ttl_ms
        self._view = client.register_script(_VIEW)
        self._beat = client.register_script(BEAT_SCRIPT)
        self._claim = client.register_script(CLAIM_PRESENCE_SCRIPT)
        self._connected = False

    async def connect(self) -> None:
        """The pool connects with its first call: nothing to reach yet."""
        self._connected = True

    def connected(self) -> bool:
        return self._connected

    async def beat(self, beat: Beat) -> None:
        with cut_as_timeout(self._cut):
            await self._beat(keys=self._keys.beat(beat.node), args=[self._ttl_ms, _payload(beat)])

    async def beater(self) -> BeatWriter:
        return _RedisBeatWriter(self._connect_beat, self._keys, self._ttl_ms, self._cut)

    async def view(self, forget_after_ms: int) -> RegistryView:
        keys = self._keys
        with cut_as_timeout(self._cut):
            raw = await self._view(
                keys=[keys.alive, keys.declarations, keys.owners], args=[forget_after_ms]
            )
        return _view(raw)

    async def forget_owner(self, node: str) -> None:
        with cut_as_timeout(self._cut):
            await self._client.zrem(self._keys.owners, node)

    async def beats(
        self, members: Sequence[str], running: Sequence[tuple[str, Sequence[str]]]
    ) -> Beats:
        """One round trip, and one transaction when *members* are asked: a
        node's last beat and the turns it runs are then read together."""
        asked = [(node, list(turns)) for node, turns in running if turns]
        if not members and not asked:
            return Beats(last=[], running=[[] for _ in running])
        pipe = self._client.pipeline(transaction=bool(members))
        if members:
            pipe.zmscore(self._keys.alive, list(members))
        for node, turns in asked:
            pipe.smismember(self._keys.running(node), turns)
        with cut_as_timeout(self._cut):
            found = await pipe.execute()
        scores = found.pop(0) if members else []
        flags = iter(found)
        return Beats(
            last=[None if score is None else int(score) for score in scores],
            running=[
                [bool(runs) for runs in next(flags)] if turns else [] for _, turns in running
            ],
        )

    async def holder(self, channel: str, room: str) -> str | None:
        with cut_as_timeout(self._cut):
            stored = await self._client.get(self._keys.holder(channel, room))
        return _text(stored) if stored is not None else None

    async def hold(self, channel: str, room: str, member: str) -> None:
        with cut_as_timeout(self._cut):
            await self._client.set(self._keys.holder(channel, room), member, ex=_HOLDER_TTL_S)

    async def forget_holder(self, channel: str, room: str) -> None:
        with cut_as_timeout(self._cut):
            await self._client.delete(self._keys.holder(channel, room))

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        key = self._keys.agent_presence(channel_id, room_id)
        state = "" if online is None else "online" if online else "offline"
        with cut_as_timeout(self._cut):
            return bool(await self._claim(keys=[key], args=[state, _PRESENCE_TTL_MS]))


class _RedisBeatWriter(BeatWriter):
    """Beats on a connection of its own, a synchronous one: the thread that
    beats runs nothing else meanwhile."""

    def __init__(
        self,
        connect: Callable[[], Any],
        keys: Keys,
        ttl_ms: int,
        cut: tuple[type[Exception], ...],
    ) -> None:
        self._client = connect()
        self._script = self._client.register_script(BEAT_SCRIPT)
        self._keys = keys
        self._ttl_ms = ttl_ms
        self._cut = cut

    async def beat(self, beat: Beat) -> None:
        with cut_as_timeout(self._cut):
            self._script(keys=self._keys.beat(beat.node), args=[self._ttl_ms, _payload(beat)])

    async def close(self) -> None:
        self._client.close()


def _payload(beat: Beat) -> str:
    """*beat* as the beat script reads it."""
    return json.dumps(
        {
            "node": beat.node,
            "generation": beat.generation,
            "declared": beat.declared,
            "undeclared": beat.undeclared,
            "gone": beat.gone,
            "running": beat.running,
            "owner": beat.owner,
        }
    )


def _view(raw: Any) -> RegistryView:
    """The view the view script wrote."""
    data = json.loads(raw)
    return RegistryView(
        now=int(data["now"]),
        executors={
            member: (int(beat), declared if isinstance(declared, str) else None)
            for member, (beat, declared) in _entries(data.get("executors"))
        },
        owners={node: int(beat) for node, beat in _entries(data.get("owners"))},
    )


def _entries(value: Any) -> Iterable[tuple[str, Any]]:
    """A Lua table cjson wrote: an empty one comes as ``{}`` or ``[]``."""
    return value.items() if isinstance(value, dict) else ()


def _text(value: Any) -> str:
    return value if isinstance(value, str) else bytes(value).decode()
