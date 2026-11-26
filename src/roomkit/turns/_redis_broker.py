"""The turn backend's messages, carried by Redis (RFC §10.6).

Each process has one stream, its inbox (``<prefix>:inbox:<node>``), which
every other process writes to and it alone reads, with one blocking read for
all its turns, on a connection of its own. An entry carries a message's
fields as they are: its kind, the node that sent it, its body, the executor
or the turn it is for, and a reference tying an answer to its question. What
was read is deleted; the inbox of a process that stopped expires.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from roomkit.turns.broker import BrokerMessage, InboxEntry, TurnBroker, TurnInbox

if TYPE_CHECKING:
    from roomkit.turns._redis_clients import RedisClients
    from roomkit.turns._redis_keys import Keys

_READ_COUNT = 256


class RedisTurnBroker(TurnBroker):
    """The inboxes of one backend's processes, on Redis.

    Args:
        clients: The backend's connections: its writes go through the pool,
            its inbox's reads and what they delete through the inbox's own.
        keys: The keys of the backend's processes.
        ttl_ms: How long an inbox nobody writes to lives.
    """

    def __init__(self, clients: RedisClients, keys: Keys, ttl_ms: int) -> None:
        self._client = clients.pooled
        self._inbox = clients.inbox
        self._keys = keys
        self._ttl_ms = ttl_ms

    async def post(self, node: str, messages: Sequence[BrokerMessage]) -> None:
        """One round trip for *messages*."""
        key = self._keys.inbox(node)
        pipe = self._client.pipeline(transaction=False)
        for message in messages:
            pipe.xadd(key, _fields(message))
        pipe.pexpire(key, self._ttl_ms)
        await pipe.execute()

    async def inbox(self, node: str, block: float) -> TurnInbox:
        return _RedisInbox(self._inbox, self._keys.inbox(node), block)

    async def acknowledge(self, node: str, ids: Sequence[str]) -> None:
        await self._inbox.xdel(self._keys.inbox(node), *ids)

    async def close(self) -> None:
        """Close the inbox's connection: the broker's own, whatever pool
        the backend was given."""
        await self._inbox.aclose()


class _RedisInbox(TurnInbox):
    """One stream read from the server, *block* seconds a read at most."""

    def __init__(self, client: Any, key: str, block: float) -> None:
        self._client = client
        self._key = key
        # Redis reads a block of 0 as forever.
        self._block_ms = max(1, round(block * 1000))

    async def read(self, last: str) -> list[InboxEntry]:
        answer = await self._client.xread(
            {self._key: last or "0-0"}, count=_READ_COUNT, block=self._block_ms
        )
        return [
            InboxEntry(_text(entry_id), _message(fields))
            for _stream, entries in answer or ()
            for entry_id, fields in entries
        ]


def _fields(message: BrokerMessage) -> dict[str, str]:
    return {
        "kind": message.kind,
        "sender": message.sender,
        "body": message.body,
        "target": message.target,
        "ref": message.ref,
    }


def _message(fields: dict[Any, Any]) -> BrokerMessage | None:
    """The message of a stream entry's fields, bytes or text; none when
    they do not read as one."""
    try:
        text = {_text(key): _text(value) for key, value in fields.items()}
        return BrokerMessage(
            kind=text["kind"],
            sender=text["sender"],
            body=text["body"],
            target=text.get("target", ""),
            ref=text.get("ref", ""),
        )
    except (KeyError, UnicodeDecodeError):
        return None


def _text(value: Any) -> str:
    return value if isinstance(value, str) else bytes(value).decode()
