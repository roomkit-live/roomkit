"""The Redis connections a turn backend opens (RFC §10.6.8).

Each gives up a call on a peer gone without closing (a half-open connection:
a NAT or a load balancer forgot it) within half the backend's silence bound,
its inbox's blocking read the time the server holds it more, and the beat or
the read it cut is tried again at once on a new connection. While the
heartbeat interval stays under the other half of the silence bound (the
defaults: 2 s and 10 s), that try lands inside the silence bound: a process is
never taken for lost for one dead connection. redis-py before 8.0 sets no
bound of its own; 8.0 and later bound every call at 5 s.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from typing import Any, NamedTuple

from roomkit._redis import client_options
from roomkit.turns._redis_beat import call_bound
from roomkit.turns._redis_inbox import read_block
from roomkit.turns.base import TurnTimeouts

_MAX_CONNECTIONS = 32
"""The connections one process uses at most besides its inbox's: its writes
go through one writer per destination, so few are in use at once."""

_POOL_WAIT = 20.0
"""How long a caller waits for a free connection of the pool, in seconds."""


class RedisClients(NamedTuple):
    """The backend's connections: its pool, its inbox's, its beat's factory,
    and what a call its bound cut raises there."""

    pooled: Any
    inbox: Any
    beat: Callable[[], Any]
    timeout_errors: tuple[type[Exception], ...]


def redis_clients(address: str, timeouts: TurnTimeouts, client: Any = None) -> RedisClients:
    """The connections a backend opens on *address*, each bounded by
    *timeouts* (RFC §10.6.8): its pool (*client* in its place when given),
    its inbox's blocking read (bounded the time the server holds it more),
    and the factory its beat thread connects with.

    Raises:
        ImportError: redis is not installed.
    """
    try:
        import redis
        import redis.asyncio as aioredis
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except ImportError as exc:
        raise ImportError(
            "redis is required for RedisTurnBackend. Install it with: pip install roomkit[redis]"
        ) from exc
    bound = call_bound(timeouts)
    options = client_options(call_bound=bound)
    if client is None:
        pool = aioredis.BlockingConnectionPool.from_url(
            address, max_connections=_MAX_CONNECTIONS, timeout=_POOL_WAIT, **options
        )
        client = aioredis.Redis(connection_pool=pool)
    inbox_options = client_options(read_block(timeouts), call_bound=bound)
    return RedisClients(
        pooled=client,
        inbox=aioredis.from_url(address, **inbox_options),
        beat=lambda: redis.Redis.from_url(address, **options),
        timeout_errors=(TimeoutError, RedisTimeoutError),
    )


@contextlib.contextmanager
def cut_as_timeout(errors: tuple[type[Exception], ...]) -> Generator[None]:
    """A Redis call its connection's bound cut raises ``TimeoutError``, as a
    call of a broker or a registry does (redis-py's is not one): the backend
    tries it again at once (RFC §10.6.8)."""
    try:
        yield
    except TimeoutError:
        raise
    except errors as exc:
        raise TimeoutError(str(exc)) from exc
