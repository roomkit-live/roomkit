"""A proxy to Redis that silences the connections already open: a NAT or a
load balancer that forgot them (half-open). Set REDIS_URL to use it."""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse


def redis_address() -> tuple[str, int]:
    """The host and port REDIS_URL names."""
    url = urlparse(os.environ["REDIS_URL"])
    return url.hostname or "localhost", url.port or 6379


def redis_db() -> int:
    """The database REDIS_URL names: a client of a proxy reaches the keys the
    rest of the test reads there."""
    return int(urlparse(os.environ["REDIS_URL"]).path.strip("/") or 0)


class Silencing:
    """A proxy to Redis that, once told, drops every byte of the connections
    already open and keeps them open: a NAT or a load balancer that forgot
    them. A connection opened after reaches Redis, until it refuses them
    (RMK-859)."""

    def __init__(self) -> None:
        self._accepted = 0
        self._silenced_below: int | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self._server: asyncio.Server | None = None

    async def url(self) -> str:
        self._server = await asyncio.start_server(self._relay, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"redis://127.0.0.1:{port}{urlparse(os.environ['REDIS_URL']).path}"

    async def unix_url(self, path: str) -> str:
        self._server = await asyncio.start_unix_server(self._relay, path)
        return f"unix://{path}?db={urlparse(os.environ['REDIS_URL']).path.strip('/') or 0}"

    def silence(self) -> None:
        self._silenced_below = self._accepted

    def refuse(self) -> None:
        """Refuse every connection from now: Redis stopped."""
        assert self._server is not None
        self._server.close()

    async def close(self) -> None:
        assert self._server is not None
        self._server.close()
        for writer in self._writers:
            writer.close()

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        number = self._accepted
        self._accepted += 1
        server_reader, server_writer = await asyncio.open_connection(*redis_address())
        self._writers += [writer, server_writer]

        async def pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
            while data := await source.read(65536):
                if self._silenced_below is None or number >= self._silenced_below:
                    sink.write(data)
                    await sink.drain()

        await asyncio.gather(
            pipe(reader, server_writer), pipe(server_reader, writer), return_exceptions=True
        )
