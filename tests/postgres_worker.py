"""A PostgreSQL database per xdist worker.

The PostgreSQL tests empty the tables they use (``TRUNCATE``) and read what
the whole database holds: the latest room, a sender's identities, an advisory
lock. Two of them running at once on one database fail each other (RMK-927).
Under xdist, each worker therefore gets a database of its own, created next to
the one ``POSTGRES_DSN`` names when the worker starts and dropped when it
ends; ``POSTGRES_DSN`` then names it for the worker's tests and for the
processes they start. Advisory locks are per database too. A run without
xdist keeps ``POSTGRES_DSN`` as it is.
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit

import pytest

try:
    import asyncpg
except ImportError:  # the PostgreSQL tests only run with the postgres extra
    asyncpg = None  # type: ignore[assignment]

# The server's DSN and the database made for this worker, to drop at its end.
_made: list[tuple[str, str]] = []


def install() -> None:
    """Point ``POSTGRES_DSN`` at a database of this worker's own, made empty."""
    dsn = os.environ.get("POSTGRES_DSN")
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not dsn or not worker or asyncpg is None:
        return
    url = urlsplit(dsn)
    shared = url.path.lstrip("/")
    if not shared:
        raise pytest.UsageError("POSTGRES_DSN names no database to make the workers' beside")
    database = f"{shared}_{worker}"
    try:
        asyncio.run(_execute(dsn, f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        asyncio.run(_execute(dsn, f'CREATE DATABASE "{database}"'))
    except (asyncpg.PostgresError, OSError) as exc:
        raise pytest.UsageError(
            f"Parallel PostgreSQL tests make a database per worker, and that failed "
            f"({type(exc).__name__}: {exc}): check the server POSTGRES_DSN names and that "
            "its role may create databases, or run without xdist"
        ) from exc
    _made.append((dsn, database))
    os.environ["POSTGRES_DSN"] = urlunsplit(url._replace(path=f"/{database}"))


def clean_up() -> None:
    """Drop the database this worker made, whatever its tests left connected."""
    while _made:
        dsn, database = _made.pop()
        asyncio.run(_execute(dsn, f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))


async def _execute(dsn: str, statement: str) -> None:
    connection = await asyncpg.connect(dsn)
    try:
        await connection.execute(statement)
    finally:
        await connection.close()
