"""JSONB metadata works with default asyncpg pools and shared store codecs."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from roomkit.knowledge.postgres import PostgresKnowledgeSource


@pytest.mark.parametrize("metadata", ['{"category":"faq"}', {"category": "faq"}])
async def test_search_decodes_jsonb(metadata: str | dict[str, str]) -> None:
    conn, pool = AsyncMock(), MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    conn.fetch.return_value = [
        {"content": "answer", "score": 1.0, "source": "faq", "metadata": metadata}
    ]
    result = await PostgresKnowledgeSource(pool=pool).search("question")
    assert result[0].metadata == {"category": "faq"}


@pytest.mark.skipif(not os.environ.get("POSTGRES_DSN"), reason="POSTGRES_DSN not set")
@pytest.mark.parametrize("pool_kind", ["owned", "standard", "store"])
async def test_index_search_round_trip_on_real_pools(pool_kind: str) -> None:
    import asyncpg

    from roomkit.store.postgres import PostgresStore

    dsn = os.environ["POSTGRES_DSN"]
    pool = None
    store = None
    if pool_kind == "standard":
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    elif pool_kind == "store":
        store = PostgresStore(dsn=dsn)
        await store.init(min_size=1, max_size=2)
        pool = store._pool
    source = PostgresKnowledgeSource(dsn=dsn, pool=pool)
    await source.init(min_size=1, max_size=2)
    metadata = {"id": uuid4().hex, "room_id": uuid4().hex, "nested": {"tags": ["faq", "café"]}}
    try:
        await source.index("Refunds available within thirty days", metadata)
        [result] = await source.search("refund", room_id=metadata["room_id"])
        assert result.metadata == metadata
        # Verify storage is a JSON object, not a double-encoded JSON string.
        async with source._pool.acquire() as conn:
            kind = await conn.fetchval(
                "SELECT jsonb_typeof(metadata) FROM knowledge_documents WHERE id=$1",
                metadata["id"],
            )
            assert kind == "object"
    finally:
        async with source._pool.acquire() as conn:
            await conn.execute("DELETE FROM knowledge_documents WHERE id=$1", metadata["id"])
        await source.close()
        if store is not None:
            await store.close()
        elif pool is not None:
            await pool.close()


async def test_index_and_search_use_the_configured_language() -> None:
    conn, pool = AsyncMock(), MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    conn.fetch.return_value = []
    source = PostgresKnowledgeSource(pool=pool, language="french")

    await source.index("Les chevaux mangent", {"id": "doc-1"})
    await source.search("cheval", room_id="room-1", limit=3)

    assert conn.execute.await_args.args[-1] == "french"
    assert "$6::regconfig" in conn.execute.await_args.args[0]
    assert conn.fetch.await_args.args[1:] == ("cheval", "french", 3, "room-1")
    assert "plainto_tsquery($2::regconfig, $1)" in conn.fetch.await_args.args[0]


@pytest.mark.skipif(not os.environ.get("POSTGRES_DSN"), reason="POSTGRES_DSN not set")
@pytest.mark.parametrize(("language", "found"), [("french", True), ("english", False)])
async def test_the_configured_language_stems_on_a_real_database(
    language: str, found: bool
) -> None:
    source = PostgresKnowledgeSource(dsn=os.environ["POSTGRES_DSN"], language=language)
    await source.init(min_size=1, max_size=2)
    metadata = {"id": uuid4().hex, "room_id": uuid4().hex}
    try:
        await source.index("Les chevaux mangent dans le pré", metadata)
        results = await source.search("cheval", room_id=metadata["room_id"])
        assert bool(results) is found
    finally:
        async with source._pool.acquire() as conn:
            await conn.execute("DELETE FROM knowledge_documents WHERE id=$1", metadata["id"])
        await source.close()


@pytest.mark.skipif(not os.environ.get("POSTGRES_DSN"), reason="POSTGRES_DSN not set")
async def test_init_refuses_a_language_the_database_lacks() -> None:
    source = PostgresKnowledgeSource(dsn=os.environ["POSTGRES_DSN"], language="klingon")
    with pytest.raises(ValueError, match="klingon"):
        await source.init(min_size=1, max_size=2)
    # The pool it created is closed with the refusal.
    assert source._pool is None


@pytest.mark.skipif(not os.environ.get("POSTGRES_DSN"), reason="POSTGRES_DSN not set")
async def test_a_cancelled_index_writes_nothing() -> None:
    """An index its caller cuts while a lock holds the document is cancelled on the
    server, as asyncpg cancels it: once the lock goes, the document is as it was
    (RMK-1398). Named as its Rust twin in roomkit-rs
    crates/roomkit-postgres/tests/knowledge.rs."""
    import asyncpg

    dsn = os.environ["POSTGRES_DSN"]
    source = PostgresKnowledgeSource(dsn=dsn)
    await source.init(min_size=1, max_size=1)
    holder = await asyncpg.connect(dsn)
    doc_id = uuid4().hex
    try:
        await source.index("Refund before", {"id": doc_id})
        async with holder.transaction():
            await holder.execute(
                "SELECT 1 FROM knowledge_documents WHERE id = $1 FOR UPDATE", doc_id
            )
            index = asyncio.create_task(source.index("Refund cancelled", {"id": doc_id}))
            # Cut once the server shows the write waiting: one cut before proves nothing.
            await _until_blocked_by(holder)
            index.cancel()
            with pytest.raises(asyncio.CancelledError):
                await index
        content = await holder.fetchval(
            "SELECT content FROM knowledge_documents WHERE id = $1", doc_id
        )
        assert content == "Refund before"
    finally:
        await holder.execute("DELETE FROM knowledge_documents WHERE id = $1", doc_id)
        await holder.close()
        await source.close()


async def _until_blocked_by(holder: Any) -> None:
    """Return once a session waits on a lock *holder*'s transaction holds."""
    deadline = asyncio.get_running_loop().time() + 2
    blocked = "SELECT count(*) FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid))"
    async with asyncio.timeout_at(deadline):
        while not await holder.fetchval(blocked, holder.get_server_pid()):
            await asyncio.sleep(0.005)
