"""JSONB metadata works with default asyncpg pools and shared store codecs."""

from __future__ import annotations

import os
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
