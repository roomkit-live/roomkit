"""PostgreSQL full-text search knowledge source.

Uses ``tsvector`` for relevance-ranked retrieval with no external
dependencies beyond ``asyncpg`` (already required by ``PostgresStore``).

Usage::

    from roomkit.knowledge.postgres import PostgresKnowledgeSource

    source = PostgresKnowledgeSource(dsn="postgresql://localhost/mydb")
    await source.init()

    await source.index("Refunds are available within 30 days.", metadata={"category": "faq"})

    results = await source.search("how do I get a refund?")
    # [KnowledgeResult(content="Refunds are available...", score=0.61, source="postgres")]
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.knowledge.base import KnowledgeResult, KnowledgeSource

logger = logging.getLogger("roomkit.knowledge.postgres")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    room_id TEXT,
    source TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}',
    tsv tsvector,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_knowledge_tsv ON knowledge_documents USING GIN(tsv);
CREATE INDEX IF NOT EXISTS idx_knowledge_room ON knowledge_documents(room_id);
"""

_INSERT = """
INSERT INTO knowledge_documents (id, content, room_id, source, metadata, tsv)
VALUES ($1, $2, $3, $4, $5::jsonb, to_tsvector('english', $2))
ON CONFLICT (id) DO UPDATE
    SET content = EXCLUDED.content,
        metadata = EXCLUDED.metadata,
        tsv = to_tsvector('english', EXCLUDED.content);
"""

_SEARCH = """
SELECT content, source, metadata,
       ts_rank_cd(tsv, query) AS score
FROM knowledge_documents, plainto_tsquery('english', $1) query
WHERE tsv @@ query {room_filter}
ORDER BY score DESC
LIMIT $2;
"""

_SEARCH_ROOM = _SEARCH.format(room_filter="AND room_id = $3")
_SEARCH_ALL = _SEARCH.format(room_filter="")


class PostgresKnowledgeSource(KnowledgeSource):
    """Full-text search knowledge source backed by PostgreSQL tsvector.

    Stores documents with automatic tsvector indexing and provides
    relevance-ranked retrieval via ``ts_rank_cd``.  Supports room-scoped
    queries and arbitrary metadata.

    Parameters:
        dsn: PostgreSQL connection string.
        pool: Optional pre-created asyncpg pool (shared with PostgresStore).
        source_name: Human-readable source identifier included in results.
        language: PostgreSQL text search configuration (default ``english``).
    """

    def __init__(
        self,
        dsn: str | None = None,
        pool: Any = None,
        *,
        source_name: str = "postgres",
        language: str = "english",
    ) -> None:
        try:
            import asyncpg as _asyncpg
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PostgresKnowledgeSource. "
                "Install it with: pip install roomkit[postgres]"
            ) from exc
        self._asyncpg = _asyncpg
        self._dsn = dsn
        self._pool = pool
        self._owns_pool = pool is None
        self._source_name = source_name
        self._language = language

    _acquire_timeout: float = 5.0

    @property
    def name(self) -> str:
        return f"PostgresKnowledgeSource({self._source_name})"

    async def init(self, min_size: int = 2, max_size: int = 10) -> None:
        """Create the connection pool (if needed) and ensure schema exists."""
        if self._pool is None:
            self._pool = await self._asyncpg.create_pool(
                self._dsn, min_size=min_size, max_size=max_size
            )
        async with self._acquire() as conn:
            await conn.execute(_SCHEMA)

    async def search(
        self,
        query: str,
        *,
        room_id: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeResult]:
        """Search indexed documents using PostgreSQL full-text search."""
        pool = self._ensure_pool()
        async with pool.acquire(timeout=self._acquire_timeout) as conn:
            if room_id:
                rows = await conn.fetch(_SEARCH_ROOM, query, limit, room_id)
            else:
                rows = await conn.fetch(_SEARCH_ALL, query, limit)
        return [
            KnowledgeResult(
                content=row["content"],
                score=float(row["score"]),
                source=row["source"] or self._source_name,
                metadata=dict(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    async def index(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a document for full-text search."""
        import json
        from uuid import uuid4

        meta = metadata or {}
        doc_id = meta.get("event_id") or meta.get("id") or uuid4().hex
        room_id = meta.get("room_id")
        source = meta.get("source", self._source_name)

        pool = self._ensure_pool()
        async with pool.acquire(timeout=self._acquire_timeout) as conn:
            await conn.execute(_INSERT, doc_id, content, room_id, source, json.dumps(meta))

    async def close(self) -> None:
        """Release the connection pool if we own it."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None

    def _ensure_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgresKnowledgeSource.init() must be called before use")
        return self._pool

    def _acquire(self) -> Any:
        return self._ensure_pool().acquire(timeout=self._acquire_timeout)
