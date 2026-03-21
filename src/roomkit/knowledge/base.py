"""Abstract base class for knowledge sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeResult:
    """A single result from a knowledge source search.

    Attributes:
        content: The retrieved text content.
        score: Relevance score (higher is better). Used for ranking
            and deduplication when multiple sources return results.
        source: Human-readable source identifier (e.g. "faq-db",
            "product-docs").
        metadata: Arbitrary metadata (document ID, chunk index, etc.).
    """

    content: str
    score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeSource(ABC):
    """Pluggable knowledge retrieval backend.

    Implement this ABC to provide external knowledge for AI context
    enrichment.  Backends can be vector stores, search engines, document
    indexes, SQL databases, or any system that can answer relevance
    queries.

    Lifecycle methods ``index`` and ``close`` are concrete no-ops so
    that read-only sources only need to override ``search``.
    """

    @property
    def name(self) -> str:
        """Human-readable source name."""
        return type(self).__name__

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        room_id: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeResult]:
        """Search for relevant knowledge.

        Args:
            query: The search query text.
            room_id: Optional room scope for filtering results.
            limit: Maximum number of results to return.

        Returns:
            A list of :class:`KnowledgeResult` ordered by relevance.
        """
        ...

    async def index(  # noqa: B027
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index new content (optional).

        Knowledge sources backed by writable stores can override this
        to ingest new content as conversation events arrive.  Called by
        :class:`~roomkit.memory.RetrievalMemory` during ``ingest()``.
        """

    async def close(self) -> None:  # noqa: B027
        """Release resources held by the source (optional)."""
