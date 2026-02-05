"""Conversation reference storage for Teams proactive messaging."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConversationReferenceStore(ABC):
    """Abstract store for Bot Framework ConversationReference dicts."""

    @abstractmethod
    async def save(self, conversation_id: str, reference: dict[str, Any]) -> None:
        """Persist a conversation reference."""
        ...

    @abstractmethod
    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve a conversation reference by ID."""
        ...

    @abstractmethod
    async def delete(self, conversation_id: str) -> None:
        """Remove a conversation reference."""
        ...

    @abstractmethod
    async def list_all(self) -> dict[str, dict[str, Any]]:
        """Return all stored conversation references."""
        ...


class InMemoryConversationReferenceStore(ConversationReferenceStore):
    """Dict-backed in-memory conversation reference store."""

    def __init__(self) -> None:
        self._refs: dict[str, dict[str, Any]] = {}

    async def save(self, conversation_id: str, reference: dict[str, Any]) -> None:
        self._refs[conversation_id] = reference

    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        return self._refs.get(conversation_id)

    async def delete(self, conversation_id: str) -> None:
        self._refs.pop(conversation_id, None)

    async def list_all(self) -> dict[str, dict[str, Any]]:
        return dict(self._refs)
