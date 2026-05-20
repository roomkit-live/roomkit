"""Conversation reference storage for Teams proactive messaging."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConversationReferenceStore(ABC):
    """Abstract store for Bot Framework ConversationReference dicts.

    Beyond the bare CRUD surface, implementations should keep enough
    identity metadata alongside each reference to answer "who is at the
    other end of this conversation?" without re-hitting the roster.
    Bot Framework reliably exposes the sender's AAD object ID and Teams
    user ID on every activity; resolving an email is harder (guests,
    external users, anonymous joins) so any value worth caching is
    pre-resolved by the consumer and persisted via
    :meth:`save_with_user_info`.
    """

    @abstractmethod
    async def save(self, conversation_id: str, reference: dict[str, Any]) -> None:
        """Persist a conversation reference (no identity metadata)."""
        ...

    @abstractmethod
    async def save_with_user_info(
        self,
        conversation_id: str,
        reference: dict[str, Any],
        *,
        aad_id: str | None = None,
        email: str | None = None,
        conversation_type: str = "personal",
    ) -> None:
        """Persist a reference together with the resolved sender identity.

        Re-running with the same ``conversation_id`` updates the reference
        and merges identity fields — ``None`` values must NOT overwrite an
        already-stored non-null value (so a follow-up activity that lacks
        email doesn't blank a known address).
        """
        ...

    @abstractmethod
    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve a conversation reference by ID."""
        ...

    @abstractmethod
    async def get_email_by_sender(
        self,
        conversation_id: str,
        aad_id: str,
    ) -> str | None:
        """Look up the cached email for a specific sender in a conversation.

        Group chats hold multiple senders; the ``aad_id`` filter is
        required so a reply from Alice isn't attributed to whoever
        @mentioned the bot last (Bob).
        """
        ...

    @abstractmethod
    async def get_by_email(self, email: str) -> dict[str, Any] | None:
        """Find the most recent conversation reference for a given email.

        Useful for proactive sends where the caller only knows the
        recipient's email. Implementations return a dict with at minimum
        ``conversation_id`` and ``reference``; additional fields are
        implementation-defined.
        """
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
    """Dict-backed in-memory store. Suitable for tests and single-process bots."""

    def __init__(self) -> None:
        self._refs: dict[str, dict[str, Any]] = {}
        self._identities: dict[str, dict[str, Any]] = {}

    async def save(self, conversation_id: str, reference: dict[str, Any]) -> None:
        self._refs[conversation_id] = reference

    async def save_with_user_info(
        self,
        conversation_id: str,
        reference: dict[str, Any],
        *,
        aad_id: str | None = None,
        email: str | None = None,
        conversation_type: str = "personal",
    ) -> None:
        self._refs[conversation_id] = reference
        existing = self._identities.get(conversation_id, {})
        merged = {
            "aad_id": aad_id if aad_id is not None else existing.get("aad_id"),
            "email": email if email is not None else existing.get("email"),
            "conversation_type": conversation_type,
        }
        self._identities[conversation_id] = merged

    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        return self._refs.get(conversation_id)

    async def get_email_by_sender(
        self,
        conversation_id: str,
        aad_id: str,
    ) -> str | None:
        identity = self._identities.get(conversation_id)
        if not identity:
            return None
        if identity.get("aad_id") != aad_id:
            return None
        email = identity.get("email")
        return email if isinstance(email, str) else None

    async def get_by_email(self, email: str) -> dict[str, Any] | None:
        for conv_id, identity in self._identities.items():
            if identity.get("email") == email:
                return {"conversation_id": conv_id, "reference": self._refs.get(conv_id)}
        return None

    async def delete(self, conversation_id: str) -> None:
        self._refs.pop(conversation_id, None)
        self._identities.pop(conversation_id, None)

    async def list_all(self) -> dict[str, dict[str, Any]]:
        return dict(self._refs)
