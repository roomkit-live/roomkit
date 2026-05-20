"""Abstract base class for Teams providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.teams.models import TeamsMember

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from botbuilder.core import TurnContext


class TeamsProvider(ABC):
    """Microsoft Teams delivery + dispatch provider.

    Implementations own every interaction with the Bot Framework transport
    so consumers (e.g. Luge) never touch the SDK directly. The contract
    covers both outbound delivery (:meth:`send`) and inbound dispatch
    (:meth:`process_inbound`), plus the Teams-specific roster helpers
    needed to resolve sender identity (:meth:`get_member`,
    :meth:`list_members`).
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__

    @abstractmethod
    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a Microsoft Teams message.

        Args:
            event: The room event containing the message content.
            to: Recipient Teams conversation ID.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    @abstractmethod
    async def process_inbound(
        self,
        payload: dict[str, Any],
        auth_header: str,
        on_turn: Callable[[TurnContext], Awaitable[None]],
    ) -> None:
        """Validate an inbound activity and dispatch ``on_turn`` with a TurnContext.

        Concrete implementations are expected to:

        1. Validate the JWT in ``auth_header`` against Bot Framework /
           tenant credentials (raising on failure).
        2. Build a provider-specific :class:`TurnContext` populated with
           a working connector client for the activity's ``serviceUrl``.
        3. Invoke ``on_turn(turn_context)``.

        Consumers carry their business logic in ``on_turn`` and can use
        :meth:`get_member` / :meth:`list_members` against the provider —
        they should never need direct SDK access.

        Args:
            payload: Raw Bot Framework Activity dict (as received from
                the messaging endpoint).
            auth_header: Full ``Authorization`` header value, including
                the ``Bearer `` prefix.
            on_turn: Awaitable callback invoked once the activity has
                been authenticated and a ``TurnContext`` constructed.

        Raises:
            PermissionError: If the JWT is invalid or required
                credentials are missing.
        """
        ...

    @abstractmethod
    async def get_member(
        self,
        turn_context: TurnContext,
        member_id: str,
    ) -> TeamsMember | None:
        """Resolve a single member's profile via the Bot Framework roster.

        Args:
            turn_context: The active turn for which roster API calls are
                authorized.
            member_id: Bot-framework user ID (``"29:..."`` or AAD object ID).

        Returns:
            The normalized member record, or ``None`` if the roster
            does not have an entry for ``member_id``.
        """
        ...

    @abstractmethod
    async def list_members(
        self,
        turn_context: TurnContext,
        *,
        max_count: int | None = None,
    ) -> list[TeamsMember]:
        """Return the participant list for the active conversation.

        Args:
            turn_context: The active turn for which roster API calls are
                authorized.
            max_count: Hard cap on the number of members returned across
                paginated calls. ``None`` returns all members (use with
                care in large teams).

        Returns:
            Normalized member records. Empty list when the call fails or
            the conversation has no addressable members.
        """
        ...

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify that a webhook payload was signed by the Bot Framework.

        Args:
            payload: Raw request body bytes.
            signature: Value of the ``Authorization`` header (``Bearer <token>``).

        Returns:
            True if the signature is valid, False otherwise.

        Raises:
            NotImplementedError: If the provider does not support signature
                verification.
        """
        raise NotImplementedError(f"{self.name} does not support webhook signature verification")

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""
