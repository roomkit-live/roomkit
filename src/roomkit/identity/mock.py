"""Mock identity resolver for testing."""

from __future__ import annotations

from roomkit.identity.base import IdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import IdentificationStatus
from roomkit.models.identity import Identity, IdentityResult


class MockIdentityResolver(IdentityResolver):
    """Resolves identity from a pre-configured mapping.

    Supports three resolution outcomes:
    - **Identified**: sender_id found in ``mapping`` → single identity.
    - **Ambiguous**: sender_id found in ``ambiguous`` → multiple candidates.
    - **Unknown/Pending**: no match → status controlled by ``unknown_status``.
    """

    def __init__(
        self,
        mapping: dict[str, Identity] | None = None,
        ambiguous: dict[str, list[Identity]] | None = None,
        unknown_status: IdentificationStatus = IdentificationStatus.UNKNOWN,
    ) -> None:
        self._mapping = mapping or {}
        self._ambiguous = ambiguous or {}
        self._unknown_status = unknown_status

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        # Exact match → identified
        identity = self._mapping.get(message.sender_id)
        if identity:
            return IdentityResult(
                status=IdentificationStatus.IDENTIFIED,
                identity=identity,
            )

        # Multiple candidates → ambiguous
        candidates = self._ambiguous.get(message.sender_id)
        if candidates:
            return IdentityResult(
                status=IdentificationStatus.AMBIGUOUS,
                candidates=candidates,
            )

        # No match
        return IdentityResult(status=self._unknown_status)
