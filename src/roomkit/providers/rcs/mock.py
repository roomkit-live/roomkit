"""Mock RCS provider for testing."""

from __future__ import annotations

from typing import Any

from roomkit.models.event import RoomEvent
from roomkit.providers.rcs.base import RCSDeliveryResult, RCSProvider


class MockRCSProvider(RCSProvider):
    """Mock RCS provider for testing."""

    def __init__(
        self,
        sender_id: str = "mock_rcs_agent",
        *,
        simulate_fallback: bool = False,
        simulate_failure: bool = False,
    ) -> None:
        """Initialize mock provider.

        Args:
            sender_id: Mock sender/agent ID.
            simulate_fallback: If True, simulate SMS fallback on sends.
            simulate_failure: If True, simulate send failures.
        """
        self._sender_id = sender_id
        self._simulate_fallback = simulate_fallback
        self._simulate_failure = simulate_failure
        self.calls: list[dict[str, Any]] = []
        self._message_counter = 0

    @property
    def sender_id(self) -> str:
        return self._sender_id

    async def send(
        self,
        event: RoomEvent,
        to: str,
        *,
        fallback: bool = True,
    ) -> RCSDeliveryResult:
        self._message_counter += 1
        self.calls.append(
            {
                "event": event,
                "to": to,
                "fallback": fallback,
            }
        )

        if self._simulate_failure:
            return RCSDeliveryResult(
                success=False,
                error="simulated_failure",
            )

        # Simulate fallback to SMS
        if self._simulate_fallback and fallback:
            return RCSDeliveryResult(
                success=True,
                provider_message_id=f"mock_sms_{self._message_counter}",
                channel_used="sms",
                fallback=True,
            )

        return RCSDeliveryResult(
            success=True,
            provider_message_id=f"mock_rcs_{self._message_counter}",
            channel_used="rcs",
            fallback=False,
        )

    async def check_capability(self, phone_number: str) -> bool:
        """Mock capability check - returns opposite of simulate_fallback."""
        return not self._simulate_fallback
