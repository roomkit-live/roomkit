"""Mock Teams provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.teams.base import TeamsProvider
from roomkit.providers.teams.models import TeamsMember

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from botbuilder.core import TurnContext


class MockTeamsProvider(TeamsProvider):
    """Records sent messages and serves a scripted roster for verification.

    Tests can pre-load ``self.members[conversation_id] = [TeamsMember(...), ...]``
    and the provider's :meth:`get_member` / :meth:`list_members` will return
    those records. ``process_inbound`` synthesizes a minimal stand-in turn
    context — enough for callbacks that only read the activity. Tests that
    need richer SDK behaviour should swap in a real ``BotFrameworkAdapter``
    via the concrete provider.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.processed: list[dict[str, Any]] = []
        self.members: dict[str, list[TeamsMember]] = {}

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        self.sent.append({"event": event, "to": to})
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

    async def process_inbound(
        self,
        payload: dict[str, Any],
        auth_header: str,  # noqa: ARG002 — fake provider skips JWT checks
        on_turn: Callable[[TurnContext], Awaitable[None]],
    ) -> None:
        self.processed.append({"payload": payload})
        ctx = _FakeTurnContext(payload, self.members)
        await on_turn(ctx)  # ty: ignore[invalid-argument-type]

    async def get_member(
        self,
        turn_context: TurnContext,
        member_id: str,
    ) -> TeamsMember | None:
        # The fake turn context carries the scripted roster.
        members = getattr(turn_context, "_members", None) or []
        for member in members:
            if member.teams_user_id == member_id or member.aad_object_id == member_id:
                return member
        return None

    async def list_members(
        self,
        turn_context: TurnContext,
        *,
        max_count: int | None = None,
    ) -> list[TeamsMember]:
        members = getattr(turn_context, "_members", None) or []
        if max_count is None:
            return list(members)
        return list(members[:max_count])


class _FakeTurnContext:
    """Minimal stand-in for ``botbuilder.core.TurnContext`` in unit tests."""

    def __init__(self, payload: dict[str, Any], members: dict[str, list[TeamsMember]]) -> None:
        self.activity = _FakeActivity(payload)
        conv_id = (payload.get("conversation") or {}).get("id", "")
        self._members = members.get(conv_id, [])


class _FakeActivity:
    """Thin dict-backed activity stub exposing the fields callbacks usually read."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.type = payload.get("type", "")
        self.text = payload.get("text", "")
        self.id = payload.get("id", "")
        self.conversation = _Bag(payload.get("conversation") or {})
        self.from_property = _Bag(payload.get("from") or {})
        self.recipient = _Bag(payload.get("recipient") or {})
        self.channel_data = payload.get("channelData") or {}
        self.service_url = payload.get("serviceUrl", "")


class _Bag:
    """Attribute access over a dict for snake/camel field equivalence."""

    _ALIASES = {
        "aad_object_id": "aadObjectId",
        "user_principal_name": "userPrincipalName",
    }

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            return self._data[item]
        camel = self._ALIASES.get(item)
        if camel and camel in self._data:
            return self._data[camel]
        return None
