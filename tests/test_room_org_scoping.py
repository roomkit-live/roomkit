"""Room operations can be scoped to one organization (RFC §17.2).

A library has no caller or auth context of its own, so the scope has to come
from the caller: the operations below take an optional `organization_id` and,
when given one, refuse a room belonging to anyone else.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit import (
    Access,
    AgentResponsePolicy,
    HookExecution,
    HookTrigger,
    RoomKit,
    RoomTimers,
    Swarm,
)
from roomkit.channels import SMSChannel
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.identity import Identity
from roomkit.models.steering import Cancel, InjectMessage
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.realtime.base import EphemeralCallback, EphemeralEvent
from roomkit.realtime.memory import InMemoryRealtime
from tests.test_framework import SimpleChannel


async def _two_tenants() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="acme-room", organization_id="acme")
    await kit.create_room(room_id="globex-room", organization_id="globex")
    return kit


class TestScopedReads:
    async def test_matching_organization_reads_the_room(self) -> None:
        kit = await _two_tenants()
        room = await kit.get_room("acme-room", organization_id="acme")
        assert room.id == "acme-room"

    async def test_other_organization_reads_as_not_found(self) -> None:
        """Reported as missing, not as "wrong organization": a distinct error
        would let a caller probe which room ids exist outside its tenant."""
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.get_room("globex-room", organization_id="acme")

    async def test_unscoped_read_is_unchanged(self) -> None:
        kit = await _two_tenants()
        assert (await kit.get_room("globex-room")).id == "globex-room"

    async def test_timeline_is_scoped(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.get_timeline("globex-room", organization_id="acme")
        assert await kit.get_timeline("globex-room", organization_id="globex") is not None


class TestScopedMutations:
    async def test_attach_channel_is_scoped(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.attach_channel("globex-room", "sms1", organization_id="acme")
        # The refusal left nothing behind.
        assert await kit.store.get_binding("globex-room", "sms1") is None

        binding = await kit.attach_channel("globex-room", "sms1", organization_id="globex")
        assert binding.channel_id == "sms1"

    async def test_detach_channel_is_scoped(self) -> None:
        kit = await _two_tenants()
        await kit.attach_channel("globex-room", "sms1")

        with pytest.raises(RoomNotFoundError):
            await kit.detach_channel("globex-room", "sms1", organization_id="acme")
        assert await kit.store.get_binding("globex-room", "sms1") is not None

        assert await kit.detach_channel("globex-room", "sms1", organization_id="globex") is True

    async def test_close_and_archive_are_scoped(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.close_room("globex-room", organization_id="acme")
        with pytest.raises(RoomNotFoundError):
            await kit.archive_room("globex-room", organization_id="acme")

        assert (await kit.get_room("globex-room")).status.value == "active"

    async def test_send_event_is_scoped(self) -> None:
        kit = await _two_tenants()
        await kit.attach_channel("globex-room", "sms1")

        with pytest.raises(RoomNotFoundError):
            await kit.send_event(
                "globex-room",
                "sms1",
                TextContent(body="from the wrong tenant"),
                organization_id="acme",
            )
        assert await kit.get_timeline("globex-room") == await kit.get_timeline("globex-room")

        event = await kit.send_event(
            "globex-room",
            "sms1",
            TextContent(body="from the right tenant"),
            organization_id="globex",
        )
        assert event.id

    async def test_binding_operations_share_the_room_scope(self) -> None:
        kit = await _two_tenants()
        await kit.attach_channel("globex-room", "sms1")

        operations = [
            kit.mute("globex-room", "sms1", organization_id="acme"),
            kit.unmute("globex-room", "sms1", organization_id="acme"),
            kit.mute_output("globex-room", "sms1", organization_id="acme"),
            kit.unmute_output("globex-room", "sms1", organization_id="acme"),
            kit.set_visibility("globex-room", "sms1", "none", organization_id="acme"),
            kit.set_access("globex-room", "sms1", Access.NONE, organization_id="acme"),
            kit.update_binding_metadata(
                "globex-room", "sms1", {"leaked": True}, organization_id="acme"
            ),
            kit.get_binding("globex-room", "sms1", organization_id="acme"),
            kit.list_bindings("globex-room", organization_id="acme"),
        ]
        for operation in operations:
            with pytest.raises(RoomNotFoundError):
                await operation

        binding = await kit.get_binding("globex-room", "sms1")
        assert binding.access is Access.READ_WRITE
        assert binding.muted is False
        assert binding.output_muted is False
        assert "leaked" not in binding.metadata

    async def test_lifecycle_and_participant_operations_share_the_room_scope(self) -> None:
        kit = await _two_tenants()

        operations = [
            kit.set_room_timers(
                "globex-room", RoomTimers(closed_after_seconds=10), organization_id="acme"
            ),
            kit.set_agent_response_policy(
                "globex-room",
                AgentResponsePolicy.ADDRESSED_ONLY,
                organization_id="acme",
            ),
            kit.check_room_timers("globex-room", organization_id="acme"),
            kit.update_room_metadata("globex-room", {"leaked": True}, organization_id="acme"),
            kit.ensure_participant("globex-room", "sms1", "p1", organization_id="acme"),
            kit.resolve_participant("globex-room", "p1", "identity-1", organization_id="acme"),
        ]
        for operation in operations:
            with pytest.raises(RoomNotFoundError):
                await operation

        room = await kit.get_room("globex-room")
        assert "leaked" not in room.metadata
        assert await kit.store.get_participant("globex-room", "p1") is None

    async def test_query_and_read_tracking_operations_share_the_room_scope(self) -> None:
        kit = await _two_tenants()

        operations = [
            kit.list_tasks("globex-room", organization_id="acme"),
            kit.list_observations("globex-room", organization_id="acme"),
            kit.mark_read("globex-room", "sms1", "event-1", organization_id="acme"),
            kit.mark_all_read("globex-room", "sms1", organization_id="acme"),
            kit.list_read_markers("globex-room", organization_id="acme"),
        ]
        for operation in operations:
            with pytest.raises(RoomNotFoundError):
                await operation

        assert await kit.list_read_markers("globex-room") == {}

    async def test_check_all_timers_can_be_limited_to_one_tenant(self) -> None:
        kit = await _two_tenants()
        expired = RoomTimers(
            inactive_after_seconds=1,
            last_activity_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        await kit.set_room_timers("globex-room", expired)

        transitioned = await kit.check_all_timers(organization_id="acme")

        assert transitioned == []
        assert (await kit.get_room("globex-room")).status.value == "active"


class TestUnscopedRooms:
    async def test_a_room_without_an_organization_is_reachable_unscoped_only(self) -> None:
        """A room created with no organization belongs to no tenant, so a
        scoped caller does not reach it."""
        kit = RoomKit()
        await kit.create_room(room_id="solo")

        assert (await kit.get_room("solo")).id == "solo"
        with pytest.raises(RoomNotFoundError):
            await kit.get_room("solo", organization_id="acme")


def _message() -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="+15550001", content=TextContent(body="hi"))


class _AnswersWith(InboundRoomRouter):
    """A custom router giving one answer, recording the organization and the
    identity it was given."""

    def __init__(self, room_id: str | None) -> None:
        self.room_id = room_id
        self.asked: list[tuple[str | None, str | None]] = []

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
        identity: Identity | None = None,
    ) -> str | None:
        self.asked.append((organization_id, identity.id if identity is not None else None))
        return self.room_id


class TestScopedInbound:
    """``process_inbound`` scopes the room a message lands in: routed or
    explicit, a room of another organization is not found to the caller, before
    any event is committed or any channel auto-attached."""

    async def test_an_explicit_room_of_another_organization_is_not_found(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.process_inbound(_message(), room_id="globex-room", organization_id="acme")
        # The refusal left nothing behind: no event, and no auto-attach either.
        assert await kit.store.list_events("globex-room") == []
        assert await kit.store.get_binding("globex-room", "sms1") is None

    async def test_a_matching_organization_processes_the_message(self) -> None:
        kit = await _two_tenants()
        result = await kit.process_inbound(_message(), room_id="acme-room", organization_id="acme")
        assert result.event is not None and result.event.room_id == "acme-room"

    async def test_a_room_that_does_not_exist_is_created_under_the_caller(self) -> None:
        kit = await _two_tenants()
        await kit.process_inbound(_message(), room_id="fresh-room", organization_id="acme")
        assert (await kit.get_room("fresh-room")).organization_id == "acme"

    @pytest.mark.parametrize("names_the_sender", [True, False], ids=["step-1", "step-3"])
    async def test_the_default_router_never_returns_another_organizations_room(
        self, names_the_sender: bool
    ) -> None:
        """RFC §10.4: the default strategy searches the caller's organization
        only, so the sender's room at another tenant, or the one room of
        another tenant bound to the channel, is not a candidate: the message
        gets a room of its own organization."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="globex-room", organization_id="globex")
        sender = _message().sender_id if names_the_sender else None
        await kit.attach_channel("globex-room", "sms1", participant_id=sender)
        before = await kit.store.list_events("globex-room")

        result = await kit.process_inbound(_message(), organization_id="acme")

        assert result.event is not None and result.event.room_id != "globex-room"
        assert (await kit.get_room(result.event.room_id)).organization_id == "acme"
        assert await kit.store.list_events("globex-room") == before

    async def test_a_router_answering_another_organizations_room_fails_the_message(
        self,
    ) -> None:
        """The answer is held to step 2's rules whoever gave it (RFC §10.1)."""
        router = _AnswersWith("globex-room")
        kit = RoomKit(inbound_router=router)
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="globex-room", organization_id="globex")
        await kit.attach_channel("globex-room", "sms1")
        before = await kit.store.list_events("globex-room")

        with pytest.raises(RoomNotFoundError):
            await kit.process_inbound(_message(), organization_id="acme")

        assert await kit.store.list_events("globex-room") == before
        binding = await kit.store.get_binding("globex-room", "sms1")
        assert binding is not None and binding.participant_id is None

    async def test_a_room_a_router_answers_that_does_not_exist_is_created_with_its_id(
        self,
    ) -> None:
        kit = RoomKit(inbound_router=_AnswersWith("fresh-room"))
        kit.register_channel(SimpleChannel("sms1"))

        result = await kit.process_inbound(_message(), organization_id="acme")

        assert result.event is not None and result.event.room_id == "fresh-room"
        assert (await kit.get_room("fresh-room")).organization_id == "acme"
        binding = await kit.store.get_binding("fresh-room", "sms1")
        assert binding is not None and binding.participant_id == _message().sender_id

    async def test_the_router_is_given_the_organization_and_the_identity(self) -> None:
        """Read once, scoped to the caller's organization (RFC §10.4, §17.2):
        an identity of an organization is no one's to an unscoped call."""
        router = _AnswersWith(None)
        kit = RoomKit(inbound_router=router)
        kit.register_channel(SimpleChannel("sms1"))
        await kit.store.create_identity(Identity(id="id-alice", organization_id="acme"))
        await kit.store.link_address("id-alice", "sms", _message().sender_id, "acme")

        await kit.process_inbound(_message(), organization_id="acme")
        await kit.process_inbound(_message())

        assert router.asked == [("acme", "id-alice"), (None, None)]

    async def test_an_address_linked_within_an_organization_finds_its_persons_room(
        self,
    ) -> None:
        """The host added alice under her identity, then linked her number in
        its organization: her first text from that number lands in her room,
        which only step 1, by the identity, can find (two rooms share the
        number)."""
        kit = RoomKit()
        kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
        for room_id in ("alice-room", "carol-room"):
            await kit.create_room(room_id=room_id, organization_id="acme")
        await kit.attach_channel("carol-room", "sms", participant_id="+15550000003")
        await kit.attach_channel("alice-room", "sms")
        await kit.add_member("alice-room", "sms", "id-alice")
        await kit.store.create_identity(Identity(id="id-alice", organization_id="acme"))
        await kit.store.link_address("id-alice", str(ChannelType.SMS), "+15550000001", "acme")

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")
            ),
            organization_id="acme",
        )

        assert result.event is not None and result.event.room_id == "alice-room"

    async def test_unscoped_processing_is_unchanged(self) -> None:
        kit = await _two_tenants()
        result = await kit.process_inbound(_message(), room_id="globex-room")
        assert result.event is not None and result.event.room_id == "globex-room"

    async def test_a_routed_miss_creates_the_room_under_the_caller(self) -> None:
        """No room routes for this channel: the auto-created one belongs to
        the caller's organization."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        result = await kit.process_inbound(_message(), organization_id="acme")
        assert (await kit.get_room(result.event.room_id)).organization_id == "acme"

    @pytest.mark.parametrize("organization_id", [None, "acme", "globex"])
    async def test_a_named_room_created_meanwhile_is_read_again(
        self, monkeypatch: pytest.MonkeyPatch, organization_id: str | None
    ) -> None:
        """RFC §10.1 step 2, §14.3: a creation the store refuses because
        another call created the room meanwhile reads the room again, as one
        that exists, checked for the caller's organization; the room it
        holds is never replaced."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="shared", organization_id="acme", metadata={"k": "v"})
        # The existence read is stale: room_exists unscoped, get_room scoped.
        reads = {"stale": organization_id is not None}
        real_get_room = kit.store.get_room

        async def not_yet(room_id: str) -> bool:
            return False

        async def stale_once(room_id: str) -> Any:
            if reads.pop("stale", False):
                return None
            return await real_get_room(room_id)

        monkeypatch.setattr(kit.store, "room_exists", not_yet)
        monkeypatch.setattr(kit.store, "get_room", stale_once)

        if organization_id == "globex":
            with pytest.raises(RoomNotFoundError):
                await kit.process_inbound(
                    _message(), room_id="shared", organization_id=organization_id
                )
        else:
            result = await kit.process_inbound(
                _message(), room_id="shared", organization_id=organization_id
            )
            assert result.event is not None and result.event.room_id == "shared"
        room = await real_get_room("shared")
        assert room is not None and (room.organization_id, room.metadata) == ("acme", {"k": "v"})
        await kit.close()

    async def test_a_deferred_send_to_another_organization_is_refused_synchronously(self) -> None:
        kit = await _two_tenants()
        with pytest.raises(RoomNotFoundError):
            await kit.process_inbound(
                _message(), room_id="globex-room", organization_id="acme", defer_delivery=True
            )
        assert await kit.store.list_events("globex-room") == []

    async def test_a_matching_organization_still_auto_attaches_the_channel(self) -> None:
        kit = await _two_tenants()
        assert await kit.store.get_binding("acme-room", "sms1") is None
        await kit.process_inbound(_message(), room_id="acme-room", organization_id="acme")
        assert await kit.store.get_binding("acme-room", "sms1") is not None


async def _globex_with_alice() -> RoomKit:
    kit = await _two_tenants()
    await kit.attach_channel("globex-room", "sms1")
    await kit.add_member("globex-room", "sms1", "alice", identity_id="id-alice")
    return kit


async def _globex_roster(kit: RoomKit) -> tuple[list[Any], list[Any]]:
    """What a membership call may write: the room's participants and events."""
    return (
        await kit.store.list_participants("globex-room"),
        await kit.store.list_events("globex-room"),
    )


_MEMBERSHIP: dict[str, Callable[[RoomKit, str | None], Awaitable[Any]]] = {
    "add_member": lambda kit, org: kit.add_member(
        "globex-room", "sms1", "intruder", organization_id=org
    ),
    "remove_member": lambda kit, org: kit.remove_member(
        "globex-room", "alice", organization_id=org
    ),
    "rename_member": lambda kit, org: kit.rename_member(
        "globex-room", "alice", "Mallory", organization_id=org
    ),
    "list_members": lambda kit, org: kit.list_members("globex-room", organization_id=org),
    "is_member": lambda kit, org: kit.is_member("globex-room", "id-alice", organization_id=org),
}


class TestScopedMembership:
    """The roster of another organization's room is neither read nor written
    (RFC §17.2, RMK-488)."""

    @pytest.mark.parametrize("verb", sorted(_MEMBERSHIP))
    async def test_another_organizations_room_is_not_found_and_keeps_its_roster(
        self, verb: str
    ) -> None:
        kit = await _globex_with_alice()
        before = await _globex_roster(kit)

        with pytest.raises(RoomNotFoundError):
            await _MEMBERSHIP[verb](kit, "acme")

        assert await _globex_roster(kit) == before
        await kit.close()

    @pytest.mark.parametrize("organization_id", ["globex", None], ids=["own", "unscoped"])
    async def test_the_rooms_own_organization_or_none_reads_and_writes_the_roster(
        self, organization_id: str | None
    ) -> None:
        kit = await _globex_with_alice()

        assert await kit.is_member("globex-room", "id-alice", organization_id=organization_id)
        await kit.add_member("globex-room", "sms1", "bob", organization_id=organization_id)
        await kit.rename_member("globex-room", "alice", "Alice", organization_id=organization_id)
        await kit.remove_member("globex-room", "bob", organization_id=organization_id)

        members = await kit.list_members("globex-room", organization_id=organization_id)
        assert [(member.id, member.display_name) for member in members] == [("alice", "Alice")]
        await kit.close()


async def _globex_with_agents(**options: Any) -> tuple[RoomKit, list[MockAIProvider]]:
    """Globex's room, its agent having answered one message, and a worker
    registered for a delegation."""
    agent = MockAIProvider(responses=["answer"])
    worker = MockAIProvider(responses=["done"])
    kit = RoomKit(**options)
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(Agent("ai", provider=agent, greeting="Hello!", auto_greet=False))
    kit.register_channel(AIChannel("worker", provider=worker))
    await kit.create_room(room_id="globex-room", organization_id="globex")
    await kit.attach_channel("globex-room", "sms1")
    await kit.attach_channel("globex-room", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(_message(), room_id="globex-room")
    return kit, [agent, worker]


async def _globex_state(kit: RoomKit, providers: list[MockAIProvider]) -> tuple[Any, ...]:
    """What a turn, a delivery or a host call may leave: events, rooms,
    observations, model calls."""
    return (
        await kit.store.list_events("globex-room"),
        [room.id for room in await kit.store.list_rooms()],
        await kit.store.list_observations("globex-room"),
        [len(provider.calls) for provider in providers],
    )


_TURNS: dict[str, Callable[[RoomKit, str | None], Awaitable[Any]]] = {
    "deliver": lambda kit, org: kit.deliver("globex-room", "from acme", organization_id=org),
    "send_greeting": lambda kit, org: kit.send_greeting("globex-room", organization_id=org),
    "regenerate_target": lambda kit, org: kit.regenerate_target(
        "globex-room", organization_id=org
    ),
    "regenerate_response": lambda kit, org: kit.regenerate_response(
        "globex-room", organization_id=org
    ),
    "delegate": lambda kit, org: kit.delegate(
        "globex-room", "worker", "Summarize the room.", wait=True, organization_id=org
    ),
}


class TestScopedTurnsAndDeliveries:
    """Another organization's room is not made to speak, delivered to, read
    for a regeneration or delegated from (RFC §17.2, RMK-490)."""

    @pytest.mark.parametrize("verb", sorted(_TURNS))
    async def test_another_organizations_room_is_not_found_and_nothing_runs(
        self, verb: str
    ) -> None:
        kit, providers = await _globex_with_agents()
        before = await _globex_state(kit, providers)

        with pytest.raises(RoomNotFoundError):
            await _TURNS[verb](kit, "acme")

        assert await _globex_state(kit, providers) == before
        await kit.close()

    async def test_a_queued_delivery_to_another_organization_never_reaches_the_queue(
        self,
    ) -> None:
        backend = InMemoryDeliveryBackend()
        kit, _ = await _globex_with_agents(delivery_backend=backend)

        with pytest.raises(RoomNotFoundError):
            await kit.deliver("globex-room", "from acme", organization_id="acme")

        assert await backend.get_queue_depth() == 0
        await kit.close()

    @pytest.mark.parametrize("organization_id", ["globex", None], ids=["own", "unscoped"])
    async def test_the_rooms_own_organization_or_none_runs_each_verb(
        self, organization_id: str | None
    ) -> None:
        kit, (agent, worker) = await _globex_with_agents()

        target = await kit.regenerate_target("globex-room", organization_id=organization_id)
        assert target is not None and target.content.body == "hi"
        regenerated = await kit.regenerate_response("globex-room", organization_id=organization_id)
        assert regenerated is not None and len(agent.calls) == 2
        await kit.send_greeting("globex-room", organization_id=organization_id)
        bodies = [e.content.body for e in await kit.get_timeline("globex-room")]
        assert "Hello!" in bodies
        outcome = await kit.deliver("globex-room", "a nudge", organization_id=organization_id)
        assert outcome.status == "sent" and len(agent.calls) == 3
        task = await kit.delegate(
            "globex-room", "worker", "Summarize.", wait=True, organization_id=organization_id
        )
        assert task.parent_room_id == "globex-room" and len(worker.calls) == 1
        await kit.close()


_FROM_ANOTHER_ORGANIZATION: dict[str, Callable[[RoomKit], Awaitable[Any]]] = {
    "regenerate_response": lambda kit: kit.regenerate_response(
        "globex-room", organization_id="acme"
    ),
    "process_inbound": lambda kit: kit.process_inbound(
        _message(), room_id="globex-room", organization_id="acme"
    ),
    "send_event": lambda kit: kit.send_event(
        "globex-room", "sms1", TextContent(body="from acme"), organization_id="acme"
    ),
    "post": lambda kit: kit.post("globex-room", "from acme", organization_id="acme"),
    "deliver": lambda kit: kit.deliver("globex-room", "from acme", organization_id="acme"),
}


@pytest.mark.parametrize("call", sorted(_FROM_ANOTHER_ORGANIZATION))
async def test_a_refused_call_draws_nothing_from_the_hooks_budget(call: str) -> None:
    """The scope is read before the call draws on the budget of the ASYNC
    hook's body it is made from (RFC §9.5.1, §17.2): eleven refused calls,
    past the cascade's ten hook calls (depth limit 1), leave the budget
    whole, and the body's direct change still writes."""
    kit = RoomKit(max_chain_depth=1)
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="globex-room", organization_id="globex")
    await kit.attach_channel("globex-room", "sms1")
    outcomes: list[str] = []
    done = asyncio.Event()

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="probe")
    async def probe(event: Any, context: Any) -> None:
        if done.is_set():
            return
        try:
            for _ in range(11):
                try:
                    await _FROM_ANOTHER_ORGANIZATION[call](kit)
                    outcomes.append("not refused")
                except RoomNotFoundError as error:
                    outcomes.append(type(error).__name__)
            updated = await kit.update_event("globex-room", event.id, metadata={"seen": True})
            outcomes.append("updated" if updated is not None else "not updated")
        except Exception as error:  # the type is the assertion
            outcomes.append(type(error).__name__)
        finally:
            done.set()

    await kit.process_inbound(_message(), room_id="globex-room")
    await asyncio.wait_for(done.wait(), 2)

    assert outcomes == ["RoomNotFoundError"] * 11 + ["updated"]
    await kit.close()


_HOST_DOORS: dict[str, Callable[[RoomKit, str | None], Awaitable[Any]]] = {
    "start_task": lambda kit, org: kit.start_task(
        "globex-room", "ai", "Summarize the room.", organization_id=org
    ),
    "submit_feedback": lambda kit, org: kit.submit_feedback(
        "globex-room", 0.9, comment="great", organization_id=org
    ),
    "steer": lambda kit, org: kit.steer(
        "globex-room", "ai", InjectMessage(content="Mind the budget."), organization_id=org
    ),
}


class _Paused:
    """A host tool that holds its turn until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.started.set()
        await self.release.wait()
        return "waited"


async def _globex_turn_held(tool: _Paused) -> tuple[RoomKit, MockAIProvider]:
    """Globex's room, its agent's turn held in a tool call."""
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="wait_here", arguments={})])
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="done")])
    wait = AITool(name="wait_here", description="Wait.", parameters={"type": "object"})
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        Agent("ai", provider=provider, role="agent", tools=[wait], tool_handler=tool)
    )
    await kit.create_room(room_id="globex-room", organization_id="globex")
    await kit.attach_channel("globex-room", "sms1")
    await kit.attach_channel("globex-room", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


class TestScopedHostDoors:
    """A background task, a feedback and a steer reach no room of another
    organization (RFC §17.2, RMK-487)."""

    @pytest.mark.parametrize("verb", sorted(_HOST_DOORS))
    async def test_another_organizations_room_is_not_found_and_nothing_runs(
        self, verb: str
    ) -> None:
        kit, providers = await _globex_with_agents()
        before = await _globex_state(kit, providers)

        with pytest.raises(RoomNotFoundError):
            await _HOST_DOORS[verb](kit, "acme")

        assert await _globex_state(kit, providers) == before
        await kit.close()

    @pytest.mark.parametrize("organization_id", ["globex", None], ids=["own", "unscoped"])
    async def test_the_rooms_own_organization_or_none_runs_each_verb(
        self, organization_id: str | None
    ) -> None:
        kit, (agent, _) = await _globex_with_agents()

        task = await kit.start_task(
            "globex-room", "ai", "Summarize.", organization_id=organization_id
        )
        await asyncio.wait_for(task.wait(), 5)
        assert len(agent.calls) > 1  # the task ran on the room's agent
        await kit.submit_feedback("globex-room", 0.9, organization_id=organization_id)
        assert len(await kit.store.list_observations("globex-room")) == 1
        steered = await kit.steer("globex-room", "ai", Cancel(), organization_id=organization_id)
        assert steered == 0  # no turn runs
        await kit.close()

    @pytest.mark.parametrize("organization_id", ["globex", None], ids=["own", "unscoped"])
    async def test_a_steer_for_the_rooms_own_organization_or_none_reaches_its_turn(
        self, organization_id: str | None
    ) -> None:
        tool = _Paused()
        kit, provider = await _globex_turn_held(tool)
        turn = asyncio.create_task(kit.process_inbound(_message(), room_id="globex-room"))
        await asyncio.wait_for(tool.started.wait(), 2)

        count = await kit.steer(
            "globex-room",
            "ai",
            InjectMessage(content="Mind the budget."),
            organization_id=organization_id,
        )
        tool.release.set()
        await asyncio.wait_for(turn, 5)

        assert count == 1
        assert any("Mind the budget." in str(m.content) for m in provider.calls[1].messages)
        await kit.close()


@pytest.mark.parametrize(
    ("organization_id", "cancelled"), [("acme", False), ("globex", True), (None, True)]
)
async def test_a_task_of_another_organizations_room_is_not_cancelled(
    organization_id: str | None, cancelled: bool
) -> None:
    """``cancel_task(task_id, organization_id=...)`` cancels only a task of
    that organization's rooms (RFC §17.2): another's is a task it does not
    hold, ``False`` and left running."""
    tool = _Paused()
    kit, _ = await _globex_turn_held(tool)
    task = await kit.delegate("globex-room", "ai", "Wait here.", notify="ai")
    await asyncio.wait_for(tool.started.wait(), 2)

    assert await kit.cancel_task(task.id, organization_id=organization_id) is cancelled
    assert task.status.value == ("cancelled" if cancelled else "in_progress")
    tool.release.set()
    await kit.close()


class _RoomHooks:
    """Globex's room with a room hook reading its messages and an executor
    room hook enabled for it; what acme's host tries on it."""

    def __init__(self, kit: RoomKit) -> None:
        self.kit = kit
        self.globex_read: list[str] = []
        self.acme_read: list[str] = []

    async def globex_hook(self, event: Any, context: Any) -> None:
        self.globex_read.append(event.content.body)

    async def acme_hook(self, event: Any, context: Any) -> None:
        self.acme_read.append(event.content.body)


async def _globex_room_hooks() -> _RoomHooks:
    kit = await _two_tenants()
    await kit.attach_channel("globex-room", "sms1")
    hooks = _RoomHooks(kit)
    for name in ("audit", "acme-audit"):
        kit.turn_executor.room_hook(name, HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(
            hooks.acme_hook
        )
    await kit.add_room_hook(
        "globex-room",
        HookTrigger.AFTER_BROADCAST,
        HookExecution.ASYNC,
        hooks.globex_hook,
        name="globex-hook",
    )
    await kit.enable_room_hook("globex-room", "audit")
    return hooks


_ROOM_HOOK_VERBS: dict[str, Callable[[_RoomHooks, str | None], Awaitable[Any]]] = {
    "add_room_hook": lambda hooks, org: hooks.kit.add_room_hook(
        "globex-room",
        HookTrigger.AFTER_BROADCAST,
        HookExecution.ASYNC,
        hooks.acme_hook,
        name="acme-hook",
        organization_id=org,
    ),
    "remove_room_hook": lambda hooks, org: hooks.kit.remove_room_hook(
        "globex-room", "globex-hook", organization_id=org
    ),
    "enable_room_hook": lambda hooks, org: hooks.kit.enable_room_hook(
        "globex-room", "acme-audit", organization_id=org
    ),
    "disable_room_hook": lambda hooks, org: hooks.kit.disable_room_hook(
        "globex-room", "audit", organization_id=org
    ),
    "claim_room_strategy": lambda hooks, org: hooks.kit.claim_room_strategy(
        "globex-room", Swarm(agents=["a", "b"], entry="a"), organization_id=org
    ),
}


class TestScopedRoomHooks:
    """Another organization's room is given no hook, loses none, enables or
    disables none, and is claimed for no strategy (RFC §17.2, RMK-487)."""

    @pytest.mark.parametrize("verb", sorted(_ROOM_HOOK_VERBS))
    async def test_another_organizations_room_is_not_found_and_its_hooks_stand(
        self, verb: str
    ) -> None:
        hooks = await _globex_room_hooks()

        with pytest.raises(RoomNotFoundError):
            await _ROOM_HOOK_VERBS[verb](hooks, "acme")

        await hooks.kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="+15550001", content=TextContent(body="secret")
            ),
            room_id="globex-room",
        )
        await hooks.kit.hook_engine.drain_observers()  # its ASYNC hooks have run
        # acme reads nothing, globex's hook still does, and the enabled set
        # is the one globex made.
        assert (hooks.acme_read, hooks.globex_read) == ([], ["secret"])
        assert hooks.kit.hook_engine.room_hook_names("globex-room") == ["globex-hook"]
        assert not await hooks.kit.disable_room_hook("globex-room", "acme-audit")
        assert await hooks.kit.disable_room_hook("globex-room", "audit")
        await hooks.kit.close()

    @pytest.mark.parametrize("organization_id", ["globex", None], ids=["own", "unscoped"])
    async def test_the_rooms_own_organization_or_none_runs_each_verb(
        self, organization_id: str | None
    ) -> None:
        hooks = await _globex_room_hooks()
        kit = hooks.kit

        added = await _ROOM_HOOK_VERBS["add_room_hook"](hooks, organization_id)
        assert added.name == "acme-hook"
        assert await _ROOM_HOOK_VERBS["remove_room_hook"](hooks, organization_id) is True
        await _ROOM_HOOK_VERBS["enable_room_hook"](hooks, organization_id)
        assert await _ROOM_HOOK_VERBS["disable_room_hook"](hooks, organization_id) is True
        await _ROOM_HOOK_VERBS["claim_room_strategy"](hooks, organization_id)
        assert kit.hook_engine.room_hook_names("globex-room") == ["acme-hook"]
        assert await kit.disable_room_hook("globex-room", "acme-audit") is True
        await kit.close()


class _RecordingRealtime(InMemoryRealtime):
    """The channels the kit publishes to and subscribes to, read without
    waiting on a subscription's queue."""

    def __init__(self) -> None:
        super().__init__()
        self.reached: list[str] = []

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        self.reached.append(channel)
        await super().publish(channel, event)

    async def subscribe(self, channel: str, callback: EphemeralCallback) -> str:
        self.reached.append(channel)
        return await super().subscribe(channel, callback)


async def _ignore(event: EphemeralEvent) -> None:
    return None


_EPHEMERALS: dict[str, Callable[[RoomKit, str | None], Awaitable[Any]]] = {
    "publish_typing": lambda kit, org: kit.publish_typing(
        "globex-room", "u1", organization_id=org
    ),
    "publish_tool_call": lambda kit, org: kit.publish_tool_call(
        "globex-room", "ai", [{"id": "c1", "name": "lookup"}], organization_id=org
    ),
    "publish_presence": lambda kit, org: kit.publish_presence(
        "globex-room", "u1", "online", organization_id=org
    ),
    "publish_read_receipt": lambda kit, org: kit.publish_read_receipt(
        "globex-room", "u1", "e1", organization_id=org
    ),
    "publish_reaction": lambda kit, org: kit.publish_reaction(
        "globex-room", "u1", "e1", "+1", organization_id=org
    ),
    "subscribe_room": lambda kit, org: kit.subscribe_room(
        "globex-room", _ignore, organization_id=org
    ),
}


class TestScopedEphemerals:
    """Nothing is published to, or heard from, another organization's room
    (RFC §17.2, RMK-491)."""

    @pytest.mark.parametrize("verb", sorted(_EPHEMERALS))
    async def test_another_organizations_room_is_not_found_and_nothing_is_published(
        self, verb: str
    ) -> None:
        realtime = _RecordingRealtime()
        kit = RoomKit(realtime=realtime)
        await kit.create_room(room_id="globex-room", organization_id="globex")

        with pytest.raises(RoomNotFoundError):
            await _EPHEMERALS[verb](kit, "acme")

        assert realtime.reached == []
        await kit.close()

    @pytest.mark.parametrize("organization_id", ["globex", None], ids=["own", "unscoped"])
    @pytest.mark.parametrize("verb", sorted(_EPHEMERALS))
    async def test_the_rooms_own_organization_or_none_reaches_the_room(
        self, verb: str, organization_id: str | None
    ) -> None:
        realtime = _RecordingRealtime()
        kit = RoomKit(realtime=realtime)
        await kit.create_room(room_id="globex-room", organization_id="globex")

        await _EPHEMERALS[verb](kit, organization_id)

        assert realtime.reached == ["room:globex-room"]
        await kit.close()

    async def test_a_socket_is_not_connected_to_another_organizations_room(self) -> None:
        kit = await _two_tenants()
        ws = WebSocketChannel("ws")
        kit.register_channel(ws)

        with pytest.raises(RoomNotFoundError):
            await kit.connect_websocket(
                "ws", "c1", _send_nothing, room_id="globex-room", organization_id="acme"
            )

        assert (ws.connection_count, ws.rooms_for("c1")) == (0, set())
        await kit.connect_websocket(
            "ws", "c1", _send_nothing, room_id="globex-room", organization_id="globex"
        )
        await kit.connect_websocket("ws", "c2", _send_nothing, room_id="globex-room")
        assert ws.rooms_for("c1") == ws.rooms_for("c2") == {"globex-room"}
        await kit.close()


async def _send_nothing(connection_id: str, event: Any) -> None:
    return None
