"""An executor serves the organizations it declares (RFC §10.6.7, §17.2).

The conformance scenarios ``executors/an-executor-takes-the-turns-of-the-
organizations-it-serves-alone``, ``executors/an-executor-serving-every-
organization-takes-every-room-s-turns`` and ``executors/an-agent-is-offline-
in-a-room-of-an-organization-its-executors-do-not-serve`` cover placement and
presence; these cover the declaration, and the executor's own refusal of work
a defective placement sends it.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import pytest

from roomkit import (
    Agent,
    AgentIdentity,
    ChannelCategory,
    ChannelDeclaration,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from roomkit.turns._organizations import RoomNeeds
from roomkit.turns.wire import CommittedEvent, RoomAttached
from tests.conftest import make_event
from tests.test_framework import SimpleChannel


@pytest.mark.parametrize(
    ("organizations", "served", "refused"),
    [
        (None, [None], ["acme"]),
        ({"acme", "globex"}, ["acme", "globex"], [None, "initech"]),
        ({"*"}, [None, "acme"], []),
    ],
    ids=["none", "exact-ids", "every"],
)
def test_an_executor_serves_the_organizations_it_declares(
    organizations: Collection[str] | None, served: list[str | None], refused: list[str | None]
) -> None:
    executor = TurnExecutor(lambda _: None, hosts={"support"}, organizations=organizations)

    assert executor.organizations == (None if organizations is None else frozenset(organizations))
    assert [executor.takes("support", organization_id=o) for o in served] == [True] * len(served)
    assert [executor.takes("support", organization_id=o) for o in refused] == [False] * len(
        refused
    )


@pytest.mark.parametrize(
    ("organizations", "error"),
    [("acme", TypeError), ((), ValueError), ({"acme-*"}, ValueError), ({"ac*me"}, ValueError)],
    ids=["one-string", "empty", "prefix", "star-inside"],
)
def test_an_executor_refuses_a_declaration_that_is_not_exact_ids(
    organizations: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        TurnExecutor(lambda _: None, organizations=organizations)


class _Careless(InMemoryTurnBackend):
    """A backend that places every turn and channel call on one executor,
    whatever the room's organization: a defect the executor's own check
    catches."""

    def __init__(self, executor: TurnExecutor) -> None:
        super().__init__()
        self._only = executor

    def _place(self, channel_id: str, room_id: str, needs: RoomNeeds) -> TurnExecutor:
        return self._only


async def _careless_kit() -> tuple[RoomKit, list[str]]:
    """A kit whose ``support`` runs on an executor serving ``acme`` alone,
    placed even in a ``globex`` room; the channels the factory built."""
    built: list[str] = []

    def build(channel_id: str) -> Agent:
        built.append(channel_id)
        return Agent(channel_id, provider=MockAIProvider(responses=["Leaked."]))

    pool = TurnExecutor(build, hosts={"support"}, organizations={"acme"}, executor_id="acme-pool")
    backend = _Careless(pool)
    await backend.serve(pool)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(SimpleChannel("ws"))
    kit.register_channel(ChannelDeclaration(channel_id="support", identity=AgentIdentity()))
    await kit.create_room(room_id="r-globex", organization_id="globex")
    await kit.attach_channel("r-globex", "ws")
    await kit.attach_channel("r-globex", "support", category=ChannelCategory.INTELLIGENCE)
    return kit, built


async def test_an_executor_refuses_a_turn_and_a_call_of_an_organization_it_does_not_serve() -> (
    None
):
    kit, built = await _careless_kit()
    try:
        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="Hello.")),
            room_id="r-globex",
        )
        with pytest.raises(TurnLinkError) as call:
            await kit.channel_call("support", "r-globex", "set_config_option")
        timeline = await kit.store.list_events("r-globex")
    finally:
        await kit.close()

    assert call.value.reason == "no_executor"
    assert "globex" in str(call.value)
    assert not [e for e in timeline if e.source.channel_id == "support"]
    assert built == []  # nothing built for a room it does not serve


async def test_an_executor_drops_an_attach_and_an_event_of_an_organization_it_does_not_serve() -> (
    None
):
    built: list[str] = []
    pool = TurnExecutor(
        lambda channel_id: built.append(channel_id), hosts={"support"}, organizations={"acme"}
    )
    binding = ChannelBinding(channel_id="support", room_id="r-globex", channel_type=ChannelType.AI)

    attached = RoomAttached(
        channel_id="support", room_id="r-globex", binding=binding, organization_id="globex"
    )
    event = CommittedEvent(
        channel_id="support",
        room_id="r-globex",
        event=make_event(room_id="r-globex"),
        organization_id="globex",
    )

    assert await pool.control(attached) is None
    assert await pool.control(event) is None
    assert built == []
    await pool.close()


class _Listening(TurnExecutor):
    """An executor recording each channel control message it is sent."""

    def __init__(self, organizations: set[str]) -> None:
        super().__init__(lambda _: None, hosts={"support"}, organizations=organizations)
        self.told: list[str] = []

    async def control(self, message: Any) -> Any:
        self.told.append(f"{message.type}:{getattr(message, 'room_id', '')}")
        return await super().control(message)


async def test_a_room_s_attach_and_detach_reach_only_the_executors_of_its_organization() -> None:
    """An executor of another organization never learns of the room; a
    release, the channel's in all its rooms, reaches it (RFC §10.6.7)."""
    acme, globex = _Listening({"acme"}), _Listening({"globex"})
    backend = InMemoryTurnBackend()
    kit = RoomKit(turn_backend=backend)
    for executor in (acme, globex):
        await backend.serve(executor)
    kit.register_channel(ChannelDeclaration(channel_id="support", identity=AgentIdentity()))
    await kit.create_room(room_id="r-globex", organization_id="globex")
    try:
        await kit.attach_channel("r-globex", "support", category=ChannelCategory.INTELLIGENCE)
        await kit.detach_channel("r-globex", "support")
        await kit.release_channel("support")
    finally:
        await kit.close()
        for executor in (acme, globex):
            await executor.close()

    assert acme.told == ["channel_released:"]
    assert globex.told == ["room_attached:r-globex", "room_detached:r-globex", "channel_released:"]
