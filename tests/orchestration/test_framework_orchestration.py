"""Tests for orchestration integration with the RoomKit framework."""

from __future__ import annotations

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.pipeline import Pipeline
from roomkit.orchestration.strategies.swarm import Swarm
from roomkit.providers.ai.mock import MockAIProvider

# -- Helpers ------------------------------------------------------------------


def _make_agent(channel_id: str, description: str | None = None) -> Agent:
    return Agent(
        channel_id=channel_id,
        provider=MockAIProvider(responses=["ok"]),
        description=description,
    )


# -- Tests --------------------------------------------------------------------


class TestCreateRoomOrchestration:
    async def test_kit_level_orchestration(self):
        """orchestration= on RoomKit applies to all rooms."""
        agents = [_make_agent("a"), _make_agent("b")]
        orch = Pipeline(agents=agents)
        kit = RoomKit(orchestration=orch)

        room = await kit.create_room()

        # Agents should be registered and attached
        assert "a" in kit.channels
        assert "b" in kit.channels
        bindings = await kit.store.list_bindings(room.id)
        bound_ids = {b.channel_id for b in bindings}
        assert "a" in bound_ids
        assert "b" in bound_ids

        # Conversation state should be set
        room = await kit.get_room(room.id)
        state = get_conversation_state(room)
        assert state.active_agent_id == "a"

        await kit.close()

    async def test_per_room_orchestration(self):
        """orchestration= on create_room overrides kit default."""
        kit = RoomKit()
        agents = [_make_agent("x"), _make_agent("y")]
        orch = Pipeline(agents=agents)

        room = await kit.create_room(orchestration=orch)

        assert "x" in kit.channels
        bindings = await kit.store.list_bindings(room.id)
        bound_ids = {b.channel_id for b in bindings}
        assert "x" in bound_ids
        assert "y" in bound_ids

        await kit.close()

    async def test_per_room_none_disables_kit_default(self):
        """Passing orchestration=None explicitly disables kit default."""
        agents = [_make_agent("a1"), _make_agent("b1")]
        orch = Pipeline(agents=agents)
        kit = RoomKit(orchestration=orch)

        room = await kit.create_room(orchestration=None)

        # No orchestration should be applied — no bindings
        bindings = await kit.store.list_bindings(room.id)
        assert len(bindings) == 0

        await kit.close()

    async def test_no_orchestration_is_noop(self):
        """No orchestration= means rooms are created without orchestration."""
        kit = RoomKit()
        room = await kit.create_room()

        bindings = await kit.store.list_bindings(room.id)
        assert len(bindings) == 0

        await kit.close()

    async def test_swarm_orchestration(self):
        """Swarm strategy works end-to-end through create_room."""
        agents = [_make_agent("s1"), _make_agent("s2")]
        orch = Swarm(agents=agents, entry="s2")
        kit = RoomKit(orchestration=orch)

        room = await kit.create_room()

        room = await kit.get_room(room.id)
        state = get_conversation_state(room)
        assert state.active_agent_id == "s2"
        assert state.phase == "swarm"

        await kit.close()

    async def test_agents_registered_once(self):
        """Agents should not be re-registered if already on the kit."""
        a = _make_agent("shared")
        kit = RoomKit()
        kit.register_channel(a)

        orch = Pipeline(agents=[a, _make_agent("other")])
        await kit.create_room(orchestration=orch)

        # Should not raise — agent was already registered
        assert "shared" in kit.channels

        await kit.close()
