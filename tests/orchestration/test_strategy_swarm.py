"""Tests for the Swarm orchestration strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.swarm import Swarm
from roomkit.providers.ai.mock import MockAIProvider

# -- Helpers ------------------------------------------------------------------


class _NoopLock:
    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_agent(channel_id: str, description: str | None = None) -> Agent:
    return Agent(
        channel_id=channel_id,
        provider=MockAIProvider(responses=["ok"]),
        description=description,
    )


def _ai_binding(channel_id: str, room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _make_mock_kit(room: Room) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.update_room = AsyncMock()
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    return kit


# -- Tests --------------------------------------------------------------------


class TestSwarmAgents:
    def test_agents_returns_all(self):
        agents = [_make_agent("a"), _make_agent("b"), _make_agent("c")]
        s = Swarm(agents=agents)
        assert [a.channel_id for a in s.agents()] == ["a", "b", "c"]

    def test_default_entry_is_first_agent(self):
        agents = [_make_agent("first"), _make_agent("second")]
        s = Swarm(agents=agents)
        assert s._entry == "first"

    def test_custom_entry(self):
        agents = [_make_agent("a"), _make_agent("b")]
        s = Swarm(agents=agents, entry="b")
        assert s._entry == "b"


class TestSwarmInstall:
    async def test_installs_router_hook(self):
        agents = [_make_agent("a"), _make_agent("b")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Swarm(agents=agents)
        await s.install(kit, "r1")

        kit.hook_engine.add_room_hook.assert_called_once()

    async def test_sets_initial_state(self):
        agents = [_make_agent("a"), _make_agent("b")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Swarm(agents=agents, entry="b")
        await s.install(kit, "r1")

        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.active_agent_id == "b"
        assert state.phase == "swarm"

    async def test_bidirectional_handoff_tools(self):
        agents = [
            _make_agent("a", "A desc"),
            _make_agent("b", "B desc"),
            _make_agent("c", "C desc"),
        ]
        kit = _make_mock_kit(Room(id="r1"))

        s = Swarm(agents=agents)
        await s.install(kit, "r1")

        # Each agent should have handoff tool
        for agent in agents:
            assert any(t.name == "handoff_conversation" for t in agent._injected_tools)

        # Agent A's handoff tool should have B and C as targets
        a_tool = next(t for t in agents[0]._injected_tools if t.name == "handoff_conversation")
        target_enum = a_tool.parameters["properties"]["target"].get("enum", [])
        assert "b" in target_enum
        assert "c" in target_enum
        assert "a" not in target_enum  # Can't hand off to self

    async def test_double_install_skips_handoff(self):
        agents = [_make_agent("a"), _make_agent("b")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Swarm(agents=agents)
        await s.install(kit, "r1")
        kit2 = _make_mock_kit(Room(id="r2"))
        await s.install(kit2, "r2")

        for agent in agents:
            handoff_count = sum(
                1 for t in agent._injected_tools if t.name == "handoff_conversation"
            )
            assert handoff_count == 1
