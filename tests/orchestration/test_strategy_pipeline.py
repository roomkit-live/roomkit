"""Tests for the Pipeline orchestration strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.pipeline import Pipeline
from roomkit.providers.ai.mock import MockAIProvider
from tests.tool_room import room_tool_names

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


def _make_mock_kit(room: Room, bindings: list[ChannelBinding]) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.list_bindings = AsyncMock(return_value=bindings)
    kit.store.update_room = AsyncMock()
    kit.store.patch_room_metadata = AsyncMock(return_value=room)
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    return kit


# -- Tests --------------------------------------------------------------------


class TestPipelineAgents:
    def test_agents_returns_all(self):
        agents = [_make_agent("a"), _make_agent("b"), _make_agent("c")]
        p = Pipeline(agents=agents)
        assert [a.channel_id for a in p.agents()] == ["a", "b", "c"]

    def test_agents_returns_copy(self):
        agents = [_make_agent("a")]
        p = Pipeline(agents=agents)
        assert p.agents() is not agents


class TestPipelineBuildStages:
    def test_linear_chain(self):
        agents = [_make_agent("a"), _make_agent("b"), _make_agent("c")]
        p = Pipeline(agents=agents)
        stages = p._build_stages()

        assert len(stages) == 3
        assert stages[0].phase == "a"
        assert stages[0].next == "b"
        assert stages[1].phase == "b"
        assert stages[1].next == "c"
        assert stages[2].phase == "c"
        assert stages[2].next is None

    def test_single_agent(self):
        agents = [_make_agent("solo")]
        p = Pipeline(agents=agents)
        stages = p._build_stages()

        assert len(stages) == 1
        assert stages[0].next is None


class TestPipelineInstall:
    async def test_installs_router_hook(self):
        agents = [_make_agent("triage"), _make_agent("support")]
        room = Room(id="r1")
        bindings = [_ai_binding("triage"), _ai_binding("support")]
        kit = _make_mock_kit(room, bindings)

        p = Pipeline(agents=agents)
        await p.install(kit, "r1")

        kit.hook_engine.add_room_hook.assert_called_once()
        call_args = kit.hook_engine.add_room_hook.call_args
        assert call_args[0][0] == "r1"

    async def test_sets_initial_state(self):
        agents = [_make_agent("triage"), _make_agent("support")]
        room = Room(id="r1")
        bindings = [_ai_binding("triage"), _ai_binding("support")]
        kit = _make_mock_kit(room, bindings)

        p = Pipeline(agents=agents)
        await p.install(kit, "r1")

        # Verify the conversation state was saved
        kit.store.patch_room_metadata.assert_called_once()
        updated_room = Room(id="saved", metadata=kit.store.patch_room_metadata.call_args[0][1])
        state = get_conversation_state(updated_room)
        assert state.active_agent_id == "triage"
        assert state.phase == "triage"

    async def test_wires_handoff_tools(self):
        agents = [_make_agent("a", "Agent A"), _make_agent("b", "Agent B")]
        room = Room(id="r1")
        bindings = [_ai_binding("a"), _ai_binding("b")]
        kit = _make_mock_kit(room, bindings)

        p = Pipeline(agents=agents)
        await p.install(kit, "r1")

        # Agent A should have handoff tool targeting B, in the installed room
        assert "handoff_conversation" in room_tool_names(agents[0], "r1")

        # Agent B is the last stage — handoff tool with no targets
        assert "handoff_conversation" in room_tool_names(agents[1], "r1")

    async def test_double_install_skips_handoff(self):
        """Shared Agent instances get one handoff per installed room (RFC §19.7)."""
        agents = [_make_agent("a"), _make_agent("b")]
        room = Room(id="r1")
        bindings = [_ai_binding("a"), _ai_binding("b")]
        kit = _make_mock_kit(room, bindings)

        p = Pipeline(agents=agents)
        await p.install(kit, "r1")
        # Second install with same agent instances
        kit2 = _make_mock_kit(room, bindings)
        await p.install(kit2, "r2")

        # Exactly one handoff tool per agent in each room
        for agent in agents:
            for room_id in ("r1", "r2"):
                assert room_tool_names(agent, room_id).count("handoff_conversation") == 1
