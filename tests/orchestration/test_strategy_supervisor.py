"""Tests for the Supervisor orchestration strategy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.supervisor import Supervisor
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


def _make_mock_kit(room: Room) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.update_room = AsyncMock()
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    kit.register_channel = MagicMock()
    return kit


# -- Tests --------------------------------------------------------------------


class TestSupervisorAgents:
    def test_agents_returns_only_supervisor(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1"), _make_agent("w2")]
        s = Supervisor(supervisor=boss, workers=workers)

        result = s.agents()
        assert len(result) == 1
        assert result[0].channel_id == "boss"


class TestSupervisorInstall:
    async def test_installs_router_hook(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        kit.hook_engine.add_room_hook.assert_called_once()

    async def test_registers_workers_on_kit(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1"), _make_agent("w2")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        # Workers should be registered via register_channel
        assert kit.register_channel.call_count == 2

    async def test_skips_already_registered_workers(self):
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.channels = {"w1": w1}  # Already registered

        s = Supervisor(supervisor=boss, workers=[w1])
        await s.install(kit, "r1")

        kit.register_channel.assert_not_called()

    async def test_injects_delegation_tools(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1", "Worker 1"), _make_agent("w2", "Worker 2")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        tool_names = [t.name for t in boss._injected_tools]
        assert "delegate_to_w1" in tool_names
        assert "delegate_to_w2" in tool_names

    async def test_sets_initial_state(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.active_agent_id == "boss"
        assert state.phase == "supervisor"

    async def test_delegation_tool_handler(self):
        """Test that the delegation tool handler calls kit.delegate."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-123"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=False)
        await s.install(kit, "r1")

        # Call the delegation tool handler
        result = await boss.tool_handler("delegate_to_w1", {"task": "Do something"})
        parsed = json.loads(result)

        assert parsed["status"] == "delegated"
        assert parsed["worker"] == "w1"
        kit.delegate.assert_called_once()

    async def test_unknown_tool_falls_through(self):
        """Non-delegation tools should fall through to original handler."""
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=[_make_agent("w1")])
        await s.install(kit, "r1")

        result = await boss.tool_handler("unknown_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed

    async def test_double_install_skips_tools(self):
        """Second install should not duplicate delegation tools."""
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))
        mock_task = MagicMock()
        mock_task.task_id = "t1"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")
        kit2 = _make_mock_kit(Room(id="r2"))
        kit2.delegate = AsyncMock(return_value=mock_task)
        await s.install(kit2, "r2")

        tool_count = sum(1 for t in boss._injected_tools if t.name == "delegate_to_w1")
        assert tool_count == 1
