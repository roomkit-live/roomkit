"""Tests for the Supervisor orchestration strategy."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import HookExecution, HookResult, HookTrigger
from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel
from tests.tool_room import room_tool_names, tool_call_in

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
    kit._closed = False
    kit.get_room = AsyncMock(return_value=room)
    kit.store.update_room = AsyncMock()
    kit.store.patch_room_metadata = AsyncMock(return_value=room)
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    kit.register_channel = MagicMock()
    # Pass 1's rows ride the room's lane, whose cascade the turn finishes.
    kit._max_chain_depth = 5
    kit._finish_cascade = AsyncMock(return_value=(None, None))
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

        # Declared in the room it was installed in (RFC §19.7), not channel-wide.
        tool_names = room_tool_names(boss, "r1")
        assert tool_names == ["delegate_to_w1", "delegate_to_w2"]
        assert boss.extra_tools == []

    async def test_sets_initial_state(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        updated_room = Room(id="saved", metadata=kit.store.patch_room_metadata.call_args[0][1])
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
        with tool_call_in("r1"):
            result = await boss._channel_tool_handler("delegate_to_w1", {"task": "Do something"})
        parsed = json.loads(result)

        assert parsed["status"] == "delegated"
        assert parsed["worker"] == "w1"
        await until(lambda: kit.delegate.called)  # in the background run
        kit.delegate.assert_called_once()

    async def test_unknown_tool_falls_through(self):
        """A tool nothing serves reaches no delegation: it is unserved, which
        ON_TOOL_CALL's hooks may still serve."""
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=[_make_agent("w1")])
        await s.install(kit, "r1")

        with tool_call_in("r1"), pytest.raises(UnservedToolCallError):
            await boss._channel_tool_handler("unknown_tool", {})

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

        for room_id in ("r1", "r2"):
            assert room_tool_names(boss, room_id) == ["delegate_to_w1"]


class TestSupervisorShareChannels:
    """Tests for the share_channels parameter."""

    async def test_per_worker_tool_passes_share_channels(self):
        """Per-worker delegation tools pass share_channels to kit.delegate()."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            wait_for_result=False,
            share_channels=["system", "ws-status"],
        )
        await s.install(kit, "r1")

        with tool_call_in("r1"):
            await boss._channel_tool_handler("delegate_to_w1", {"task": "Do something"})
        await until(lambda: kit.delegate.called)  # in the background run, when not waited

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system", "ws-status"]

    async def test_per_worker_tool_inline_passes_share_channels(self):
        """Inline (wait=True) per-worker tools pass share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        mock_task.result = MagicMock(status="completed", output="done", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            wait_for_result=True,
            share_channels=["email-out"],
        )
        await s.install(kit, "r1")

        with tool_call_in("r1"):
            await boss._channel_tool_handler("delegate_to_w1", {"task": "Do something"})
        await until(lambda: kit.delegate.called)  # in the background run, when not waited

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["email-out"]

    async def test_strategy_sequential_passes_share_channels(self):
        """Strategy-based sequential delegation passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="sequential",
            share_channels=["system"],
        )
        await s.install(kit, "r1")

        with tool_call_in("r1"):
            await boss._channel_tool_handler("delegate_workers", {"task": "Analyze this"})

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system"]

    async def test_strategy_parallel_passes_share_channels(self):
        """Strategy-based parallel delegation passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        w2 = _make_agent("w2")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1, w2],
            strategy="parallel",
            share_channels=["ws-status"],
        )
        await s.install(kit, "r1")

        with tool_call_in("r1"):
            await boss._channel_tool_handler("delegate_workers", {"task": "Analyze this"})

        assert kit.delegate.call_count == 2
        for call in kit.delegate.call_args_list:
            _, kwargs = call
            assert kwargs["share_channels"] == ["ws-status"]

    async def test_default_share_channels_is_empty(self):
        """Without share_channels, kit.delegate() receives empty list."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=False)
        await s.install(kit, "r1")

        with tool_call_in("r1"):
            await boss._channel_tool_handler("delegate_to_w1", {"task": "Do something"})
        await until(lambda: kit.delegate.called)  # in the background run, when not waited

        _, kwargs = kit.delegate.call_args
        assert not kwargs["share_channels"]

    async def test_auto_delegate_one_pass_passes_share_channels(self):
        """auto_delegate with refine_task=False passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        room = Room(id="r1")
        kit = _make_mock_kit(room)

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="worker result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="sequential",
            auto_delegate=True,
            refine_task=False,
            share_channels=["system"],
        )
        await s.install(kit, "r1")

        # Build a user message event to trigger auto-delegate
        event = RoomEvent(
            room_id="r1",
            type=EventType.MESSAGE,
            source=EventSource(channel_id="user", channel_type=ChannelType.SMS),
            content=TextContent(body="Analyze this topic"),
        )
        binding = ChannelBinding(
            channel_id="user",
            room_id="r1",
            channel_type=ChannelType.SMS,
        )
        context = RoomContext(room=room, bindings=[], recent_events=[])

        # Invoke the wrapped on_event
        await boss.on_event(event, binding, context)

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system"]

    async def test_auto_delegate_two_pass_passes_share_channels(self):
        """auto_delegate with refine_task=True passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        room = Room(id="r1")
        kit = _make_mock_kit(room)

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="worker result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="parallel",
            auto_delegate=True,
            refine_task=True,
            share_channels=["ws-status", "email-out"],
        )
        await s.install(kit, "r1")

        event = RoomEvent(
            room_id="r1",
            type=EventType.MESSAGE,
            source=EventSource(channel_id="user", channel_type=ChannelType.SMS),
            content=TextContent(body="Analyze this topic"),
        )
        binding = ChannelBinding(
            channel_id="user",
            room_id="r1",
            channel_type=ChannelType.SMS,
        )
        context = RoomContext(room=room, bindings=[], recent_events=[])

        await boss.on_event(event, binding, context)

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["ws-status", "email-out"]

    async def test_share_channels_defensive_copy(self):
        """Mutating the original list after construction has no effect."""
        channels = ["system"]
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            wait_for_result=False,
            share_channels=channels,
        )

        # Mutate the original list after construction
        channels.append("hacked")

        await s.install(kit, "r1")
        with tool_call_in("r1"):
            await boss._channel_tool_handler("delegate_to_w1", {"task": "Do something"})
        await until(lambda: kit.delegate.called)  # in the background run, when not waited

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system"]

    async def test_async_delivery_passes_share_channels(self):
        """async_delivery=True propagates share_channels to background workers."""
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)
        kit.deliver = AsyncMock()

        # Create a mock that passes isinstance check for RealtimeVoiceChannel
        mock_voice = MagicMock(spec=RealtimeVoiceChannel)
        mock_voice.channel_id = "voice"
        mock_voice._registry = ChannelRegistry("voice", list)
        kit.channels = {"voice": mock_voice}

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="sequential",
            auto_delegate=True,
            async_delivery=True,
            share_channels=["system", "ws-status"],
        )
        await s.install(kit, "r1")

        # Call the tool set up for r1, as the voice channel's session in r1 would
        entry = mock_voice._registry.lookup("delegate_workers", "r1")
        with tool_call_in("r1"):
            result = await entry.serve({"task": "Analyze"})
        parsed = json.loads(result)
        assert parsed["status"] == "dispatched"

        # Let the background task complete
        await asyncio.sleep(0.05)

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system", "ws-status"]


@pytest.mark.parametrize("refine_task", [False, True], ids=["one-pass", "two-pass"])
async def test_the_supervisors_turns_answer_at_the_depth_of_its_event(refine_task: bool) -> None:
    """RFC §8.3, §19.7.3: the worker results stand in for the event, so the
    supervisor's answer does not restart the chain at 1: every turn the boss
    runs answers an event at the depth of the one that woke it. The room
    writes the streamed answer one deeper than that event."""
    boss = _make_agent("boss")
    room = Room(id="r1")
    kit = _make_mock_kit(room)
    mock_task = MagicMock()
    mock_task.result = MagicMock(output="worker result", error=None)
    kit.delegate = AsyncMock(return_value=mock_task)
    supervisor = Supervisor(
        supervisor=boss,
        workers=[_make_agent("w1")],
        strategy="sequential",
        auto_delegate=True,
        refine_task=refine_task,
    )
    await supervisor.install(kit, "r1")
    answered: list[RoomEvent] = []
    respond = boss._respond

    async def spy(
        event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        answered.append(event)
        return await respond(event, binding, context)

    boss._respond = spy  # type: ignore[method-assign]

    output = await boss.on_event(
        RoomEvent(
            room_id="r1",
            type=EventType.MESSAGE,
            source=EventSource(channel_id="user", channel_type=ChannelType.SMS),
            content=TextContent(body="Analyze this topic"),
            chain_depth=2,
        ),
        ChannelBinding(channel_id="user", room_id="r1", channel_type=ChannelType.SMS),
        RoomContext(room=room, bindings=[], recent_events=[]),
    )

    assert output.response_stream is not None
    assert answered
    assert {e.chain_depth for e in answered} == {2}


_LOOKUP = AITool(
    name="lookup", description="look up", parameters={"type": "object", "properties": {}}
)


def _tool_round(call_id: str, text: str = "") -> AIResponse:
    return AIResponse(
        content=text,
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name="lookup", arguments={})],
    )


async def _two_pass_room(responses: list[AIResponse], **boss: Any) -> tuple[RoomKit, list[str]]:
    """A room whose boss supervises one worker in two passes; the tasks the
    workers were handed."""

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "secret-555-1234"

    supervisor = Agent(
        "boss",
        provider=MockAIProvider(streaming=True, ai_responses=responses),
        tools=[_LOOKUP],
        tool_handler=handler,
        tool_search=False,
        **boss,
    )
    worker = Agent("w1", provider=MockAIProvider(responses=["Worker analysis"]))
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(supervisor)
    kit.register_channel(worker)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "boss", category=ChannelCategory.INTELLIGENCE)
    tasks: list[str] = []
    delegate = kit.delegate

    async def spy(room_id: str, channel_id: str, task: str, **kwargs: Any) -> Any:
        tasks.append(task)
        return await delegate(room_id, channel_id, task, **kwargs)

    kit.delegate = spy  # type: ignore[method-assign]
    await Supervisor(
        supervisor=supervisor, workers=[worker], strategy="parallel", auto_delegate=True
    ).install(kit, "r1")
    return kit, tasks


async def _say(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body="Analyse"))
    )
    await asyncio.sleep(0.1)


async def test_pass_one_hands_on_its_answer_and_stores_its_calls() -> None:
    """RMK-396, RFC §19.7.3: pass 1 is read as every streamed turn is. The task the
    worker receives is its final answer, not the narration of its tool round
    glued to it, and its calls have their TOOL_CALL rows in the room."""
    kit, tasks = await _two_pass_room(
        [
            _tool_round("c1", "Let me check."),
            AIResponse(content="Analyse Anthropic"),
            AIResponse(content="Here is the analysis."),
        ]
    )

    await _say(kit)

    assert tasks == ["Analyse Anthropic"]
    rows = [
        (e.type, e.content.outcome if isinstance(e.content, ToolCallContent) else e.content.body)
        for e in await kit.store.list_events("r1")
        if e.source.channel_id == "boss"
    ]
    assert rows == [
        (EventType.TOOL_CALL_START, None),
        (EventType.TOOL_CALL_END, "served"),
        (EventType.MESSAGE, "Here is the analysis."),
    ]
    await kit.close()


async def test_pass_ones_rows_cross_the_rooms_gate() -> None:
    """Its rows are the room's: a BEFORE_BROADCAST hook rewrites them as it
    rewrites pass 2's (RMK-396)."""
    kit, _ = await _two_pass_room(
        [
            _tool_round("c1"),
            AIResponse(content="Analyse Anthropic"),
            _tool_round("c2"),
            AIResponse(content="Here is the analysis."),
        ]
    )

    @kit.hook(HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC)
    async def redact(event: RoomEvent, ctx: Any) -> HookResult:
        if event.type == EventType.TOOL_CALL_END and isinstance(event.content, ToolCallContent):
            content = event.content.model_copy(update={"result": "[REDACTED]"})
            return HookResult.modify(event.model_copy(update={"content": content}))
        return HookResult.allow()

    await _say(kit)

    ends = [
        (e.content.tool_id, e.content.result)
        for e in await kit.store.list_events("r1")
        if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
    ]
    assert ends == [("c1", "[REDACTED]"), ("c2", "[REDACTED]")]
    await kit.close()


@pytest.mark.parametrize("narration", ["Still checking.", ""], ids=["narrated", "silent"])
async def test_pass_one_cut_by_its_round_cap_hands_on_no_task(
    narration: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn cut short has no answer; its narration is no task (RFC §6.4)."""
    kit, tasks = await _two_pass_room(
        [_tool_round("c1", narration), _tool_round("c2", narration)],
        max_tool_rounds=1,
    )

    await _say(kit)

    assert tasks == []
    assert "ended max_rounds: no answer to hand on" in caplog.text
    await kit.close()
