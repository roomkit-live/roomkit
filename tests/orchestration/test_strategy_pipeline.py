"""Tests for the Pipeline orchestration strategy."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import RoomKit
from roomkit.channels import WebSocketChannel
from roomkit.channels.agent import Agent
from roomkit.channels.voice import VoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration._handoff_greeting import HandoffGreeting
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.pipeline import Pipeline, _build_stages
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns._registry_tools import OwnerToolStore
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
    kit.claim_room_strategy = AsyncMock(return_value=None)
    kit.get_room = AsyncMock(return_value=room)
    kit.store.list_bindings = AsyncMock(return_value=bindings)
    kit.store.update_room = AsyncMock()
    kit.store.patch_room_metadata = AsyncMock(return_value=room)
    kit.hook_engine = MagicMock()
    kit.hook_engine.replace_room_hook = MagicMock()
    kit._lock_manager = MagicMock()
    kit._lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    # The kit keeps the tools the install sets up for an agent (RFC §10.6.2).
    kit._owner_tools = OwnerToolStore(kit.channels.get)
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
        stages = _build_stages(Pipeline(agents=agents).agents())

        assert len(stages) == 3
        assert stages[0].phase == "a"
        assert stages[0].next == "b"
        assert stages[1].phase == "b"
        assert stages[1].next == "c"
        assert stages[2].phase == "c"
        assert stages[2].next is None

    def test_single_agent(self):
        agents = [_make_agent("solo")]
        stages = _build_stages(Pipeline(agents=agents).agents())

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

        kit.hook_engine.replace_room_hook.assert_called_once()
        call_args = kit.hook_engine.replace_room_hook.call_args
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
        assert "handoff_conversation" in room_tool_names(kit, "a", "r1")

        # Agent B is the last stage — handoff tool with no targets
        assert "handoff_conversation" in room_tool_names(kit, "b", "r1")

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
        # The second mock stands for the same kit's other room: one store.
        kit2._owner_tools = kit._owner_tools
        await p.install(kit2, "r2")

        # Exactly one handoff tool per agent in each room
        for agent in agents:
            for room_id in ("r1", "r2"):
                names = room_tool_names(kit, agent.channel_id, room_id)
                assert names.count("handoff_conversation") == 1


async def test_a_pipeline_with_no_agent_is_refused_at_its_install() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r")

    with pytest.raises(ValueError, match="needs at least one agent"):
        await kit.install_strategy("r", Pipeline(agents=[]))

    assert kit.room_strategy("r") is None
    await kit.close()


async def test_greet_on_handoff_without_a_voice_channel_is_refused() -> None:
    """RMK-873: the refusal ConversationPipeline.install makes, at construction."""
    with pytest.raises(ValueError, match="voice_channel_id"):
        Pipeline(agents=[_make_agent("a")], greet_on_handoff=True)


async def _voice_kit(strategy: Pipeline) -> tuple[RoomKit, VoiceChannel]:
    """A kit with a voice channel and agents a and b (voices v-a, v-b), room
    r1 run by *strategy* and room r2 by none."""
    kit = RoomKit()
    voice = VoiceChannel("voice")
    kit.register_channel(voice)
    for channel_id, speaks in (("a", "v-a"), ("b", "v-b")):
        provider = MockAIProvider(responses=["x"])
        kit.register_channel(Agent(channel_id, provider=provider, voice=speaks))
    await kit.create_room(room_id="r1", orchestration=strategy)
    await kit.create_room(room_id="r2")
    return kit, voice


async def test_the_pipeline_strategy_gives_its_voice_channel_the_agents_voices() -> None:
    """RMK-873: on a voice channel, the agents' voices are its voice map."""
    kit, voice = await _voice_kit(Pipeline(agents=["a", "b"], voice_channel_id="voice"))
    await kit.close()

    assert voice._voice_map == {"a": "v-a", "b": "v-b"}


async def test_the_pipeline_strategy_greets_on_a_handoff_in_its_room_only() -> None:
    """RMK-873: a handoff in a room the strategy runs cues the new agent to
    introduce itself, through hooks of that room, never another's."""
    strategy = Pipeline(
        agents=["a", "b"],
        voice_channel_id="voice",
        greet_on_handoff=True,
        greeting_prompt="Introduce yourself.",
        farewell_prompt="Goodbye.",
    )
    kit, _ = await _voice_kit(strategy)
    cues: list[tuple[str, str, list[str] | None]] = []

    async def cue(message: InboundMessage, *, room_id: str) -> InboundResult:
        cues.append((room_id, message.content.body, message.addressed_to))
        return InboundResult()

    kit.process_inbound = cue  # type: ignore[method-assign]
    for room_id in ("r1", "r2"):
        context = RoomContext(room=await kit.get_room(room_id), bindings=[])
        await kit.hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_HANDOFF, _handoff(room_id), context
        )
    for _ in range(100):
        if cues:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    triggers = {hook.trigger for hook in kit.hook_engine._room_hooks.get("r1", [])}
    await kit.close()

    assert cues == [("r1", "Introduce yourself.", ["b"])]
    assert {HookTrigger.ON_HANDOFF, HookTrigger.BEFORE_TTS} <= triggers
    assert not kit.hook_engine._room_hooks.get("r2")


@pytest.mark.parametrize("to_agent", ["", "gone"], ids=["no-agent", "unregistered"])
async def test_a_handoff_greeting_the_room_refuses_is_logged(
    to_agent: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A cue to no agent the kit registers is refused (RFC §10.1.1 step 1):
    the handoff says so rather than going quiet."""
    kit, _ = await _voice_kit(Pipeline(agents=["a", "b"], voice_channel_id="voice"))
    greeting = HandoffGreeting(kit, "voice", "Introduce yourself.", None)

    with caplog.at_level(logging.WARNING, logger="roomkit.orchestration"):
        await greeting._greet("r1", to_agent)
    await kit.close()

    assert any(
        "r1" in r.getMessage() and "instruction_unaddressed" in r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING"
    )


async def test_a_handoff_cue_reaches_its_agent_under_the_framework_s_name() -> None:
    """The cue names its room, so the refusal of a message routed under the
    framework's name (RFC §10.1 step 2) leaves it: the new agent is asked, and
    answers in that room."""
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(Agent("b", provider=MockAIProvider(responses=["x"])))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ws")
    await kit.attach_channel("r1", "b")
    greeting = HandoffGreeting(kit, "ws", "Introduce yourself.", None)

    await greeting._greet("r1", "b")
    await kit.close()

    answers = [e for e in await kit.store.list_events("r1") if e.source.channel_id == "b"]
    assert [(e.type, e.content.body) for e in answers] == [
        (EventType.INSTRUCTION_READ, "Introduce yourself."),  # its own copy (RFC §10.1.1)
        (EventType.MESSAGE, "x"),
    ]


def _handoff(room_id: str) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="a", channel_type=ChannelType.AI),
        content=TextContent(body="handoff"),
        metadata={"from_agent": "a", "to_agent": "b"},
    )
