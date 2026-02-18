"""Tests for ConversationPipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.orchestration.handoff import HandoffHandler
from roomkit.orchestration.pipeline import (
    ConversationPipeline,
    PipelineStage,
)
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.state import ConversationState
from tests.conftest import make_event

# -- Helpers ------------------------------------------------------------------


def _ai_binding(channel_id: str, room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _transport_binding(channel_id: str, room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.SMS,
    )


# -- Validation ---------------------------------------------------------------


class TestPipelineValidation:
    def test_valid_pipeline(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ]
        )
        assert len(pipeline.stages) == 2

    def test_invalid_next_phase(self):
        with pytest.raises(ValueError, match="not a valid phase"):
            ConversationPipeline(
                stages=[
                    PipelineStage(phase="a", agent_id="agent-a", next="nonexistent"),
                ]
            )

    def test_invalid_can_return_to(self):
        with pytest.raises(ValueError, match="not a valid phase"):
            ConversationPipeline(
                stages=[
                    PipelineStage(
                        phase="a",
                        agent_id="agent-a",
                        can_return_to={"nonexistent"},
                    ),
                ]
            )

    def test_empty_pipeline(self):
        pipeline = ConversationPipeline(stages=[])
        assert pipeline.stages == []

    def test_self_referencing_next_is_valid(self):
        # A stage can loop back to itself
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="loop", agent_id="agent-a", next="loop"),
            ]
        )
        assert len(pipeline.stages) == 1


# -- to_router ----------------------------------------------------------------


class TestToRouter:
    def test_generates_rules_per_stage(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="analysis", agent_id="agent-discuss", next="coding"),
                PipelineStage(phase="coding", agent_id="agent-coder", next="review"),
                PipelineStage(phase="review", agent_id="agent-reviewer", next=None),
            ]
        )
        router = pipeline.to_router()

        ctx = RoomContext(
            room=Room(id="r1"),
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-discuss"),
                _ai_binding("agent-coder"),
                _ai_binding("agent-reviewer"),
            ],
        )
        event = make_event(room_id="r1", channel_id="sms1")

        # Analysis phase routes to discuss agent
        assert (
            router.select_agent(event, ctx, ConversationState(phase="analysis")) == "agent-discuss"
        )

        # Coding phase routes to coder
        assert router.select_agent(event, ctx, ConversationState(phase="coding")) == "agent-coder"

        # Review phase routes to reviewer
        assert (
            router.select_agent(event, ctx, ConversationState(phase="review")) == "agent-reviewer"
        )

    def test_default_agent_is_first_stage(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="intake", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-handler", next=None),
            ]
        )
        router = pipeline.to_router()

        ctx = RoomContext(
            room=Room(id="r1"),
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-triage"),
                _ai_binding("agent-handler"),
            ],
        )
        event = make_event(room_id="r1", channel_id="sms1")

        # Unknown phase falls to default (first stage's agent)
        assert (
            router.select_agent(event, ctx, ConversationState(phase="unknown")) == "agent-triage"
        )

    def test_custom_default_phase(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
            default_phase="b",
        )
        router = pipeline.to_router()

        ctx = RoomContext(
            room=Room(id="r1"),
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-a"),
                _ai_binding("agent-b"),
            ],
        )
        event = make_event(room_id="r1", channel_id="sms1")

        # Default falls to agent-b (default_phase="b")
        assert router.select_agent(event, ctx, ConversationState(phase="unknown")) == "agent-b"

    def test_supervisor_propagated(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next=None),
            ],
            supervisor_id="agent-sup",
        )
        router = pipeline.to_router()
        assert router._supervisor_id == "agent-sup"


# -- get_phase_map ------------------------------------------------------------


class TestGetPhaseMap:
    def test_maps_agents_to_phases(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="analysis", agent_id="agent-discuss", next="coding"),
                PipelineStage(phase="coding", agent_id="agent-coder", next=None),
            ]
        )
        phase_map = pipeline.get_phase_map()
        assert phase_map == {
            "agent-discuss": "analysis",
            "agent-coder": "coding",
        }


# -- get_allowed_transitions --------------------------------------------------


class TestGetAllowedTransitions:
    def test_linear_pipeline(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next="c"),
                PipelineStage(phase="c", agent_id="agent-c", next=None),
            ]
        )
        transitions = pipeline.get_allowed_transitions()
        assert transitions == {
            "a": {"b"},
            "b": {"c"},
            "c": set(),
        }

    def test_pipeline_with_loop(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="coding", agent_id="agent-coder", next="review"),
                PipelineStage(
                    phase="review",
                    agent_id="agent-reviewer",
                    next="report",
                    can_return_to={"coding"},
                ),
                PipelineStage(phase="report", agent_id="agent-writer", next=None),
            ]
        )
        transitions = pipeline.get_allowed_transitions()
        assert transitions == {
            "coding": {"review"},
            "review": {"report", "coding"},
            "report": set(),
        }

    def test_terminal_stage(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="done", agent_id="agent-a", next=None),
            ]
        )
        transitions = pipeline.get_allowed_transitions()
        assert transitions == {"done": set()}


# -- PipelineStage model ------------------------------------------------------


class TestPipelineStage:
    def test_defaults(self):
        stage = PipelineStage(phase="a", agent_id="agent-a")
        assert stage.next is None
        assert stage.can_return_to == set()

    def test_can_return_to_set(self):
        stage = PipelineStage(
            phase="review",
            agent_id="agent-reviewer",
            next="report",
            can_return_to={"coding", "analysis"},
        )
        assert stage.can_return_to == {"coding", "analysis"}


# -- install ------------------------------------------------------------------


def _mock_agent(channel_id: str) -> MagicMock:
    agent = MagicMock()
    agent.channel_id = channel_id
    agent._extra_tools = []
    agent._tool_handler = None
    return agent


def _mock_kit() -> MagicMock:
    kit = MagicMock()
    # kit.hook(trigger, ...)(fn) — returns the decorator then calls it
    hook_decorator = MagicMock()
    kit.hook.return_value = hook_decorator
    return kit


class TestPipelineInstall:
    def test_install_registers_hook(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit = _mock_kit()
        agents = [_mock_agent("agent-a"), _mock_agent("agent-b")]

        pipeline.install(kit, agents)

        kit.hook.assert_called_once_with(
            HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            priority=-100,
        )

    def test_install_wires_handoff_on_all_agents(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit = _mock_kit()
        agents = [_mock_agent("agent-a"), _mock_agent("agent-b")]

        pipeline.install(kit, agents)

        # Each agent should have the handoff tool injected
        for agent in agents:
            assert len(agent._extra_tools) == 1
            assert agent._extra_tools[0].name == "handoff_conversation"

    def test_install_returns_router_and_handler(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next=None),
            ],
        )
        kit = _mock_kit()
        agents = [_mock_agent("agent-a")]

        router, handler = pipeline.install(kit, agents)

        assert isinstance(router, ConversationRouter)
        assert isinstance(handler, HandoffHandler)

    def test_install_passes_aliases(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next=None),
            ],
        )
        kit = _mock_kit()
        agents = [_mock_agent("agent-a")]

        _, handler = pipeline.install(kit, agents, agent_aliases={"alias-a": "agent-a"})

        assert handler._aliases == {"alias-a": "agent-a"}

    def test_install_enforces_transitions(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit = _mock_kit()
        agents = [_mock_agent("agent-a"), _mock_agent("agent-b")]

        _, handler = pipeline.install(kit, agents)

        assert handler._allowed_transitions == {"a": {"b"}, "b": set()}

    def test_install_custom_hook_priority(self):
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next=None),
            ],
        )
        kit = _mock_kit()
        agents = [_mock_agent("agent-a")]

        pipeline.install(kit, agents, hook_priority=-50)

        kit.hook.assert_called_once_with(
            HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            priority=-50,
        )


# -- greet_on_handoff ---------------------------------------------------------


def _mock_kit_capturing_hooks() -> tuple[MagicMock, dict[tuple[str, str], list]]:
    """Return a mock kit that captures registered hook functions by trigger."""
    kit = MagicMock()
    captured: dict[tuple[str, str], list] = {}

    def _hook_side_effect(*args, **kwargs):
        trigger = args[0] if args else kwargs.get("trigger")
        execution = kwargs.get("execution", HookExecution.ASYNC)

        def decorator(fn):
            key = (str(trigger), str(execution))
            captured.setdefault(key, []).append(fn)
            return fn

        return decorator

    kit.hook.side_effect = _hook_side_effect
    return kit, captured


class TestGreetOnHandoff:
    def test_greet_on_handoff_requires_voice_channel_id(self):
        pipeline = ConversationPipeline(
            stages=[PipelineStage(phase="a", agent_id="agent-a", next=None)],
        )
        kit, _ = _mock_kit_capturing_hooks()
        agents = [_mock_agent("agent-a")]

        with pytest.raises(ValueError, match="voice_channel_id"):
            pipeline.install(kit, agents, greet_on_handoff=True)

    def test_greet_on_handoff_registers_hooks(self):
        pipeline = ConversationPipeline(
            stages=[PipelineStage(phase="a", agent_id="agent-a", next=None)],
        )
        kit, captured = _mock_kit_capturing_hooks()
        agents = [_mock_agent("agent-a")]

        pipeline.install(kit, agents, greet_on_handoff=True, voice_channel_id="voice")

        # Should have ON_HANDOFF (ASYNC) and BEFORE_TTS (SYNC) hooks
        on_handoff_key = (str(HookTrigger.ON_HANDOFF), str(HookExecution.ASYNC))
        before_tts_key = (str(HookTrigger.BEFORE_TTS), str(HookExecution.SYNC))

        assert on_handoff_key in captured, f"ON_HANDOFF not registered: {captured.keys()}"
        assert before_tts_key in captured, f"BEFORE_TTS not registered: {captured.keys()}"

    async def test_greet_blocks_farewell_then_allows(self):
        """BEFORE_TTS blocks during handoff, allows after greeting clears flag."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit, captured = _mock_kit_capturing_hooks()
        kit.process_inbound = AsyncMock()
        agents = [_mock_agent("agent-a"), _mock_agent("agent-b")]

        pipeline.install(kit, agents, greet_on_handoff=True, voice_channel_id="voice")

        on_handoff_key = (str(HookTrigger.ON_HANDOFF), str(HookExecution.ASYNC))
        before_tts_key = (str(HookTrigger.BEFORE_TTS), str(HookExecution.SYNC))

        on_handoff_fn = captured[on_handoff_key][0]
        before_tts_fn = captured[before_tts_key][0]

        # Simulate handoff event
        handoff_event = RoomEvent(
            room_id="room-1",
            source=EventSource(channel_id="agent-a", channel_type=ChannelType.AI),
            content=TextContent(body="transferring"),
            metadata={"from_agent": "agent-a", "to_agent": "agent-b"},
        )
        ctx = RoomContext(room=Room(id="room-1"), bindings=[])

        # Fire ON_HANDOFF — sets flag
        await on_handoff_fn(handoff_event, ctx)

        # BEFORE_TTS should block (farewell from old agent)
        result = await before_tts_fn("Goodbye!", ctx)
        assert isinstance(result, HookResult)
        assert result.action == "block"

        # Let background task run (clears flag + triggers greeting)
        await asyncio.sleep(0.05)

        # BEFORE_TTS should now allow
        result = await before_tts_fn("Hello, I'm agent B", ctx)
        assert isinstance(result, HookResult)
        assert result.action == "allow"

    async def test_greet_triggers_process_inbound(self):
        """ON_HANDOFF should trigger kit.process_inbound with voice_channel_id."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit, captured = _mock_kit_capturing_hooks()
        kit.process_inbound = AsyncMock()
        agents = [_mock_agent("agent-a"), _mock_agent("agent-b")]

        pipeline.install(kit, agents, greet_on_handoff=True, voice_channel_id="voice")

        on_handoff_key = (str(HookTrigger.ON_HANDOFF), str(HookExecution.ASYNC))
        on_handoff_fn = captured[on_handoff_key][0]

        handoff_event = RoomEvent(
            room_id="room-1",
            source=EventSource(channel_id="agent-a", channel_type=ChannelType.AI),
            content=TextContent(body="transferring"),
            metadata={"from_agent": "agent-a", "to_agent": "agent-b"},
        )
        ctx = RoomContext(room=Room(id="room-1"), bindings=[])

        await on_handoff_fn(handoff_event, ctx)
        await asyncio.sleep(0.05)

        kit.process_inbound.assert_called_once()
        call_args = kit.process_inbound.call_args
        msg = call_args[0][0]
        assert msg.channel_id == "voice"
        assert msg.sender_id == "system"
        assert call_args[1]["room_id"] == "room-1"

    def test_greet_on_handoff_false_no_extra_hooks(self):
        """Without greet_on_handoff, only BEFORE_BROADCAST hook is registered."""
        pipeline = ConversationPipeline(
            stages=[PipelineStage(phase="a", agent_id="agent-a", next=None)],
        )
        kit, captured = _mock_kit_capturing_hooks()
        agents = [_mock_agent("agent-a")]

        pipeline.install(kit, agents)

        # Only BEFORE_BROADCAST should be registered
        assert len(captured) == 1
        bb_key = (str(HookTrigger.BEFORE_BROADCAST), str(HookExecution.SYNC))
        assert bb_key in captured
