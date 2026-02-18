"""Tests for ConversationPipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.orchestration.handoff import HANDOFF_TOOL, HandoffHandler
from roomkit.orchestration.pipeline import (
    ConversationPipeline,
    PipelineStage,
)
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.voice.base import VoiceSession, VoiceSessionState
from tests.conftest import make_event

# -- Helpers ------------------------------------------------------------------


class _NoopLock:
    """Async context manager that does nothing (replaces real room lock in tests)."""

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, *args: object) -> None:
        pass


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


def _mock_agent(channel_id: str, **kwargs) -> Agent:
    from roomkit.providers.ai.mock import MockAIProvider

    return Agent(channel_id, provider=MockAIProvider(responses=["ok"]), **kwargs)


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


# -- Agent-aware install ------------------------------------------------------


class TestAgentAwareInstall:
    def test_install_per_agent_tool_with_targets(self):
        """Agent instances get a handoff tool with enum-constrained targets."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="intake", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-advisor", next="intake"),
            ],
        )
        kit = _mock_kit()
        triage = _mock_agent("agent-triage", role="Triage", description="Routes callers")
        advisor = _mock_agent("agent-advisor", role="Advisor", description="Gives advice")

        pipeline.install(kit, [triage, advisor])

        # triage should have tool with enum=["agent-advisor"]
        assert len(triage._extra_tools) == 1
        tool = triage._extra_tools[0]
        target_prop = tool.parameters["properties"]["target"]
        assert target_prop["enum"] == ["agent-advisor"]

        # advisor should have tool with enum=["agent-triage"]
        assert len(advisor._extra_tools) == 1
        tool = advisor._extra_tools[0]
        target_prop = tool.parameters["properties"]["target"]
        assert target_prop["enum"] == ["agent-triage"]

    def test_install_uses_agent_description(self):
        """Agent.description appears in the handoff tool description."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit = _mock_kit()
        agent_a = _mock_agent("agent-a", description="Does A things")
        agent_b = _mock_agent("agent-b", description="Does B things")

        pipeline.install(kit, [agent_a, agent_b])

        # agent-a's tool should mention agent-b's description
        tool = agent_a._extra_tools[0]
        assert "Does B things" in tool.description

    def test_install_uses_stage_description_fallback(self):
        """PipelineStage.description used when Agent has no description."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(
                    phase="b",
                    agent_id="agent-b",
                    description="Stage B desc",
                ),
            ],
        )
        kit = _mock_kit()
        agent_a = _mock_agent("agent-a", role="A")
        agent_b = _mock_agent("agent-b", role="B")

        pipeline.install(kit, [agent_a, agent_b])

        # agent-a's tool should use PipelineStage.description for agent-b
        tool = agent_a._extra_tools[0]
        assert "Stage B desc" in tool.description

    def test_install_auto_wires_voice_map(self):
        """Agent.voice fields are auto-wired into VoiceChannel._voice_map."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit = _mock_kit()
        from roomkit.channels.voice import VoiceChannel

        vc = MagicMock(spec=VoiceChannel)
        vc.update_voice_map = MagicMock()
        kit._channels = {"voice": vc}

        agent_a = _mock_agent("agent-a", voice="voice-a")
        agent_b = _mock_agent("agent-b", voice="voice-b")

        pipeline.install(kit, [agent_a, agent_b], voice_channel_id="voice")

        vc.update_voice_map.assert_called_once_with(
            {
                "agent-a": "voice-a",
                "agent-b": "voice-b",
            }
        )

    def test_install_no_voice_wiring_without_agent_voices(self):
        """No voice map wiring when no Agent has voice set."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next=None),
            ],
        )
        kit = _mock_kit()
        vc = MagicMock()
        vc.update_voice_map = MagicMock()
        kit._channels = {"voice": vc}

        agent_a = _mock_agent("agent-a", role="A")

        pipeline.install(kit, [agent_a], voice_channel_id="voice")

        vc.update_voice_map.assert_not_called()

    def test_install_terminal_agent_gets_empty_enum(self):
        """Agent at terminal stage (next=None, no can_return_to) gets generic tool."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit = _mock_kit()
        agent_a = _mock_agent("agent-a", description="first")
        agent_b = _mock_agent("agent-b", description="last")

        pipeline.install(kit, [agent_a, agent_b])

        # agent-b has no reachable targets → empty targets → generic tool
        assert agent_b._extra_tools[0] is HANDOFF_TOOL


# -- Realtime (speech-to-speech) install ------------------------------------


def _make_rtv(channel_id: str = "rtv") -> RealtimeVoiceChannel:
    """Create a real RealtimeVoiceChannel with mock provider + transport."""
    from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

    return RealtimeVoiceChannel(
        channel_id,
        provider=MockRealtimeProvider(),
        transport=MockRealtimeTransport(),
    )


def _mock_kit_with_rtv(rtv_channel_id: str = "rtv"):
    """Return a mock kit that has a real RealtimeVoiceChannel registered."""
    kit = MagicMock()
    hook_decorator = MagicMock()
    kit.hook.return_value = hook_decorator
    rtv = _make_rtv(rtv_channel_id)
    kit._channels = {rtv_channel_id: rtv}
    return kit, rtv


class TestRealtimeInstall:
    def test_install_detects_realtime_channel(self):
        """install() detects RealtimeVoiceChannel and uses _wire_realtime."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        # Config-only agents (no provider)
        a = Agent("agent-a", role="A", description="Does A", voice="v-a", system_prompt="Be A.")
        b = Agent("agent-b", role="B", description="Does B", voice="v-b", system_prompt="Be B.")

        pipeline.install(kit, [a, b], voice_channel_id="rtv")

        # Agents should NOT have handoff tools injected directly
        # (they're config-only, wiring goes through the RTV tool handler)
        assert len(a._extra_tools) == 0
        assert len(b._extra_tools) == 0

        # RTV should have a tool handler installed
        assert rtv._tool_handler is not None

    def test_wire_realtime_sets_initial_config(self):
        """Initial agent's config is applied to the RTV channel."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        triage = Agent(
            "agent-triage",
            role="Triage",
            voice="v-triage",
            system_prompt="Greet callers.",
        )
        advisor = Agent(
            "agent-advisor",
            role="Advisor",
            voice="v-advisor",
            system_prompt="Give advice.",
        )

        pipeline.install(kit, [triage, advisor], voice_channel_id="rtv")

        # Initial config should be the first stage (triage)
        assert rtv._voice == "v-triage"
        assert rtv._system_prompt is not None
        assert "Greet callers." in rtv._system_prompt
        assert "Role: Triage" in rtv._system_prompt

        # Tools should contain the handoff tool
        assert rtv._tools is not None
        assert len(rtv._tools) == 1
        assert rtv._tools[0]["name"] == "handoff_conversation"

    def test_wire_realtime_per_agent_tool_has_enum(self):
        """Per-agent handoff tool has enum-constrained targets."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
                PipelineStage(
                    phase="handling",
                    agent_id="agent-advisor",
                    next="triage",
                    can_return_to={"triage"},
                ),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        triage = Agent(
            "agent-triage",
            role="Triage",
            description="Routes callers",
            voice="v-t",
            system_prompt="Hi.",
        )
        advisor = Agent(
            "agent-advisor",
            role="Advisor",
            description="Gives advice",
            voice="v-a",
            system_prompt="Help.",
        )

        pipeline.install(kit, [triage, advisor], voice_channel_id="rtv")

        # Initial tool (triage) should have enum=["agent-advisor"]
        tool = rtv._tools[0]
        assert tool["parameters"]["properties"]["target"]["enum"] == ["agent-advisor"]

    def test_wire_realtime_known_agents_on_handler(self):
        """Handler should have known_agents set for realtime mode."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        a = Agent("agent-a", role="A", system_prompt="A.")
        b = Agent("agent-b", role="B", system_prompt="B.")

        _, handler = pipeline.install(kit, [a, b], voice_channel_id="rtv")

        assert handler._known_agents == {"agent-a", "agent-b"}

    async def test_wire_realtime_tool_handler_handoff(self):
        """Tool handler intercepts handoff_conversation and calls handler."""
        from roomkit.orchestration.state import ConversationState
        from roomkit.voice.base import VoiceSession

        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        triage = Agent(
            "agent-triage",
            role="Triage",
            description="Routes callers",
            voice="v-t",
            system_prompt="Hi.",
        )
        advisor = Agent(
            "agent-advisor",
            role="Advisor",
            description="Gives advice",
            voice="v-a",
            system_prompt="Help.",
        )

        # Set up room with conversation state
        room = Room(id="r1")
        state = ConversationState(active_agent_id="agent-triage", phase="triage")
        room = set_conversation_state(room, state)

        # Mock kit methods needed by handler
        kit.get_room = AsyncMock(return_value=room)
        kit.store.list_bindings = AsyncMock(return_value=[])
        kit.store.update_room = AsyncMock()
        kit.send_event = AsyncMock()
        kit._hook_engine.run_async_hooks = AsyncMock()
        kit._lock_manager = MagicMock()
        kit._lock_manager.locked = MagicMock(return_value=_NoopLock())

        pipeline.install(kit, [triage, advisor], voice_channel_id="rtv")

        # Simulate a session in the RTV
        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="user1",
            channel_id="rtv",
        )
        rtv._sessions["s1"] = session
        rtv._session_rooms["s1"] = "r1"

        # Call the tool handler
        result = await rtv._tool_handler(
            session,
            "handoff_conversation",
            {"target": "agent-advisor", "reason": "needs help", "summary": "context"},
        )

        assert isinstance(result, dict)
        assert result["accepted"] is True
        assert result["new_agent_id"] == "agent-advisor"

    async def test_wire_realtime_reconfigures_on_handoff(self):
        """on_handoff_complete reconfigures the RTV session."""
        from roomkit.orchestration.state import ConversationState
        from roomkit.voice.base import VoiceSession

        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        triage = Agent(
            "agent-triage",
            role="Triage",
            voice="v-t",
            system_prompt="Hi.",
        )
        advisor = Agent(
            "agent-advisor",
            role="Advisor",
            voice="v-a",
            system_prompt="Help.",
        )

        room = Room(id="r1")
        state = ConversationState(active_agent_id="agent-triage", phase="triage")
        room = set_conversation_state(room, state)

        kit.get_room = AsyncMock(return_value=room)
        kit.store.list_bindings = AsyncMock(return_value=[])
        kit.store.update_room = AsyncMock()
        kit.send_event = AsyncMock()
        kit._hook_engine.run_async_hooks = AsyncMock()
        kit._lock_manager = MagicMock()
        kit._lock_manager.locked = MagicMock(return_value=_NoopLock())

        _, handler = pipeline.install(
            kit,
            [triage, advisor],
            voice_channel_id="rtv",
        )

        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="user1",
            channel_id="rtv",
        )
        session.state = VoiceSessionState.ACTIVE
        rtv._sessions["s1"] = session
        rtv._session_rooms["s1"] = "r1"

        # Trigger handoff via tool handler
        await rtv._tool_handler(
            session,
            "handoff_conversation",
            {"target": "agent-advisor", "reason": "help", "summary": "ctx"},
        )

        # Provider should have been reconfigured (disconnect + reconnect)
        provider = rtv._provider
        reconfig_calls = [c for c in provider.calls if c.method == "disconnect"]
        assert len(reconfig_calls) >= 1  # reconfigure disconnects the old session

    async def test_wire_realtime_greet_on_handoff(self):
        """When greet_on_handoff=True, tool result includes greeting message."""
        from roomkit.orchestration.state import ConversationState
        from roomkit.voice.base import VoiceSession

        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        triage = Agent(
            "agent-triage",
            role="Triage",
            voice="v-t",
            system_prompt="Hi.",
        )
        advisor = Agent(
            "agent-advisor",
            role="Advisor",
            voice="v-a",
            system_prompt="Help.",
        )

        room = Room(id="r1")
        state = ConversationState(active_agent_id="agent-triage", phase="triage")
        room = set_conversation_state(room, state)

        kit.get_room = AsyncMock(return_value=room)
        kit.store.list_bindings = AsyncMock(return_value=[])
        kit.store.update_room = AsyncMock()
        kit.send_event = AsyncMock()
        kit._hook_engine.run_async_hooks = AsyncMock()
        kit._lock_manager = MagicMock()
        kit._lock_manager.locked = MagicMock(return_value=_NoopLock())

        pipeline.install(
            kit,
            [triage, advisor],
            voice_channel_id="rtv",
            greet_on_handoff=True,
        )

        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="user1",
            channel_id="rtv",
        )
        rtv._sessions["s1"] = session
        rtv._session_rooms["s1"] = "r1"

        result = await rtv._tool_handler(
            session,
            "handoff_conversation",
            {"target": "agent-advisor", "reason": "help", "summary": "ctx"},
        )

        assert "introduce yourself" in result["message"].lower()

    async def test_wire_realtime_non_handoff_tool_delegates(self):
        """Non-handoff tools are delegated to the original handler."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next=None),
            ],
        )
        kit, rtv = _mock_kit_with_rtv()
        a = Agent("agent-a", role="A", system_prompt="A.")

        # Set up an original tool handler
        original_called = {}

        async def original_handler(session, name, args):
            original_called["name"] = name
            return {"result": "from_original"}

        rtv._tool_handler = original_handler

        pipeline.install(kit, [a], voice_channel_id="rtv")

        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="user1",
            channel_id="rtv",
        )

        result = await rtv._tool_handler(session, "get_weather", {"city": "NYC"})

        assert original_called["name"] == "get_weather"
        assert result == {"result": "from_original"}

    def test_traditional_mode_not_affected(self):
        """When voice_channel_id points to VoiceChannel (not RTV), traditional wiring."""
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="a", agent_id="agent-a", next="b"),
                PipelineStage(phase="b", agent_id="agent-b", next=None),
            ],
        )
        kit = _mock_kit()
        from roomkit.channels.voice import VoiceChannel

        vc = MagicMock(spec=VoiceChannel)
        vc.update_voice_map = MagicMock()
        kit._channels = {"voice": vc}

        agent_a = _mock_agent("agent-a", voice="v-a")
        agent_b = _mock_agent("agent-b", voice="v-b")

        pipeline.install(kit, [agent_a, agent_b], voice_channel_id="voice")

        # Traditional: agents get handoff tools directly
        assert len(agent_a._extra_tools) == 1
        assert len(agent_b._extra_tools) == 1
        vc.update_voice_map.assert_called_once()
