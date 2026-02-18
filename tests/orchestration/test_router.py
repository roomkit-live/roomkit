"""Tests for ConversationRouter."""

from __future__ import annotations

from unittest.mock import MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.room import Room
from roomkit.orchestration.handoff import HandoffHandler
from roomkit.orchestration.router import (
    ConversationRouter,
    RoutingConditions,
    RoutingRule,
)
from roomkit.orchestration.state import (
    ConversationState,
    set_conversation_state,
)
from tests.conftest import make_event

# -- Helpers ------------------------------------------------------------------


def _ai_binding(channel_id: str, room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _transport_binding(
    channel_id: str,
    room_id: str = "r1",
    channel_type: ChannelType = ChannelType.SMS,
) -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=channel_type,
    )


def _context(
    room: Room | None = None,
    bindings: list[ChannelBinding] | None = None,
) -> RoomContext:
    return RoomContext(
        room=room or Room(id="r1"),
        bindings=bindings or [],
    )


# -- select_agent: no router / backward compat --------------------------------


class TestSelectAgentNoRules:
    def test_no_rules_no_default_returns_none(self):
        router = ConversationRouter()
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(bindings=[_transport_binding("sms1")])
        state = ConversationState()

        assert router.select_agent(event, ctx, state) is None

    def test_no_rules_with_default(self):
        router = ConversationRouter(default_agent_id="agent-triage")
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(bindings=[_transport_binding("sms1"), _ai_binding("agent-triage")])
        state = ConversationState()

        assert router.select_agent(event, ctx, state) == "agent-triage"


# -- select_agent: agent affinity (sticky routing) ----------------------------


class TestAgentAffinity:
    def test_sticky_routing_when_active_agent_set(self):
        router = ConversationRouter(default_agent_id="agent-default")
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-a"),
                _ai_binding("agent-default"),
            ]
        )
        state = ConversationState(active_agent_id="agent-a")

        assert router.select_agent(event, ctx, state) == "agent-a"

    def test_falls_through_when_active_agent_not_in_room(self):
        router = ConversationRouter(default_agent_id="agent-fallback")
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-fallback"),
            ]
        )
        # agent-gone is not in bindings
        state = ConversationState(active_agent_id="agent-gone")

        assert router.select_agent(event, ctx, state) == "agent-fallback"


# -- select_agent: rule matching ----------------------------------------------


class TestRuleMatching:
    def test_phase_rule(self):
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-intake",
                    conditions=RoutingConditions(phases={"intake"}),
                ),
                RoutingRule(
                    agent_id="agent-handling",
                    conditions=RoutingConditions(phases={"handling"}),
                ),
            ],
        )
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-intake"),
                _ai_binding("agent-handling"),
            ]
        )

        state_intake = ConversationState(phase="intake")
        assert router.select_agent(event, ctx, state_intake) == "agent-intake"

        state_handling = ConversationState(phase="handling")
        assert router.select_agent(event, ctx, state_handling) == "agent-handling"

    def test_channel_type_rule(self):
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-voice",
                    conditions=RoutingConditions(
                        channel_types={ChannelType.VOICE},
                    ),
                ),
                RoutingRule(
                    agent_id="agent-text",
                    conditions=RoutingConditions(
                        channel_types={ChannelType.SMS, ChannelType.WEBSOCKET},
                    ),
                ),
            ],
        )
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-voice"),
                _ai_binding("agent-text"),
            ]
        )

        sms_event = make_event(room_id="r1", channel_id="sms1", channel_type=ChannelType.SMS)
        assert router.select_agent(sms_event, ctx, ConversationState()) == "agent-text"

        voice_event = make_event(room_id="r1", channel_id="voice1", channel_type=ChannelType.VOICE)
        assert router.select_agent(voice_event, ctx, ConversationState()) == "agent-voice"

    def test_source_channel_id_rule(self):
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-vip",
                    conditions=RoutingConditions(source_channel_ids={"vip-ws"}),
                ),
            ],
            default_agent_id="agent-default",
        )
        ctx = _context(
            bindings=[
                _transport_binding("vip-ws", channel_type=ChannelType.WEBSOCKET),
                _transport_binding("normal-ws", channel_type=ChannelType.WEBSOCKET),
                _ai_binding("agent-vip"),
                _ai_binding("agent-default"),
            ]
        )

        vip_event = make_event(
            room_id="r1", channel_id="vip-ws", channel_type=ChannelType.WEBSOCKET
        )
        assert router.select_agent(vip_event, ctx, ConversationState()) == "agent-vip"

        normal_event = make_event(
            room_id="r1", channel_id="normal-ws", channel_type=ChannelType.WEBSOCKET
        )
        assert router.select_agent(normal_event, ctx, ConversationState()) == "agent-default"

    def test_intent_rule(self):
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-billing",
                    conditions=RoutingConditions(intents={"billing", "payment"}),
                ),
            ],
            default_agent_id="agent-general",
        )
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-billing"),
                _ai_binding("agent-general"),
            ]
        )

        billing_event = make_event(room_id="r1", channel_id="sms1", metadata={"intent": "billing"})
        assert router.select_agent(billing_event, ctx, ConversationState()) == "agent-billing"

        other_event = make_event(room_id="r1", channel_id="sms1", metadata={"intent": "support"})
        assert router.select_agent(other_event, ctx, ConversationState()) == "agent-general"

    def test_custom_predicate(self):
        def is_urgent(event, context, state):
            return (event.metadata or {}).get("urgent") is True

        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-urgent",
                    conditions=RoutingConditions(custom=is_urgent),
                ),
            ],
            default_agent_id="agent-normal",
        )
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-urgent"),
                _ai_binding("agent-normal"),
            ]
        )

        urgent = make_event(room_id="r1", channel_id="sms1", metadata={"urgent": True})
        assert router.select_agent(urgent, ctx, ConversationState()) == "agent-urgent"

        normal = make_event(room_id="r1", channel_id="sms1", metadata={"urgent": False})
        assert router.select_agent(normal, ctx, ConversationState()) == "agent-normal"

    def test_combined_conditions_and_logic(self):
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-special",
                    conditions=RoutingConditions(
                        phases={"handling"},
                        channel_types={ChannelType.SMS},
                    ),
                ),
            ],
            default_agent_id="agent-default",
        )
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-special"),
                _ai_binding("agent-default"),
            ]
        )
        state = ConversationState(phase="handling")

        # Matches both conditions
        sms = make_event(room_id="r1", channel_id="sms1", channel_type=ChannelType.SMS)
        assert router.select_agent(sms, ctx, state) == "agent-special"

        # Phase matches but channel type doesn't
        ws = make_event(room_id="r1", channel_id="ws1", channel_type=ChannelType.WEBSOCKET)
        assert router.select_agent(ws, ctx, state) == "agent-default"

    def test_no_matching_rule_falls_to_default(self):
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-nope",
                    conditions=RoutingConditions(phases={"nonexistent"}),
                ),
            ],
            default_agent_id="agent-default",
        )
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-default"),
            ]
        )

        event = make_event(room_id="r1", channel_id="sms1")
        assert router.select_agent(event, ctx, ConversationState()) == "agent-default"


# -- select_agent: priority ---------------------------------------------------


class TestRulePriority:
    def test_lower_priority_wins(self):
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-low",
                    conditions=RoutingConditions(phases={"intake"}),
                    priority=10,
                ),
                RoutingRule(
                    agent_id="agent-high",
                    conditions=RoutingConditions(phases={"intake"}),
                    priority=0,
                ),
            ],
        )
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-low"),
                _ai_binding("agent-high"),
            ]
        )

        event = make_event(room_id="r1", channel_id="sms1")
        assert router.select_agent(event, ctx, ConversationState()) == "agent-high"


# -- select_agent: intelligence source skipped --------------------------------


class TestIntelligenceSourceSkipped:
    def test_events_from_ai_channel_not_routed(self):
        router = ConversationRouter(default_agent_id="agent-a")
        event = make_event(room_id="r1", channel_id="agent-a", channel_type=ChannelType.AI)
        ctx = _context(
            bindings=[
                _ai_binding("agent-a"),
                _ai_binding("agent-b"),
            ]
        )

        assert router.select_agent(event, ctx, ConversationState()) is None


# -- as_hook: integration with HookResult -------------------------------------


class TestAsHook:
    async def test_hook_returns_allow_when_no_routing_from_transport(self):
        router = ConversationRouter()  # no rules, no default
        hook = router.as_hook()
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(bindings=[_transport_binding("sms1")])

        result = await hook(event, ctx)
        assert result.action == "allow"

    async def test_hook_blocks_intelligence_source_reentry(self):
        """Events FROM an AI agent should not trigger other AI agents."""
        router = ConversationRouter(default_agent_id="agent-a")
        hook = router.as_hook()
        event = make_event(room_id="r1", channel_id="agent-a", channel_type=ChannelType.AI)
        ctx = _context(
            bindings=[
                _ai_binding("agent-a"),
                _ai_binding("agent-b"),
            ]
        )

        result = await hook(event, ctx)
        assert result.action == "modify"
        assert result.event is not None
        assert result.event.metadata["_routed_to"] == "_none_"

    async def test_hook_intelligence_source_includes_supervisor(self):
        """Supervisor still in _always_process even for intelligence-sourced events."""
        router = ConversationRouter(
            default_agent_id="agent-a",
            supervisor_id="agent-sup",
        )
        hook = router.as_hook()
        event = make_event(room_id="r1", channel_id="agent-a", channel_type=ChannelType.AI)
        ctx = _context(
            bindings=[
                _ai_binding("agent-a"),
                _ai_binding("agent-sup"),
            ]
        )

        result = await hook(event, ctx)
        assert result.event is not None
        assert result.event.metadata["_routed_to"] == "_none_"
        assert "agent-sup" in result.event.metadata["_always_process"]

    async def test_hook_stamps_routed_to_metadata(self):
        router = ConversationRouter(default_agent_id="agent-a")
        hook = router.as_hook()
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(bindings=[_transport_binding("sms1"), _ai_binding("agent-a")])

        result = await hook(event, ctx)
        assert result.action == "modify"
        assert result.event is not None
        assert result.event.metadata["_routed_to"] == "agent-a"

    async def test_hook_includes_supervisor_in_always_process(self):
        router = ConversationRouter(
            default_agent_id="agent-a",
            supervisor_id="agent-supervisor",
        )
        hook = router.as_hook()
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-a"),
                _ai_binding("agent-supervisor"),
            ]
        )

        result = await hook(event, ctx)
        assert result.event is not None
        assert "agent-supervisor" in result.event.metadata["_always_process"]

    async def test_hook_preserves_existing_metadata(self):
        router = ConversationRouter(default_agent_id="agent-a")
        hook = router.as_hook()
        event = make_event(room_id="r1", channel_id="sms1", metadata={"custom": "value"})
        ctx = _context(bindings=[_transport_binding("sms1"), _ai_binding("agent-a")])

        result = await hook(event, ctx)
        assert result.event is not None
        assert result.event.metadata["custom"] == "value"
        assert result.event.metadata["_routed_to"] == "agent-a"

    async def test_hook_reads_state_from_room_metadata(self):
        """Router reads ConversationState from Room.metadata for rule matching."""
        router = ConversationRouter(
            rules=[
                RoutingRule(
                    agent_id="agent-handler",
                    conditions=RoutingConditions(phases={"handling"}),
                ),
            ],
            default_agent_id="agent-triage",
        )
        hook = router.as_hook()

        # Room with handling phase in metadata
        room = Room(id="r1")
        state = ConversationState(phase="handling")
        room = set_conversation_state(room, state)

        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(
            room=room,
            bindings=[
                _transport_binding("sms1"),
                _ai_binding("agent-handler"),
                _ai_binding("agent-triage"),
            ],
        )

        result = await hook(event, ctx)
        assert result.event is not None
        assert result.event.metadata["_routed_to"] == "agent-handler"

    async def test_hook_always_process_is_list(self):
        """_always_process must be a list (not set) for JSON serialization."""
        router = ConversationRouter(
            default_agent_id="agent-a",
            supervisor_id="agent-sup",
        )
        hook = router.as_hook()
        event = make_event(room_id="r1", channel_id="sms1")
        ctx = _context(bindings=[_transport_binding("sms1"), _ai_binding("agent-a")])

        result = await hook(event, ctx)
        assert result.event is not None
        always = result.event.metadata["_always_process"]
        assert isinstance(always, list)


# -- install ------------------------------------------------------------------


def _mock_agent(channel_id: str) -> Agent:
    from roomkit.providers.ai.mock import MockAIProvider

    return Agent(channel_id, provider=MockAIProvider(responses=["ok"]))


def _mock_kit() -> MagicMock:
    kit = MagicMock()
    hook_decorator = MagicMock()
    kit.hook.return_value = hook_decorator
    return kit


class TestRouterInstall:
    def test_install_registers_hook_and_returns_handler(self):
        router = ConversationRouter(default_agent_id="agent-a")
        kit = _mock_kit()
        agents = [_mock_agent("agent-a")]

        handler = router.install(kit, agents)

        assert isinstance(handler, HandoffHandler)
        kit.hook.assert_called_once_with(
            HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            priority=-100,
        )

    def test_install_with_phase_map(self):
        router = ConversationRouter(default_agent_id="agent-a")
        kit = _mock_kit()
        agents = [_mock_agent("agent-a"), _mock_agent("agent-b")]

        handler = router.install(
            kit,
            agents,
            phase_map={"agent-a": "intake", "agent-b": "handling"},
        )

        assert handler._phase_map == {"agent-a": "intake", "agent-b": "handling"}

    def test_install_wires_handoff(self):
        router = ConversationRouter(default_agent_id="agent-a")
        kit = _mock_kit()
        agents = [_mock_agent("agent-a"), _mock_agent("agent-b")]

        router.install(kit, agents)

        for agent in agents:
            assert len(agent._extra_tools) == 1
            assert agent._extra_tools[0].name == "handoff_conversation"

    def test_install_passes_aliases(self):
        router = ConversationRouter(default_agent_id="agent-a")
        kit = _mock_kit()
        agents = [_mock_agent("agent-a")]

        handler = router.install(kit, agents, agent_aliases={"alias": "agent-a"})

        assert handler._aliases == {"alias": "agent-a"}

    def test_install_no_allowed_transitions(self):
        """Router install does not set allowed_transitions (pipeline-only)."""
        router = ConversationRouter(default_agent_id="agent-a")
        kit = _mock_kit()
        agents = [_mock_agent("agent-a")]

        handler = router.install(kit, agents)

        assert handler._allowed_transitions is None
