"""The tools orchestration injects stay declared under Tool Search (RFC §21.1; RMK-277).

Tool Search hides a large catalogue behind ``find_tools``. A handoff, a
delegation, a delegation's result tool or a strategy's tool is one the agent is
told to call: it stays declared as a pinned tool does, and it does not count
toward the catalogue whose size decides the collapse.
"""

from __future__ import annotations

from typing import Any

from roomkit import RoomKit
from roomkit.channels._tool_search import estimate_tool_tokens
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.mixins._result_capture import capture_result
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIResponseEvent
from roomkit.orchestration.handoff import setup_handoff
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.result import SUBMIT_RESULT
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import DelegateHandler, setup_delegation, setup_realtime_delegation
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.tool_loop_modes import respond
from tests.tool_room import owner_tools_in

ORCHESTRATION = {"handoff_conversation", "delegate_task", "submit_result"}
CRM = [
    AITool(
        name=f"crm_action_{i}",
        description=f"Perform CRM operation number {i} on a customer record, "
        "with filters, pagination and audit metadata.",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    for i in range(30)
]


class _Orchestrator:
    """The handoff's and the delegation's handler: a delegation is accepted.

    It sets its tools up on *kit*, which serves them to the channel's turns."""

    _max_conversation = 0

    def __init__(self, kit: RoomKit) -> None:
        self._kit = kit

    async def handle(self, **kwargs: Any) -> Any:
        return {"status": "delegated"}


class _UnknownWindow(MockAIProvider):
    """A model whose context window the catalog does not know."""

    @property
    def context_window(self) -> int | None:
        return None


async def _crm(name: str, arguments: dict[str, Any]) -> str:
    return "ok"


def _orchestrated(
    streaming: bool,
    tools: list[AITool],
    *script: AIResponse,
    provider: MockAIProvider | None = None,
    **kwargs: Any,
) -> tuple[RoomKit, AIChannel]:
    """A channel with a handoff and a delegation wired on it, and the kit
    that serves them."""
    answers = [*script, *(AIResponse(content="noted") for _ in range(4))]
    provider = provider or MockAIProvider(ai_responses=answers, streaming=streaming)
    channel = AIChannel("ai1", provider=provider, tools=tools, tool_handler=_crm, **kwargs)
    kit = RoomKit()
    setup_handoff(channel, _Orchestrator(kit))
    setup_delegation(channel, _Orchestrator(kit))
    return kit, channel


def _calling(name: str, **arguments: Any) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"call-{name}", name=name, arguments=arguments)],
    )


async def _turn(kit: RoomKit, channel: AIChannel, room_id: str) -> int:
    """Run one turn in *room_id*, given the tools *kit* serves it; the index
    of its first provider call."""
    binding = ChannelBinding(
        channel_id="ai1",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    before = len(_model(channel).calls)
    with owner_tools_in(kit, "ai1", room_id):
        await respond(
            channel,
            make_event(room_id=room_id, body="go", channel_id="sms1"),
            binding,
            RoomContext(room=Room(id=room_id)),
        )
    return before


def _model(channel: AIChannel) -> MockAIProvider:
    provider = channel.provider
    assert isinstance(provider, MockAIProvider)
    return provider


def _declared_at(channel: AIChannel, call: int) -> set[str]:
    return {tool.name for tool in _model(channel).calls[call].tools or []}


async def _declared_in(kit: RoomKit, channel: AIChannel, room_id: str) -> set[str]:
    return _declared_at(channel, await _turn(kit, channel, room_id))


async def test_the_tools_orchestration_injects_stay_declared(streaming: bool) -> None:
    kit, channel = _orchestrated(streaming, CRM)

    with capture_result(kit._owner_tools.of("ai1"), "r1", SUBMIT_RESULT):
        declared = await _declared_in(kit, channel, "r1")

    assert declared >= ORCHESTRATION
    assert "find_tools" in declared  # Tool Search is on
    assert "crm_action_0" not in declared


async def test_a_result_tool_stays_in_its_child_room(streaming: bool) -> None:
    kit, channel = _orchestrated(streaming, CRM)

    with capture_result(kit._owner_tools.of("ai1"), "parent::task-1", SUBMIT_RESULT):
        elsewhere = await _declared_in(kit, channel, "r1")
        child = await _declared_in(kit, channel, "parent::task-1")

    assert "submit_result" not in elsewhere
    assert "submit_result" in child


async def test_orchestration_tools_do_not_tip_a_catalogue_into_tool_search(
    streaming: bool,
) -> None:
    host = CRM[:5]
    window = MockAIProvider().context_window
    assert window is not None
    # A budget the host tools fit under, and that the orchestration tools
    # would overflow if they counted.
    budget_pct = (sum(estimate_tool_tokens(t) for t in host) + 1) * 100 / window
    kit, channel = _orchestrated(streaming, host, tool_search_threshold_pct=budget_pct)

    declared = await _declared_in(kit, channel, "r1")

    assert "find_tools" not in declared
    assert {"crm_action_0", "handoff_conversation", "delegate_task"} <= declared


async def test_orchestration_tools_do_not_count_when_the_window_is_unknown(
    streaming: bool,
) -> None:
    """Without a window the rule counts tools: 20 host tools stay under the
    default threshold of 20 whatever orchestration adds."""
    provider = _UnknownWindow(responses=["hi"], streaming=streaming)
    kit, channel = _orchestrated(streaming, CRM[:20], provider=provider)

    declared = await _declared_in(kit, channel, "r1")

    assert "find_tools" not in declared
    assert {"crm_action_19", "handoff_conversation", "delegate_task"} <= declared


async def test_a_later_round_keeps_the_result_tool_in_its_child_room(streaming: bool) -> None:
    kit, channel = _orchestrated(streaming, CRM, _calling("find_tools", query="crm record"))

    with capture_result(kit._owner_tools.of("ai1"), "parent::task-1", SUBMIT_RESULT):
        first = await _turn(kit, channel, "parent::task-1")

    assert "submit_result" in _declared_at(channel, first)
    assert "submit_result" in _declared_at(channel, first + 1)


async def test_find_tools_does_not_name_an_orchestration_tool(streaming: bool) -> None:
    search = _calling("find_tools", query="delegate a task to another agent")
    kit, channel = _orchestrated(streaming, CRM, search)

    first = await _turn(kit, channel, "r1")

    results = [
        str(part.result)
        for message in _model(channel).calls[first + 1].messages
        if message.role == "tool"
        for part in message.content
    ]
    assert results
    assert all("delegate_task" not in result for result in results)


async def test_an_orchestration_tool_is_declared_always_even_once_used(streaming: bool) -> None:
    """A tool the room already called turns sticky; one orchestration
    injected is never gated by Tool Search, so it stays ``always``."""
    delegate = _calling("delegate_task", agent="worker", task="research")
    kit, channel = _orchestrated(streaming, CRM, delegate)
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    channel._after_response_hook = observe

    await _turn(kit, channel, "r1")
    await _turn(kit, channel, "r1")

    origins = [
        {tool.name: tool.origin for tool in event.declared_tools}["delegate_task"]
        for event in seen
    ]
    assert origins == ["always", "always"]


def _voice(provider: MockRealtimeProvider) -> RealtimeVoiceChannel:
    """A voice channel whose catalogue is behind Tool Search."""
    tools = [
        {"name": t.name, "description": t.description, "parameters": t.parameters} for t in CRM
    ]
    return RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=_crm,
        tool_search=True,
    )


def _connected(provider: MockRealtimeProvider) -> set[str]:
    connected = [c.args for c in provider.calls if c.method == "connect"][-1]
    return {tool["name"] for tool in connected["tools"] or []}


async def test_a_voice_supervisors_tool_stays_declared() -> None:
    provider = MockRealtimeProvider()
    voice = _voice(provider)
    kit = RoomKit()
    kit.register_channel(voice)
    supervisor = Supervisor(
        Agent("sup", provider=MockAIProvider(responses=["ok"])),
        [Agent("worker", provider=MockAIProvider(responses=["done"]))],
        strategy="sequential",
        auto_delegate=True,
        async_delivery=True,
    )
    await kit.create_room(room_id="r1", orchestration=supervisor)
    await kit.attach_channel("r1", "voice")

    await voice.start_session("r1", "caller", "ws")

    declared = _connected(provider)
    assert "delegate_workers" in declared
    assert "crm_action_0" not in declared
    await kit.close()


class _FixedProvider(MockRealtimeProvider):
    """A provider whose declarations cannot change mid-session."""

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return False


async def test_a_voice_loop_and_a_fixed_declaration_delegation_stay_declared() -> None:
    provider = _FixedProvider()
    voice = _voice(provider)
    kit = RoomKit()
    kit.register_channel(voice)
    setup_realtime_delegation(voice, DelegateHandler(kit))
    loop = Loop(
        agent=Agent("writer", provider=MockAIProvider(responses=["draft"])),
        reviewer=Agent("editor", provider=MockAIProvider(responses=["ok"])),
        async_delivery=True,
    )
    await kit.create_room(room_id="r1", orchestration=loop)
    await kit.attach_channel("r1", "voice")

    await voice.start_session("r1", "caller", "ws")

    declared = _connected(provider)
    assert {"delegate_task", "delegate_loop", "call_tool"} <= declared
    assert "crm_action_0" not in declared
    await kit.close()


async def test_a_realtime_delegation_and_a_voice_pipelines_handoff_stay_declared() -> None:
    provider = MockRealtimeProvider()
    voice = _voice(provider)
    kit = RoomKit()
    triage, billing = (Agent(a, provider=MockAIProvider()) for a in ("triage", "billing"))
    for channel in (voice, triage, billing):
        kit.register_channel(channel)
    setup_realtime_delegation(voice, DelegateHandler(kit))
    ConversationPipeline(
        stages=[
            PipelineStage(phase="triage", agent_id="triage", next="billing"),
            PipelineStage(phase="billing", agent_id="billing", next=None),
        ]
    ).install(kit, [triage, billing], voice_channel_id="voice")
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice")

    await voice.start_session("r1", "caller", "ws")

    declared = _connected(provider)
    assert {"delegate_task", "handoff_conversation"} <= declared
    assert "crm_action_0" not in declared
    await kit.close()
