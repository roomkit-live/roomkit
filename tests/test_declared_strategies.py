"""Strategies and the owner's tools take an agent the kit holds by its
declaration alone, named by its id (RFC §10.6.2, §10.6.5)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import (
    ACPChannel,
    Agent,
    AgentIdentity,
    AIChannel,
    ChannelDeclaration,
    ChannelNotRegisteredError,
    ConversationPipeline,
    DeclaredChannel,
    Discussion,
    Loop,
    Pipeline,
    PipelineStage,
    RoomKit,
    Supervisor,
    Swarm,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.orchestration import ConversationRouter, HandoffHandler, setup_handoff
from roomkit.orchestration.strategies.supervisor.results import _worker_profile
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.tasks import DelegateHandler, setup_delegation
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from roomkit.voice import AudioPipelineConfig
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.test_framework import SimpleChannel


def _calls(name: str, **arguments: Any) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"call-{name}", name=name, arguments=arguments)],
    )


class _Host:
    """A host process: its executor builds the agents it hosts, on the kit's
    backend, the store shared; the kit holds each by its declaration."""

    def __init__(self, **answers: list[AIResponse]) -> None:
        self.store = InMemoryStore()
        self.backend = InMemoryTurnBackend()
        self.providers = {
            channel_id: MockAIProvider(ai_responses=[*script, *(AIResponse(content="ok"),) * 6])
            for channel_id, script in answers.items()
        }
        self.executor = TurnExecutor(
            self._build, hosts=lambda channel_id: channel_id in answers, store=self.store
        )

    def _build(self, channel_id: str) -> Agent | None:
        provider = self.providers.get(channel_id)
        return Agent(channel_id, provider=provider) if provider is not None else None

    def tools_seen(self, channel_id: str, call: int = 0) -> set[str]:
        return {tool.name for tool in self.providers[channel_id].calls[call].tools or []}

    def asked(self, channel_id: str) -> int:
        return len(self.providers[channel_id].calls)

    async def kit(self, *declarations: ChannelDeclaration) -> RoomKit:
        await self.backend.serve(self.executor)
        kit = RoomKit(store=self.store, turn_backend=self.backend)
        kit.register_channel(SimpleChannel("sms"))
        for declaration in declarations:
            kit.register_channel(declaration)
        return kit

    async def close(self, kit: RoomKit) -> None:
        await kit.close()
        await self.executor.close()


def _declared(channel_id: str, **identity: Any) -> ChannelDeclaration:
    return ChannelDeclaration(channel_id=channel_id, identity=AgentIdentity(**identity))


async def _say(kit: RoomKit, text: str, room_id: str = "r") -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="marie", content=TextContent(body=text)),
        room_id=room_id,
    )


async def _said(kit: RoomKit, channel_id: str, room_id: str = "r") -> list[str]:
    return [
        e.content.body
        for e in await kit.store.list_events(room_id)
        if e.source.channel_id == channel_id and isinstance(e.content, TextContent)
    ]


async def _room(kit: RoomKit, orchestration: Any) -> None:
    await kit.create_room(room_id="r", orchestration=orchestration)
    await kit.attach_channel("r", "sms")


# -- Strategies with declared agents --


async def test_a_supervisor_delegates_to_a_declared_worker_through_the_owner() -> None:
    host = _Host(
        boss=[
            _calls("delegate_to_ana", task="The weather in Quebec."),
            AIResponse(content="Sunny."),
        ],
        ana=[AIResponse(content="Sunny, 21 C.")],
    )
    kit = await host.kit(
        _declared("boss", role="Supervisor"),
        _declared("ana", description="Forecasts the weather", purpose="weather"),
    )
    await _room(kit, Supervisor(supervisor="boss", workers=["ana"]))

    await _say(kit, "What is the weather in Quebec?")
    async with asyncio.timeout(5):  # the worker's answer reached the supervisor's answer
        while "Sunny." not in (said := await _said(kit, "boss")):
            await asyncio.sleep(0.005)
    await host.close(kit)

    assert "delegate_to_ana" in host.tools_seen("boss")
    assert host.asked("ana") >= 1
    assert said[-1] == "Sunny."


async def test_a_loop_reviews_a_declared_producer_with_a_declared_reviewer() -> None:
    host = _Host(
        writer=[AIResponse(content="Draft: sunny tomorrow.")],
        critic=[AIResponse(content="APPROVED.")],
    )
    kit = await host.kit(_declared("writer", role="Writer"), _declared("critic", role="Critic"))
    await _room(kit, Loop(agent="writer", reviewer="critic"))

    await _say(kit, "Write the forecast.")
    said = await _said(kit, "writer")
    await host.close(kit)

    assert host.asked("critic") >= 1
    assert said == ["Draft: sunny tomorrow."]


@pytest.mark.parametrize("strategy", [Swarm, Pipeline])
async def test_a_handoff_strategy_gives_a_declared_agent_its_handoff(strategy: Any) -> None:
    host = _Host(ana=[AIResponse(content="Hello.")], bob=[AIResponse(content="Hi.")])
    kit = await host.kit(
        _declared("ana", description="Front desk"), _declared("bob", description="Billing")
    )
    await _room(kit, strategy(agents=["ana", "bob"]))

    await _say(kit, "Hello.")
    seen = host.providers["ana"].calls[0].tools or []
    await host.close(kit)

    [handoff] = [tool for tool in seen if tool.name == "handoff_conversation"]
    assert "bob: Billing" in handoff.description


async def test_a_discussion_holds_declared_agents() -> None:
    host = _Host(ana=[AIResponse(content="Ana here.")], bob=[AIResponse(content="Bob here.")])
    kit = await host.kit(_declared("ana", name="Ana"), _declared("bob", name="Bob"))
    await _room(kit, Discussion(agents=["ana", "bob"]))

    await _say(kit, "Hello everyone.")
    async with asyncio.timeout(5):  # the turns come after the message, given by the queue
        while not (spoken := await _said(kit, "ana") + await _said(kit, "bob")):
            await asyncio.sleep(0.005)
    await host.close(kit)

    assert spoken


async def test_a_discussion_refuses_a_declared_agent_that_thinks_while_listening() -> None:
    kit = RoomKit()
    kit.register_channel(ChannelDeclaration(channel_id="ana", thinks_while_listening=True))
    kit.register_channel(ChannelDeclaration(channel_id="bob"))

    with pytest.raises(ValueError, match="think"):
        await kit.create_room(room_id="r", orchestration=Discussion(agents=["ana", "bob"]))
    await kit.close()


# -- The owner's tools for a declared agent --


async def test_the_owner_serves_its_tools_to_a_declared_agent_named_by_id() -> None:
    host = _Host(ana=[AIResponse(content="Noted.")])
    kit = await host.kit(_declared("ana"))
    kit.setup_task_tools("ana")
    kit.setup_background("ana")
    setup_delegation("ana", DelegateHandler(kit))
    setup_handoff("ana", HandoffHandler(kit, ConversationRouter(default_agent_id="ana")))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)

    await _say(kit, "Hello.")
    seen = host.tools_seen("ana")
    await host.close(kit)

    assert {
        "task_status",
        "cancel_task",
        "work_in_background",
        "delegate_task",
        "handoff_conversation",
    } <= seen


# -- What a strategy refuses --


async def test_an_id_the_kit_does_not_hold_is_refused() -> None:
    kit = RoomKit()
    kit.register_channel(ChannelDeclaration(channel_id="boss"))

    with pytest.raises(ChannelNotRegisteredError):
        await kit.create_room(
            room_id="r", orchestration=Supervisor(supervisor="boss", workers=["ghost"])
        )
    with pytest.raises(ChannelNotRegisteredError):
        kit.setup_task_tools("ghost")
    await kit.close()


async def test_an_id_naming_a_transport_is_no_agent() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))

    with pytest.raises(ValueError, match="not an intelligence channel"):
        kit.setup_task_tools("sms")
    await kit.close()


async def test_a_live_agent_under_a_transport_s_id_is_no_agent() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ana"))
    live = Agent("ana", provider=MockAIProvider(responses=["x"]))

    with pytest.raises(ValueError, match="not an intelligence channel"):
        await kit.create_room(room_id="r", orchestration=Swarm(agents=[live]))
    assert await kit.list_bindings("r") == []
    await kit.close()


async def test_a_live_agent_given_under_a_declared_id_is_told_to_pass_the_id() -> None:
    kit = RoomKit()
    kit.register_channel(ChannelDeclaration(channel_id="ana"))
    live = Agent("ana", provider=MockAIProvider(responses=["x"]))

    with pytest.raises(ValueError, match="give the strategy its id"):
        await kit.create_room(room_id="r", orchestration=Swarm(agents=[live]))
    await kit.close()


async def test_a_realtime_pipeline_takes_live_agents_only() -> None:
    kit = RoomKit()
    kit.register_channel(
        RealtimeVoiceChannel(
            "rtv", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )
    )
    kit.register_channel(ChannelDeclaration(channel_id="ana"))
    pipeline = ConversationPipeline(stages=[PipelineStage(phase="front", agent_id="ana")])

    with pytest.raises(ValueError, match="agents themselves"):
        pipeline.install(kit, ["ana"], voice_channel_id="rtv")
    await kit.close()


async def test_a_worker_s_purpose_reaches_the_supervisor_s_roster() -> None:
    """The supervisor reads a declared worker's identity as a live one's."""
    host = _Host(boss=[AIResponse(content="Sure.")], ana=[AIResponse(content="x")])
    kit = await host.kit(
        _declared("boss"), _declared("ana", description="Forecasts", purpose="the weather")
    )
    await _room(kit, Supervisor(supervisor="boss", workers=["ana"]))

    await _say(kit, "Hello.")
    [tool] = [
        t for t in host.providers["boss"].calls[0].tools or [] if t.name == "delegate_to_ana"
    ]
    await host.close(kit)

    assert "Forecasts" in tool.description


# -- Review of 514108bc/0e23974c --


def test_a_channel_says_whether_its_turns_take_the_owner_s_tools(tmp_path: Any) -> None:
    ai = AIChannel("ai", provider=MockAIProvider(responses=["x"]))
    acp = ACPChannel("acp", ["agent"], cwd=tmp_path)

    assert ai.declaration().takes_owner_tools is True
    assert acp.declaration().takes_owner_tools is False
    assert ChannelDeclaration(channel_id="d").takes_owner_tools is True


async def test_a_declared_worker_that_takes_no_owner_tools_hands_back_its_text() -> None:
    """An external agent's turns never declare the result tool: its result is
    its text, as a live one's is, after one turn."""
    host = _Host(acp=[AIResponse(content="Sunny, 21 C.")])
    kit = await host.kit(ChannelDeclaration(channel_id="acp", takes_owner_tools=False))
    await kit.create_room(room_id="parent")

    task = await kit.delegate(
        "parent", "acp", "The weather.", wait=True, require_structured_result=True
    )
    seen = host.tools_seen("acp")
    asked = host.asked("acp")
    await host.close(kit)

    assert task.result is not None and "Sunny, 21 C." in task.result.output
    assert "submit_result" not in seen
    assert asked == 1


async def test_no_owner_tool_is_set_up_for_a_channel_that_takes_none(tmp_path: Any) -> None:
    kit = RoomKit()
    kit.register_channel(ChannelDeclaration(channel_id="acp", takes_owner_tools=False))
    live = ACPChannel("live-acp", ["agent"], cwd=tmp_path)
    handler = HandoffHandler(kit, ConversationRouter(default_agent_id="acp"))

    for setup in (
        lambda: kit.setup_task_tools("acp"),
        lambda: setup_handoff("acp", handler),
        lambda: setup_delegation(live, DelegateHandler(kit)),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match="does not take the tools"):
            setup()
    await kit.close()


async def test_an_agent_registered_again_gets_its_own_tools_only() -> None:
    kit = RoomKit()
    first = Agent("ana", provider=MockAIProvider(responses=["x"]))
    kit.register_channel(first)
    setup_delegation(first, DelegateHandler(kit))
    kit.setup_background(first)

    kit.unregister_channel("ana")
    left = kit._owner_tools.get("ana")
    second = Agent("ana", provider=MockAIProvider(responses=["x"]))
    kit.register_channel(second)
    kit.setup_background(second)
    names = [e.name for e in kit._owner_tools.of("ana").owner_entries("r")]
    await kit.close()

    assert left is None
    assert names == ["work_in_background"]


async def test_a_strategy_given_a_live_ai_channel_still_takes_it() -> None:
    kit = RoomKit()
    boss = Agent("boss", provider=MockAIProvider(responses=["x"]))
    worker = AIChannel("w", provider=MockAIProvider(responses=["x"]))

    await kit.create_room(room_id="r", orchestration=Supervisor(supervisor=boss, workers=[worker]))  # type: ignore[list-item]
    kit.register_channel(AIChannel("plain", provider=MockAIProvider(responses=["x"])))

    with pytest.raises(TypeError, match="not an Agent"):
        await kit.create_room(
            room_id="r2", orchestration=Supervisor(supervisor=boss, workers=["plain"])
        )
    await kit.close()


async def test_a_router_install_naming_an_unknown_id_installs_nothing() -> None:
    kit = RoomKit()
    kit.register_channel(_declared("ana"))
    hooks = len(kit._hook_engine._global_hooks)

    with pytest.raises(ChannelNotRegisteredError):
        ConversationRouter(default_agent_id="ana").install(kit, ["ana", "ghost"])

    assert len(kit._hook_engine._global_hooks) == hooks
    registry = kit._owner_tools.get("ana")
    assert registry is None or registry.owner_entries("r") == []
    await kit.close()


async def test_a_declared_agent_s_discussion_handle_is_its_name_as_a_live_one_s() -> None:
    live_kit, declared_kit = RoomKit(), RoomKit()
    live = Discussion(
        agents=[
            Agent("agent:ana", provider=MockAIProvider(responses=["x"]), name="Ana"),
            Agent("bob", provider=MockAIProvider(responses=["x"])),
        ]
    )
    await live_kit.create_room(room_id="r", orchestration=live)
    declared_kit.register_channel(_declared("agent:ana", name="Ana"))
    declared_kit.register_channel(_declared("bob"))
    declared = Discussion(agents=["agent:ana", "bob"])
    await declared_kit.create_room(room_id="r", orchestration=declared)
    await live_kit.close()
    await declared_kit.close()

    assert declared.handles() == live.handles() == {"agent:ana": "Ana", "bob": "bob"}


def test_a_declared_worker_s_purpose_is_its_roster_line() -> None:
    worker = DeclaredChannel(_declared("ana", role="Weather", purpose="Forecasts the weather"))
    live = Agent(
        "ana", provider=MockAIProvider(responses=["x"]), purpose="Forecasts", voice="nova"
    )

    assert "Forecasts the weather" in _worker_profile(worker)
    assert (live.declaration().identity.purpose, live.declaration().identity.voice) == (
        "Forecasts",
        "nova",
    )


async def test_a_declared_agent_s_voice_reaches_the_voice_map() -> None:
    backend = MockVoiceBackend()
    stt, tts = MockSTTProvider(transcripts=["hi"]), MockTTSProvider()
    kit = RoomKit(stt=stt, tts=tts, voice=backend)
    voice = VoiceChannel(
        "voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
    )
    kit.register_channel(voice)
    kit.register_channel(_declared("ana", voice="nova"))
    pipeline = ConversationPipeline(stages=[PipelineStage(phase="front", agent_id="ana")])

    pipeline.install(kit, ["ana"], voice_channel_id="voice")
    resolved = voice._resolve_voice("ana")
    await kit.close()

    assert resolved == "nova"
