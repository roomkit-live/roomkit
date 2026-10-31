"""The room's owner holds an agent by its declaration alone (RFC §10.6.2)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import (
    ACPChannel,
    Agent,
    AgentIdentity,
    AIChannel,
    ChannelAlreadyRegisteredError,
    ChannelDeclaration,
    DeclaredChannel,
    Loop,
    Pipeline,
    RoomKit,
    Supervisor,
    Swarm,
    TurnLinkError,
)
from roomkit.channels.voice import VoiceChannel
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelMediaType,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import TextContent
from roomkit.orchestration import Discussion
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.mock import MockSpeakPolicy, MockThinker
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from roomkit.voice import AudioPipelineConfig
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.testing import VoiceTrace
from roomkit.voice.tts.mock import MockTTSProvider
from tests.test_framework import SimpleChannel

ANA = ChannelDeclaration(
    channel_id="ana",
    identity=AgentIdentity(name="Ana", role="Weather agent"),
    recent_events_window=7,
)


class _Host:
    """A host process: a store shared with the kit, an executor that builds
    the agents it hosts, served on the kit's backend."""

    def __init__(self, **agents: str) -> None:
        self.store = InMemoryStore()
        self.backend = InMemoryTurnBackend()
        self.built: list[str] = []
        self.providers: dict[str, MockAIProvider] = {}
        self._answers = agents
        self.executor = TurnExecutor(
            self._build, hosts=lambda channel_id: channel_id in agents, store=self.store
        )

    def _build(self, channel_id: str) -> Agent | None:
        if channel_id not in self._answers:
            return None
        self.built.append(channel_id)
        provider = MockAIProvider(responses=[self._answers[channel_id]])
        self.providers[channel_id] = provider
        return Agent(channel_id, provider=provider, role="Weather agent")

    async def kit(self, *declarations: ChannelDeclaration) -> RoomKit:
        await self.backend.serve(self.executor)
        kit = RoomKit(store=self.store, turn_backend=self.backend)
        kit.register_channel(SimpleChannel("sms"))
        for declaration in declarations:
            kit.register_channel(declaration)
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "sms")
        return kit

    async def close(self, kit: RoomKit) -> None:
        await kit.close()
        await self.executor.close()


async def _say(kit: RoomKit, text: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="marie", content=TextContent(body=text)),
        room_id="r",
    )


async def _said(kit: RoomKit, channel_id: str, room_id: str = "r") -> list[str]:
    return [
        e.content.body
        for e in await kit.store.list_events(room_id)
        if e.source.channel_id == channel_id and isinstance(e.content, TextContent)
    ]


# -- A channel declares itself --


def test_an_agent_declares_its_identity_and_the_history_it_reads() -> None:
    agent = Agent(
        "ana",
        provider=MockAIProvider(responses=["x"]),
        name="Ana",
        role="Weather agent",
        description="Forecasts",
        scope="Quebec",
        greeting="Hello",
        language="fr",
        auto_greet=False,
    )

    declared = agent.declaration()

    assert (declared.channel_id, declared.channel_type, declared.category) == (
        "ana",
        ChannelType.AI,
        ChannelCategory.INTELLIGENCE,
    )
    assert declared.identity == AgentIdentity(
        name="Ana",
        role="Weather agent",
        description="Forecasts",
        scope="Quebec",
        greeting="Hello",
        language="fr",
        auto_greet=False,
    )
    assert declared.capabilities == agent.capabilities()
    assert declared.recent_events_window == agent.recent_events_window > 0
    assert declared.thinks_while_listening is False


def test_an_ai_channel_with_a_thinker_declares_it_thinks_while_listening() -> None:
    ai = AIChannel(
        "ai",
        provider=MockAIProvider(responses=["x"]),
        speak_policy=MockSpeakPolicy(["speak"]),
        thinker=MockThinker([]),
    )

    assert ai.declaration().thinks_while_listening is True


def test_a_declaration_left_to_its_defaults_reads_as_an_ai_channel() -> None:
    live = AIChannel("ana", provider=MockAIProvider(responses=["x"])).declaration()

    assert ChannelDeclaration(channel_id="ana") == live


def test_an_ai_channel_and_an_acp_agent_declare_no_identity(tmp_path: Any) -> None:
    ai = AIChannel("ai", provider=MockAIProvider(responses=["x"]))
    acp = ACPChannel("acp", ["agent"], cwd=tmp_path)

    assert ai.declaration().identity is None
    assert ai.declaration().category is ChannelCategory.INTELLIGENCE
    assert acp.declaration().identity is None
    assert acp.declaration().recent_events_window == acp.recent_events_window


# -- The kit holds a declaration where it holds a channel --


async def test_a_declaration_is_bound_from_what_it_declares() -> None:
    kit = RoomKit()
    capabilities = ChannelCapabilities(media_types=[ChannelMediaType.TEXT], max_length=500)
    kit.register_channel(ANA.model_copy(update={"capabilities": capabilities}))
    await kit.create_room(room_id="r")

    binding = await kit.attach_channel("r", "ana")

    assert isinstance(kit.channels["ana"], DeclaredChannel)
    assert binding.category is ChannelCategory.INTELLIGENCE
    assert binding.channel_type is ChannelType.AI
    assert binding.capabilities == capabilities
    assert kit._resolve_recent_events_limit([binding]) == 7
    await kit.close()


async def test_a_declared_agent_s_turn_runs_on_the_executor_that_builds_it() -> None:
    host = _Host(ana="It is sunny in Quebec.")
    kit = await host.kit(ANA)
    await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)

    await _say(kit, "What is the weather in Quebec?")
    said = await _said(kit, "ana")
    await host.close(kit)

    assert said == ["It is sunny in Quebec."]
    assert host.built == ["ana"]


async def test_a_hand_written_declaration_s_turns_read_the_room_s_history() -> None:
    """A declaration that names only its id gets the history a live AI channel
    gets, though nothing else in the process reads the room's tail."""
    host = _Host(ana="It is sunny.")
    kit = await host.kit(ChannelDeclaration(channel_id="ana"))
    await kit.attach_channel("r", "ana")

    await _say(kit, "Hello.")
    await _say(kit, "What is the weather?")
    last = host.providers["ana"].calls[-1]
    await host.close(kit)

    assert [m.content for m in last.messages if m.role == "user"][-2:] == [
        "Hello.",
        "What is the weather?",
    ]


async def test_with_no_executor_to_build_it_a_declared_agent_s_turn_fails_at_once() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(ANA)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)

    await asyncio.wait_for(_say(kit, "What is the weather?"), 2)
    said = await _said(kit, "ana")
    # Its own executor never takes it: the turn found no executor, rather than
    # reaching the placeholder, which refuses it.
    taken = kit.turn_executor.takes("ana"), kit.turn_executor.takes("sms")
    await kit.close()

    assert taken == (False, True)
    assert said == []


async def test_a_channel_call_to_a_declared_agent_with_no_executor_fails_at_once() -> None:
    kit = RoomKit()
    kit.register_channel(ANA)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ana")

    with pytest.raises(TurnLinkError) as raised:
        await asyncio.wait_for(kit.channel_call("ana", "r", "set_mode", mode="quiet"), 2)
    await kit.close()

    assert raised.value.reason == "no_executor"


async def test_a_declaration_never_takes_a_live_channel_s_id_nor_the_reverse() -> None:
    kit = RoomKit()
    kit.register_channel(_agent("bob"))
    kit.register_channel(ANA)

    with pytest.raises(ChannelAlreadyRegisteredError):
        kit.register_channel(ChannelDeclaration(channel_id="bob"))
    with pytest.raises(ChannelAlreadyRegisteredError):
        kit.register_channel(_agent("ana"))

    assert isinstance(kit.channels["bob"], Agent)
    assert isinstance(kit.channels["ana"], DeclaredChannel)


def test_an_intelligence_channel_with_no_valid_declaration_is_never_registered() -> None:
    class _Unbounded(AIChannel):
        @property
        def recent_events_window(self) -> int:
            return -1

    kit = RoomKit()

    with pytest.raises(ValueError):
        kit.register_channel(_Unbounded("ai", provider=MockAIProvider(responses=["x"])))

    assert "ai" not in kit.channels


def _auto_greets(kit: RoomKit, agent_id: str) -> int:
    hooks = kit._hook_engine._global_hooks
    return sum(1 for h in hooks if h.name == f"_agent_auto_greet:{agent_id}")


def test_an_agent_registered_again_greets_by_its_new_setup_only() -> None:
    kit = RoomKit()
    greeter = Agent("ana", provider=MockAIProvider(responses=["x"]), greeting="Welcome!")
    kit.register_channel(greeter)

    kit.unregister_channel("ana")
    kit.register_channel(greeter.declaration())
    twice = _auto_greets(kit, "ana")
    kit.unregister_channel("ana")
    kit.register_channel(ChannelDeclaration(channel_id="ana", identity=AgentIdentity()))

    assert twice == 1
    assert _auto_greets(kit, "ana") == 0


@pytest.mark.parametrize(("auto_greet", "greetings"), [(True, 1), (False, 0)])
async def test_a_room_is_greeted_by_the_agent_s_last_registration(
    auto_greet: bool, greetings: int
) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("ana", provider=MockAIProvider(responses=["x"]), greeting="Hi!"))
    kit.unregister_channel("ana")
    again = Agent(
        "ana", provider=MockAIProvider(responses=["x"]), greeting="Hi!", auto_greet=auto_greet
    )
    kit.register_channel(again)

    @kit.hook(HookTrigger.ON_ROOM_CREATED, HookExecution.ASYNC)
    async def attach(event: Any, ctx: Any) -> None:
        await kit.attach_channel(event.room_id, "ana", category=ChannelCategory.INTELLIGENCE)

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="marie", content=TextContent(body="Hello"))
    )
    # The attach, then the greeting, run in observers: done once they all are.
    await kit.hook_engine.drain_observers()
    events = await kit.store.list_events(result.event.room_id)
    await kit.close()

    assert sum(1 for e in events if e.metadata.get("auto_greeting")) == greetings


async def test_a_delegation_to_a_declared_worker_runs_on_its_executor() -> None:
    host = _Host(ana="Sunny, 21 C.")
    kit = await host.kit(ANA, ChannelDeclaration(channel_id="speaker"))
    await kit.attach_channel("r", "speaker", category=ChannelCategory.INTELLIGENCE)

    task = await kit.delegate("r", "ana", "The weather in Quebec.", notify="speaker")
    result = await task.wait(timeout=5)
    await host.close(kit)

    assert result.output == "Sunny, 21 C."
    assert host.built == ["ana"]


# -- A declared agent reads as the live one on the kit --


async def _greetings(agent: Agent, *, declared: bool) -> list[tuple[str, str]]:
    backend = MockVoiceBackend()
    stt, tts = MockSTTProvider(transcripts=["hi"]), MockTTSProvider()
    kit = RoomKit(stt=stt, tts=tts, voice=backend)
    trace = VoiceTrace(kit, triggers=(HookTrigger.ON_SESSION_STARTED,))
    kit.register_channel(
        VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig())
    )
    kit.register_channel(agent.declaration() if declared else agent)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, agent.channel_id)

    await kit.connect_voice(room.id, "user-1", "voice")
    await trace.wait_for(HookTrigger.ON_SESSION_STARTED)
    await kit.hook_engine.drain_observers()  # the greeting is one of its observers
    events = await kit.store.list_events(room.id)
    await kit.close()
    return [
        (e.source.channel_id, e.content.body)
        for e in events
        if e.metadata.get("auto_greeting") is True and isinstance(e.content, TextContent)
    ]


@pytest.mark.parametrize("auto_greet", [True, False])
async def test_a_declared_agent_greets_as_the_live_one_does(auto_greet: bool) -> None:
    def agent() -> Agent:
        provider = MockAIProvider(responses=["x"])
        return Agent("ana", provider=provider, greeting="Welcome!", auto_greet=auto_greet)

    live = await _greetings(agent(), declared=False)
    declared = await _greetings(agent(), declared=True)

    assert declared == live == ([("ana", "Welcome!")] if auto_greet else [])


async def test_send_greeting_speaks_for_a_declared_agent() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(
        ANA.model_copy(update={"identity": AgentIdentity(greeting="Hi!", auto_greet=False)})
    )
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ana")

    await kit.send_greeting("r")
    said = await _said(kit, "ana")
    await kit.close()

    assert said == ["Hi!"]


# -- What a declared channel refuses --


def test_only_an_intelligence_channel_registers_by_its_declaration() -> None:
    kit = RoomKit()

    with pytest.raises(ValueError, match="intelligence"):
        kit.register_channel(
            ChannelDeclaration(channel_id="sms", category=ChannelCategory.TRANSPORT)
        )

    assert "sms" not in kit.channels


async def test_a_declared_channel_is_attached_as_an_intelligence_channel_only() -> None:
    kit = RoomKit()
    kit.register_channel(ANA)
    await kit.create_room(room_id="r")

    with pytest.raises(ValueError, match="intelligence channel only"):
        await kit.attach_channel("r", "ana", category=ChannelCategory.TRANSPORT)

    assert await kit.store.list_bindings("r") == []
    await kit.close()


async def test_a_declared_channel_runs_nothing_in_the_kit_s_process() -> None:
    channel = DeclaredChannel(ANA)

    with pytest.raises(TypeError, match="declaration alone"):
        await channel.on_event(None, None, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="declaration alone"):
        await channel.deliver(None, None, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="declaration alone"):
        await channel.handle_inbound(None, None)  # type: ignore[arg-type]


def _agent(channel_id: str) -> Agent:
    return Agent(channel_id, provider=MockAIProvider(responses=["x"]))


@pytest.mark.parametrize(
    ("strategy", "left"),
    [
        # The supervisor and the producer are the room's agents, bound before
        # the strategy's install refuses its worker or reviewer: a failed
        # install takes their bindings back, never a registration of the kit.
        (
            lambda: Supervisor(supervisor=_agent("boss"), workers=[_agent("w1"), _agent("ana")]),
            {"boss"},
        ),
        (
            lambda: Loop(agent=_agent("writer"), reviewers=[_agent("rev"), _agent("ana")]),
            {"writer"},
        ),
        (lambda: Discussion(agents=[_agent("bob"), _agent("ana")]), set()),
        (lambda: Pipeline(agents=[_agent("bob"), _agent("ana")]), set()),
        (lambda: Swarm(agents=[_agent("bob"), _agent("ana")]), set()),
    ],
    ids=["supervisor", "loop", "discussion", "pipeline", "swarm"],
)
async def test_a_strategy_refuses_an_agent_the_kit_holds_by_declaration(
    strategy: Any, left: set[str]
) -> None:
    kit = RoomKit()
    kit.register_channel(ANA)

    with pytest.raises(ValueError, match="declaration alone"):
        await kit.create_room(room_id="r", orchestration=strategy())

    assert isinstance(kit.channels["ana"], DeclaredChannel)
    # No agent the refusing call was given is registered, the ones before
    # the declared one included.
    assert set(kit.channels) == {"ana", *left}
    await kit.close()


async def test_a_declared_channel_s_events_are_its_own() -> None:
    """The kit stores a declared agent's answer as it would a live one's."""
    host = _Host(ana="Sunny.")
    kit = await host.kit(ANA)
    await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)

    await _say(kit, "Weather?")
    events = [e for e in await kit.store.list_events("r") if e.source.channel_id == "ana"]
    await host.close(kit)

    assert [e.type for e in events] == [EventType.MESSAGE]
    assert events[0].source.channel_type is ChannelType.AI


def test_no_channel_claims_the_type_reserved_to_the_framework() -> None:
    """RFC §5.3, §6.2: ``SYSTEM`` is the framework's own records' type, whose
    source may make the changes reserved to the system; a channel claiming
    it is refused, as roomkit-rs refuses it (RMK-1021 item 28). It was
    registered."""
    kit = RoomKit()

    with pytest.raises(ValueError, match="claims the reserved type system"):
        kit.register_channel(SimpleChannel("sys", ChannelType.SYSTEM))

    assert "sys" not in kit.channels
