"""Every path to a channel that hosts a realtime model reads it the same way
(RMK-516, RFC §12.4, §22.1, §23.3 step 8).

``deliver()`` with no destination prefers any such host, a conference with its
model plugged in included, whatever the attach order. Every text injected into
a session goes through its host's ``inject_text``, so ON_REALTIME_TEXT_INJECTED
hears the greeting, a recovered call's result, a handoff's language
instruction and a pipeline's handoff greeting as it hears a delivery. And every
door a call is served on names the session that issued it in the handler's
context (``get_current_voice_session()``), the conference's included.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel, get_current_voice_session
from roomkit.conference.mock import MockConferenceBackend
from roomkit.models.enums import ChannelType
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_framework import SimpleChannel

_TOOLS = [
    {
        "name": "lookup",
        "description": "look up",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]


def _injected(provider: MockRealtimeProvider) -> list[tuple[Any, str]]:
    return [
        (call.args.get("role"), str(call.args.get("text")))
        for call in provider.calls
        if call.method == "inject_text"
    ]


def _announced(kit: RoomKit) -> list[str | None]:
    """The roles ON_REALTIME_TEXT_INJECTED hears, in order."""
    roles: list[str | None] = []

    @kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
    async def audit(event: Any, context: Any) -> None:
        roles.append(event.metadata.get("injected_role"))

    return roles


# -- deliver() with no destination -------------------------------------------


def _host(kind: str, provider: MockRealtimeProvider) -> Any:
    if kind == "conference":
        config = ConferenceRealtimeConfig(provider=provider)
        return ConferenceChannel("host", backend=MockConferenceBackend(), realtime=config)
    return RealtimeVoiceChannel("host", provider=provider, transport=MockRealtimeTransport())


@pytest.mark.parametrize("order", ["text-first", "host-first"])
@pytest.mark.parametrize("kind", ["voice", "conference"])
async def test_deliver_without_a_destination_prefers_the_realtime_host(
    kind: str, order: str
) -> None:
    kit = RoomKit()
    provider = MockRealtimeProvider()
    host = _host(kind, provider)
    sms = SimpleChannel("sms")
    kit.register_channel(host)
    kit.register_channel(sms)
    await kit.create_room(room_id="r")
    for channel_id in ("sms", "host") if order == "text-first" else ("host", "sms"):
        await kit.attach_channel("r", channel_id)
    if kind == "conference":
        await host._realtime.ensure_session("r")
    else:
        await host.start_session("r", "u1", "ws")

    outcome = await kit.deliver("r", "hello")
    await kit.close()

    assert outcome.status == "sent"
    assert [text for _, text in _injected(provider)] == ["hello"]
    assert sms.delivered == []


# -- ON_REALTIME_TEXT_INJECTED on every injection ----------------------------


async def _voice(**options: Any) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rtv", provider=provider, transport=MockRealtimeTransport(), **options
    )
    kit = RoomKit()
    kit.register_channel(channel)
    return kit, channel, provider


async def _open(kit: RoomKit, channel: RealtimeVoiceChannel) -> Any:
    room = await kit.create_room(room_id="r")
    await kit.attach_channel(room.id, channel.channel_id)
    return await channel.start_session(room.id, "u1", "ws")


async def test_the_greeting_is_announced() -> None:
    kit, channel, provider = await _voice()
    roles = _announced(kit)
    kit.register_channel(
        Agent("boss", provider=MockAIProvider(responses=["x"]), greeting="Welcome!")
    )
    session = await _open(kit, channel)

    await kit.send_greeting(
        "r", agent_id="boss", session=session, channel_type=ChannelType.REALTIME_VOICE
    )
    await until(lambda: bool(roles))
    await kit.close()

    assert _injected(provider) == [("assistant", "Welcome!")]
    assert roles == ["assistant"]


async def test_a_recovered_calls_result_is_announced() -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    kit, channel, provider = await _voice(tools=_TOOLS, tool_handler=handler)
    roles = _announced(kit)
    session = await _open(kit, channel)

    assert channel._try_recover_tool_call_from_text(session, "call:lookup{q:x}")[0]
    await until(lambda: bool(roles))
    await kit.close()

    [(role, text)] = _injected(provider)
    assert (role, "found" in text) == ("user", True)
    assert roles == ["user"]


async def test_a_handoffs_language_instruction_is_announced() -> None:
    kit, channel, provider = await _voice()
    roles = _announced(kit)
    agent = Agent("agent-a", role="A", greeting="Bonjour !", language="French")
    kit.register_channel(agent)
    pipeline = ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="agent-a")])
    _, handoff = pipeline.install(kit, [agent], voice_channel_id="rtv")
    await _open(kit, channel)
    room = await kit.get_room("r")
    state = ConversationState(phase="a", active_agent_id="agent-a")
    await kit.store.update_room(set_conversation_state(room, state))

    await handoff.send_greeting("r", channel_id="rtv")
    await until(lambda: len(roles) == 2)
    await kit.close()

    assert _injected(provider) == [("system", "Respond in French."), ("assistant", "Bonjour !")]
    assert roles == ["system", "assistant"]


async def test_a_pipelines_handoff_greeting_is_announced() -> None:
    kit, channel, provider = await _voice()
    roles = _announced(kit)
    triage = Agent("agent-triage", role="Triage", system_prompt="Hi.")
    advisor = Agent("agent-advisor", role="Advisor", system_prompt="Help.")
    pipeline = ConversationPipeline(
        stages=[
            PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
            PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
        ],
    )
    pipeline.install(kit, [triage, advisor], voice_channel_id="rtv", greet_on_handoff=True)
    room = await kit.create_room(room_id="r")
    state = ConversationState(active_agent_id="agent-triage", phase="triage")
    await kit.store.update_room(set_conversation_state(room, state))
    await kit.attach_channel("r", "rtv")
    session = await channel.start_session("r", "u1", "ws")

    await provider.simulate_tool_call(
        session,
        "h1",
        "handoff_conversation",
        {"target": "agent-advisor", "reason": "help", "summary": "ctx"},
    )
    await until(lambda: bool(roles))
    await kit.close()

    assert [role for role, _ in _injected(provider)] == ["system"]
    assert roles == ["system"]


# -- the issuing session in the handler's context, on every door -------------


def _seeing(seen: list[str | None]) -> Any:
    async def handler(*args: Any) -> str:
        session = get_current_voice_session()
        seen.append(session.id if session is not None else None)
        return "found"

    return handler


class _Backend(ReasoningBackend):
    async def run(self, request: Any):  # type: ignore[override]
        text = await request.execute_tool("lookup", {"q": "x"})
        yield ReasoningOutput(text, spoken=True)


@pytest.mark.parametrize("door", ["provider", "recovered", "backend"])
async def test_a_channel_door_names_the_issuing_session(door: str) -> None:
    seen: list[str | None] = []
    options: dict[str, Any] = {"tools": _TOOLS, "tool_handler": _seeing(seen)}
    provider = MockRealtimeProvider(full_duplex=door == "backend")
    if door == "backend":
        options["reasoning_backend"] = _Backend()
    channel = RealtimeVoiceChannel(
        "rtv", provider=provider, transport=MockRealtimeTransport(), **options
    )
    kit = RoomKit()
    kit.register_channel(channel)
    session = await _open(kit, channel)

    if door == "provider":
        await provider.simulate_tool_call(session, "c1", "lookup", {"q": "x"})
    elif door == "recovered":
        assert channel._try_recover_tool_call_from_text(session, "call:lookup{q:x}")[0]
    else:
        await provider.simulate_delegation(session, "d1", "integrator")
    await until(lambda: bool(seen))
    await kit.close()

    assert seen == [session.id]


async def test_the_conference_door_names_the_issuing_session() -> None:
    seen: list[str | None] = []
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(provider=provider, tools=_TOOLS, tool_handler=_seeing(seen))
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)

    await provider.simulate_tool_call(session, "c1", "lookup", {"q": "x"})
    await until(lambda: bool(seen))
    await kit.close()

    assert seen == [session.id]
