"""A live agent's turns run on the kit's own executor, whatever other executor
is served on the kit's turn backend: the agent the kit holds answers, and the
kit's hooks fire for it (RFC §10.6.6, §10.6.7, RMK-837).

The kit's backend is in memory, or across Redis in the suite's Redis mode
(tests/redis_turn_mode.py), so each test runs on both."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import (
    AgentIdentity,
    AIChannel,
    ChannelCategory,
    ChannelDeclaration,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    TextContent,
    TurnExecutor,
    WebSocketChannel,
)
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.delivery import InboundMessage
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns.wire import CommittedEvent


async def _answers(
    build: Callable[[str], AIChannel | None], *, room_hook: str | None = None
) -> tuple[list[str], list[str]]:
    """What the room's agent said to one message, and the channels the kit's
    BEFORE_AI_GENERATION hook saw, with a host executor built on *build*
    served on the kit's backend before its first turn; *room_hook*, defined on
    the host's executor alone, enabled for the room."""
    kit = RoomKit()
    host = TurnExecutor(build)
    if room_hook is not None:

        @host.room_hook(room_hook, HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC)
        async def audit(event, ctx):
            pass

    await kit._turns.backend.serve(host)
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["from the kit"])))
    seen: list[str] = []

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def observe(event, ctx):
        seen.append(event.channel_id)
        return HookResult.allow()

    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    if room_hook is not None:
        await kit.enable_room_hook("r", room_hook)
    await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="hi"))
    )
    events = await kit.store.list_events("r")
    await kit.close()
    await host.close()
    said = [e.content.body for e in events if e.source.channel_id == "ai"]
    return said, seen


async def test_a_host_executor_served_first_does_not_take_a_live_agent_s_turn() -> None:
    """The host's executor declares every channel and was served first: the
    kit's agent answers, and the kit's hooks fire, never the host's copy."""
    providers: list[MockAIProvider] = []

    def build(channel_id: str) -> AIChannel:
        providers.append(MockAIProvider(responses=["from the host"]))
        return AIChannel(channel_id, provider=providers[-1])

    said, seen = await _answers(build)

    assert said == ["from the kit"]
    assert seen == ["ai"]
    assert all(not provider.calls for provider in providers)


async def test_a_live_agent_answers_when_the_host_has_no_channel_for_it() -> None:
    """The host's factory builds nothing for the kit's agent, though its
    executor declares every channel: the kit's executor takes the turn."""

    said, seen = await _answers(lambda channel_id: None)

    assert said == ["from the kit"]
    assert seen == ["ai"]


async def test_a_live_agent_s_turn_never_moves_to_a_host_for_a_room_hook(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The room enables a hook only the host's executor defines: the live
    agent's turn finds no executor, as the kit warns when the hook is enabled,
    rather than run on the host's copy, which defines it."""
    providers: list[MockAIProvider] = []

    def build(channel_id: str) -> AIChannel:
        providers.append(MockAIProvider(responses=["from the host"]))
        return AIChannel(channel_id, provider=providers[-1])

    with caplog.at_level(logging.ERROR, logger="roomkit"):
        said, seen = await _answers(build, room_hook="audit")

    assert said == []
    assert seen == []
    assert all(not provider.calls for provider in providers)
    refusals = [r.exc_info[1] for r in caplog.records if r.exc_info]
    assert [e.reason for e in refusals if isinstance(e, TurnLinkError)] == ["no_executor"]


async def test_a_live_agent_s_turn_fails_once_the_kit_s_executor_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RMK-880: the host closes the kit's own executor while the kit runs:
    the live agent's turn fails no_executor, and the host's copy, which
    declares the channel, never answers in its place (RFC §10.6.7)."""
    providers: list[MockAIProvider] = []

    def build(channel_id: str) -> AIChannel:
        providers.append(MockAIProvider(responses=["from the host"]))
        return AIChannel(channel_id, provider=providers[-1])

    kit = RoomKit()
    host = TurnExecutor(build)
    await kit._turns.backend.serve(host)
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["from the kit"])))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.turn_executor.close()

    with caplog.at_level(logging.ERROR, logger="roomkit"):
        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="hi"))
        )
    events = await kit.store.list_events("r")
    await kit.close()
    await host.close()

    assert [e.content.body for e in events if e.source.channel_id == "ai"] == []
    assert all(not provider.calls for provider in providers)
    refusals = [r.exc_info[1] for r in caplog.records if r.exc_info]
    assert [e.reason for e in refusals if isinstance(e, TurnLinkError)] == ["no_executor"]


async def test_a_live_agent_s_call_and_event_fail_once_the_kit_s_executor_closed() -> None:
    """RMK-880 review: a channel call and a committed event of a live agent,
    once the kit's own executor closed, fail no_executor as its turns do;
    the host's copy is never asked (RFC §10.6.7)."""
    seen: list[str] = []
    kit = RoomKit()
    host = TurnExecutor(lambda channel_id: _Tagged(channel_id, "host", seen))
    await kit._turns.backend.serve(host)
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(_Tagged("ai", "kit", seen))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="hi"))
    )
    events = await kit.store.list_events("r")
    await kit.turn_executor.close()

    with pytest.raises(TurnLinkError) as call:
        await kit.channel_call("ai", "r", "whoami")
    with pytest.raises(TurnLinkError) as event:
        await kit._turns.control(CommittedEvent(channel_id="ai", room_id="r", event=events[0]))
    await kit.close()
    await host.close()

    assert (call.value.reason, event.value.reason) == ("no_executor", "no_executor")
    assert [s for s in seen if s.endswith(":host")] == []


class _Told(TurnExecutor):
    """A host executor that records each channel control it is given."""

    def __init__(self, build: Callable[[str], Any]) -> None:
        super().__init__(build)
        self.told: list[tuple[str, str]] = []

    async def control(self, message: Any) -> Any:
        self.told.append((message.type, message.channel_id))
        return await super().control(message)


async def test_a_host_executor_builds_no_copy_of_a_live_agent_it_never_runs() -> None:
    """RMK-881: a host executor declaring every channel is told nothing of a
    live agent attached to a room then detached, and builds no copy of it: it
    never runs its turns (RFC §10.6.3, §10.6.7). An attach builds nothing; the
    channel the host alone serves is built for its first turn."""
    built: list[str] = []

    def build(channel_id: str) -> AIChannel:
        built.append(channel_id)
        return AIChannel(channel_id, provider=MockAIProvider(responses=["from the host"]))

    kit = RoomKit()
    host = _Told(build)
    await kit._turns.backend.serve(host)
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["from the kit"])))
    kit.register_channel(ChannelDeclaration(channel_id="remote", identity=AgentIdentity()))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r", "remote", category=ChannelCategory.INTELLIGENCE)
    await kit.detach_channel("r", "ai")
    after_attaches = list(built)
    hi = InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="Hi."))
    await kit.process_inbound(hi, room_id="r")
    await kit.close()
    await host.close()

    assert after_attaches == []
    assert built == ["remote"]
    assert [told for told in host.told if told[1] == "ai"] == []
    assert ("room_attached", "remote") in host.told


class _Tagged(AIChannel):
    """An agent that says whose copy answered a channel call or a committed
    event."""

    def __init__(self, channel_id: str, tag: str, seen: list[str]) -> None:
        super().__init__(channel_id, provider=MockAIProvider(responses=[f"from {tag}"]))
        self._tag = tag
        self._seen = seen

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        self._seen.append(f"call:{self._tag}")
        return self._tag

    async def on_room_event(self, room_id: str, event: Any) -> None:
        self._seen.append(f"event:{self._tag}")


async def test_a_live_agent_s_channel_calls_and_committed_events_stay_on_the_kit() -> None:
    """A channel call and a committed event of a live agent reach the kit's
    copy, as its turns do, whatever the host's executor declares."""
    seen: list[str] = []
    kit = RoomKit()
    host = TurnExecutor(lambda channel_id: _Tagged(channel_id, "host", seen))
    await kit._turns.backend.serve(host)
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(_Tagged("ai", "kit", seen))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="hi"))
    )
    events = await kit.store.list_events("r")

    answer = await kit.channel_call("ai", "r", "whoami")
    await kit._turns.control(CommittedEvent(channel_id="ai", room_id="r", event=events[0]))
    await kit.close()
    await host.close()

    assert answer == "kit"
    assert seen == ["call:kit", "event:kit"]
