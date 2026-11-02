"""A message sent while an agent's turn runs is read by that turn (RMK-1160,
RFC §6.4, §21.3).

``steer()`` hands a running turn a directive it reads at its next stop: between
two rounds of tool calls, or once the model has answered, when the same turn
asks the model again rather than drop the message. An agent built with
``steer_room_turns=True`` hands such a turn the person's next message itself:
the message is stored as any message is, and no second turn starts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import ChannelCategory, InboundMessage, RoomKit, TextContent
from roomkit.channels import SMSChannel
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.steering import InjectMessage
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import _ToolLoopContext
from tests.test_framework import SimpleChannel

WAIT = AITool(name="wait_here", description="Wait.", parameters={"type": "object"})
CALL = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="wait_here", arguments={})])
STEERED = "Only the .py files."


class _Gated(MockAIProvider):
    """Holds its generation number *held* (0-based) until *release* is set,
    and counts the generations running at once."""

    def __init__(self, responses: list[AIResponse], held: int, *, streaming: bool) -> None:
        super().__init__(ai_responses=responses, streaming=streaming)
        self.held = held
        self.answering = asyncio.Event()
        self.release = asyncio.Event()
        self.entered = 0
        self.inside: set[int] = set()
        self.most_at_once = 0

    async def _hold(self, context: AIContext) -> bool:
        if id(context) in self.inside:
            return False  # the streamed mock reads itself through generate()
        self.inside.add(id(context))
        self.most_at_once = max(self.most_at_once, len(self.inside))
        self.entered += 1
        if self.entered - 1 == self.held:
            self.answering.set()
            await self.release.wait()
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        entered = await self._hold(context)
        try:
            return await super().generate(context)
        finally:
            if entered:
                self.inside.discard(id(context))

    async def _stream(self, context: AIContext) -> Any:
        await self._hold(context)
        try:
            async for event in super()._stream(context):
                yield event
        finally:
            self.inside.discard(id(context))


class _Paused:
    """A host tool that waits until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.started.set()
        await self.release.wait()
        return "a.py b.txt"


async def _done(name: str, arguments: dict[str, Any]) -> str:
    return "a.py b.txt"


async def _kit(provider: MockAIProvider, tool: Any = _done, *, steer: bool = False) -> RoomKit:
    agent = Agent(
        "agent",
        provider=provider,
        role="agent",
        tools=[WAIT],
        tool_handler=tool,
        steer_room_turns=steer,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit


def _message(body: str, sender: str = "u1", name: str | None = None) -> InboundMessage:
    return InboundMessage(
        channel_id="ops",
        sender_id=sender,
        content=TextContent(body=body),
        metadata={"sender_name": name} if name else {},
    )


def _read(context: AIContext, text: str) -> bool:
    return any(text in str(m.content) for m in context.messages)


async def _bodies(kit: RoomKit, channel_id: str) -> list[str]:
    return [
        str(getattr(e.content, "body", ""))
        for e in await kit.store.list_events("r1")
        if e.source.channel_id == channel_id and e.type == EventType.MESSAGE
    ]


# -- steer(): the answer is a stop too -------------------------------------------


async def test_a_message_steered_while_the_model_answers_asks_it_again(streaming: bool) -> None:
    """The model answers, and a message steered meanwhile is read: the same
    turn asks it again with the answer then the message, where it was counted
    and dropped (RMK-1160: steer() said 1, nothing read it)."""
    provider = _Gated(
        [CALL, AIResponse(content="a.py, b.txt"), AIResponse(content="a.py")],
        held=1,
        streaming=streaming,
    )
    kit = await _kit(provider)
    turn = asyncio.create_task(kit.process_inbound(_message("List the files."), room_id="r1"))
    await asyncio.wait_for(provider.answering.wait(), 2)

    reached = await kit.steer(InjectMessage(content=STEERED), "agent", room_id="r1")
    provider.release.set()
    await asyncio.wait_for(turn, 5)

    asked_again = provider.calls[2]
    assert reached == 1
    assert len(provider.calls) == 3
    assert [m.role for m in asked_again.messages[-2:]] == ["assistant", "user"]
    assert _read(asked_again, "a.py, b.txt") and _read(asked_again, STEERED)
    assert await _bodies(kit, "agent") == ["a.py, b.txt", "a.py"]
    await kit.close()


def test_a_loop_that_read_its_queue_for_the_last_time_is_reached_by_nothing() -> None:
    ch = AIChannel("ai1", provider=MockAIProvider())
    ended = _ToolLoopContext(loop_id="loop-0", room_id="r1", steering_closed=True)
    ch._active_loops[ended.loop_id] = ended

    assert ch.steer(InjectMessage(content=STEERED), room_id="r1") == 0
    assert ch.steer(InjectMessage(content=STEERED), loop_id="loop-0") == 0
    assert ended.steering_queue.empty()


# -- steer_room_turns: the next message goes to the running turn -----------------


async def test_a_message_during_a_tool_round_is_stored_and_read_by_the_running_turn(
    streaming: bool,
) -> None:
    provider = _Gated([CALL, AIResponse(content="a.py")], held=-1, streaming=streaming)
    tool = _Paused()
    kit = await _kit(provider, tool, steer=True)
    turn = asyncio.create_task(kit.process_inbound(_message("List the files."), room_id="r1"))
    await asyncio.wait_for(tool.started.wait(), 2)

    sent = await kit.process_inbound(_message(STEERED), room_id="r1")
    tool.release.set()
    await asyncio.wait_for(turn, 5)

    assert not sent.blocked
    assert len(provider.calls) == 2
    assert _read(provider.calls[1], STEERED)
    assert await _bodies(kit, "ops") == ["List the files.", STEERED]
    assert await _bodies(kit, "agent") == ["a.py"]
    await kit.close()


async def test_a_message_while_the_model_answers_is_answered_by_the_same_turn(
    streaming: bool,
) -> None:
    provider = _Gated(
        [AIResponse(content="a.py, b.txt"), AIResponse(content="a.py")],
        held=0,
        streaming=streaming,
    )
    kit = await _kit(provider, steer=True)
    turn = asyncio.create_task(kit.process_inbound(_message("List the files."), room_id="r1"))
    await asyncio.wait_for(provider.answering.wait(), 2)

    await kit.process_inbound(_message(STEERED), room_id="r1")
    provider.release.set()
    await asyncio.wait_for(turn, 5)
    await kit.close()

    assert provider.most_at_once == 1
    assert len(provider.calls) == 2
    assert _read(provider.calls[1], STEERED)
    assert await _bodies(kit, "agent") == ["a.py, b.txt", "a.py"]


async def test_off_a_message_during_a_turn_starts_its_own(streaming: bool) -> None:
    """``steer_room_turns`` off (the default): the second message's turn runs
    beside the first, as before."""
    provider = _Gated(
        [AIResponse(content="first"), AIResponse(content="second")], held=0, streaming=streaming
    )
    kit = await _kit(provider)
    turn = asyncio.create_task(kit.process_inbound(_message("one"), room_id="r1"))
    await asyncio.wait_for(provider.answering.wait(), 2)
    second = asyncio.create_task(kit.process_inbound(_message("two"), room_id="r1"))
    async with asyncio.timeout(2):
        while provider.entered < 2:
            await asyncio.sleep(0.001)
    provider.release.set()
    await asyncio.wait_for(asyncio.gather(turn, second), 5)
    await kit.close()

    assert provider.most_at_once == 2


async def test_a_steered_message_carries_its_speaker_when_the_room_has_several(
    streaming: bool,
) -> None:
    """In a room where several people speak, the turn reads the message under
    its speaker's label, as its history labels the room's turns (RFC §6.4)."""
    provider = _Gated([CALL, AIResponse(content="ok")], held=-1, streaming=streaming)
    tool = _Paused()
    agent = Agent(
        "agent", provider=provider, tools=[WAIT], tool_handler=tool, steer_room_turns=True
    )
    kit = RoomKit()
    kit.register_channel(SMSChannel("sms1"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1", group=True)
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)

    def said(sender: str, name: str, body: str) -> InboundMessage:
        return InboundMessage(
            channel_id="sms1",
            sender_id=sender,
            content=TextContent(body=body),
            metadata={"sender_name": name},
        )

    turn = asyncio.create_task(kit.process_inbound(said("+15550001", "Alice", "go"), room_id="r1"))
    await asyncio.wait_for(tool.started.wait(), 2)

    await kit.process_inbound(said("+15550002", "Bob", STEERED), room_id="r1")
    tool.release.set()
    await asyncio.wait_for(turn, 5)

    assert _read(provider.calls[1], f'Bob: "{STEERED}"')
    await kit.close()
