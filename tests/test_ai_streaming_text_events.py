"""An agent's answer text, published live as TEXT_DELTA ephemerals (RFC §6.4, RMK-951).

The code that delegated a task reads its agent's answer as it is written,
coalesced on a window, in the task's child room, once the host turns it on
with ``publish_text_deltas=True``. BEFORE_BROADCAST does not cover that
stream, and only a delegated turn's trace is stored without a hook either,
so no other turn's text is published (RMK-972).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import (
    AgentResponsePolicy,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.channels import _ai_coalescers
from roomkit.channels.agent import Agent
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AITool,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.sms.mock import MockSMSProvider

WORDS = [f"word{i} " for i in range(100)]
ANSWER = "".join(WORDS)


class _Rounds(AIProvider):
    """Streams scripted rounds: each a list of text tokens, then an optional
    tool call ending the round."""

    def __init__(
        self,
        rounds: list[tuple[list[str], StreamToolCall | None]],
        *,
        first_token_delay: float = 0.0,
    ) -> None:
        self._rounds = list(rounds)
        self._first_token_delay = first_token_delay

    @property
    def model_name(self) -> str:
        return "rounds"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        tokens, _ = self._rounds[0]
        return AIResponse(content="".join(tokens), finish_reason="stop")

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for token in self._rounds[0][0]:
            yield token

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        tokens, call = self._rounds.pop(0)
        if self._first_token_delay:
            await asyncio.sleep(self._first_token_delay)
        for token in tokens:
            yield StreamTextDelta(text=token)
        if call is not None:
            yield call
        yield StreamDone(finish_reason="tool_calls" if call else "stop", usage={})


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event: Any) -> None:
        self.events.append((event.type.value, event.data or {}))

    def texts(self) -> list[dict[str, Any]]:
        return [data for kind, data in self.events if kind == "text_delta"]


@pytest.fixture
async def kit():
    kit = RoomKit()
    yield kit
    await kit.close()


async def _delegated_texts(kit: RoomKit, agent: Agent, task: str = "go") -> _Observer:
    """What a subscriber to *agent*'s child room reads while it does *task*."""
    kit.register_channel(agent)
    observer = _Observer()

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def watch(event, ctx) -> None:
        await kit.realtime.subscribe_to_room(event.metadata["child_room_id"], observer)

    room = await kit.create_room()
    await (await kit.delegate(room.id, agent.channel_id, task, wait=True)).wait()
    return observer


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def text_window_clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    # Only the coalescers read this clock; asyncio and Redis keep real time.
    clock = _Clock()
    monkeypatch.setattr(_ai_coalescers, "time", clock)
    return clock


@pytest.mark.usefixtures("text_window_clock")
@pytest.mark.parametrize("first_token_delay", [0.0, 0.12], ids=["immediate", "delayed"])
@pytest.mark.parametrize(
    ("coalesce_ms", "publishes"),
    # 690 characters: one publish per token, or one per 256-character window.
    [(0, len(WORDS)), (80, 3)],
    ids=["window off: one per token", "default window: by size"],
)
async def test_a_delegated_answer_is_published_as_its_text_on_a_window(
    kit: RoomKit, coalesce_ms: float, publishes: int, first_token_delay: float
) -> None:
    agent = Agent(
        "worker",
        provider=_Rounds([(WORDS, None)], first_token_delay=first_token_delay),
        publish_text_deltas=True,
        text_coalesce_ms=coalesce_ms,
    )

    texts = (await _delegated_texts(kit, agent)).texts()

    assert "".join(data["text"] for data in texts) == ANSWER
    assert len(texts) == publishes
    assert {data["channel_id"] for data in texts} == {"worker"}


async def test_elapsed_time_publishes_a_small_first_piece_before_the_size_window(
    kit: RoomKit, text_window_clock: _Clock
) -> None:
    class AfterTheWindow(_Rounds):
        async def generate_structured_stream(
            self, context: AIContext
        ) -> AsyncIterator[StreamEvent]:
            text_window_clock.now = 0.08
            async for event in super().generate_structured_stream(context):
                yield event

    agent = Agent(
        "worker",
        provider=AfterTheWindow([(WORDS, None)]),
        publish_text_deltas=True,
        text_coalesce_ms=80,
    )

    texts = (await _delegated_texts(kit, agent)).texts()

    assert [data["text"] for data in texts] == [
        WORDS[0],
        "".join(WORDS[1:39]),
        "".join(WORDS[39:76]),
        "".join(WORDS[76:]),
    ]
    assert "".join(data["text"] for data in texts) == ANSWER
    assert {data["channel_id"] for data in texts} == {"worker"}


@pytest.mark.parametrize("serialized", [False, True], ids=["side by side", "one at a time"])
async def test_a_delegated_turns_text_reaches_its_child_room_while_it_is_written(kit, serialized):
    # A turn that waits its place in the room is read as a trace all the same.
    kit.register_channel(
        Agent(
            "worker",
            provider=_Rounds([(WORDS, None)]),
            publish_text_deltas=True,
            serialize_room_turns=serialized,
        )
    )
    observer = _Observer()
    order: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def watch(event, ctx) -> None:
        async def seen(ephemeral: Any) -> None:
            await observer(ephemeral)
            order.append("delta")

        await kit.realtime.subscribe_to_room(event.metadata["child_room_id"], seen)

    room = await kit.create_room()
    result = await (await kit.delegate(room.id, "worker", "go", wait=True)).wait()
    order.append("returned")

    assert "".join(data["text"] for data in observer.texts()) == result.output == ANSWER
    assert order[-1] == "returned" and "delta" in order


async def test_text_before_a_tool_call_is_published_before_it_with_its_round(kit):
    call = StreamToolCall(id="c1", name="lookup", arguments={"order": "1042"})
    provider = _Rounds([(["Let me check. "], call), (["It shipped today."], None)])

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "shipped"

    agent = Agent(
        "worker",
        provider=provider,
        tools=[AITool(name="lookup", description="Look up an order", parameters={})],
        tool_handler=lookup,
        publish_text_deltas=True,
    )

    observer = await _delegated_texts(kit, agent, "where is my order?")

    kinds = [kind for kind, _ in observer.events]
    assert kinds.index("text_delta") < kinds.index("tool_call_start")
    assert [(data["round"], data["text"]) for data in observer.texts()] == [
        (0, "Let me check. "),
        (1, "It shipped today."),
    ]


SECRET = "Your card 4111 1111 1111 1111 is on file."
_MODERATIONS = {
    "block": lambda event: HookResult.block("pii"),
    "redact": lambda event: HookResult.modify(
        event.model_copy(update={"content": TextContent(body="[redacted]")})
    ),
}


async def _answer_in_the_room(kit: RoomKit, observer: _Observer) -> None:
    await kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)
    await kit.realtime.subscribe_to_room("r", observer)
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms", sender_id="+15550100001", content=TextContent(body="my card?")
        ),
        room_id="r",
    )


async def _answer_a_task_sharing_the_sms(kit: RoomKit, observer: _Observer) -> None:
    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def watch(event, ctx) -> None:
        await kit.realtime.subscribe_to_room(event.metadata["child_room_id"], observer)

    task = await kit.delegate("r", "assistant", "my card?", share_channels=["sms"], wait=True)
    await task.wait()


async def _answer_another_agent_in_a_tasks_room(kit: RoomKit, observer: _Observer) -> None:
    """A task's room keeps its own turn as a trace; a turn another agent's
    message starts there crosses the room's hooks, as any room's."""
    kit.register_channel(Agent("planner", provider=_Rounds([])))
    task = await kit.delegate("r", "assistant", "warm up", wait=True)
    await task.wait()
    child = task.child_room_id
    await kit.attach_channel(child, "planner", category=ChannelCategory.INTELLIGENCE)
    await kit.set_agent_response_policy(child, AgentResponsePolicy.ADDRESSED_ONLY)
    await kit.realtime.subscribe_to_room(child, observer)
    await kit.send_event(
        child, "planner", TextContent(body="the card?"), addressed_to=["assistant"]
    )


_WHERE = {
    "room": _answer_in_the_room,
    "task sharing the sms": _answer_a_task_sharing_the_sms,
    "another agent in a task's room": _answer_another_agent_in_a_tasks_room,
}


@pytest.mark.parametrize("moderation", sorted(_MODERATIONS))
@pytest.mark.parametrize("publish", [None, True], ids=["default", "turned on"])
@pytest.mark.parametrize("where", sorted(_WHERE))
async def test_an_answer_a_hook_judges_is_never_published_as_text(
    kit, moderation: str, publish: bool | None, where: str
) -> None:
    """A hook that blocks or redacts the answer judges the stored message,
    never a TEXT_DELTA stream: a turn that crosses BEFORE_BROADCAST, in a
    task's room a transport is shared into (RFC §23.3 step 5) or that
    another agent's message starts there included, publishes no stream,
    turned on or not, so a subscriber reads only what the room got (RMK-972:
    turned on, it read the card number the SMS never sent)."""
    judge = _MODERATIONS[moderation]
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    options = {} if publish is None else {"publish_text_deltas": publish}
    provider = _Rounds([([SECRET], None), ([SECRET], None)])
    kit.register_channel(Agent("assistant", provider=provider, **options))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms", metadata={"phone_number": "+15550100001"})
    judged: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def moderate(event, ctx) -> HookResult:
        if event.source.channel_id != "assistant":
            return HookResult.allow()
        judged.append(event.content.body)
        return judge(event)

    observer = _Observer()
    await _WHERE[where](kit, observer)
    await kit.close()

    assert judged == [SECRET]
    assert observer.texts() == []
