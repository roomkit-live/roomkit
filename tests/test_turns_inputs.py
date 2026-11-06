"""A human-input request of a turn can be answered on the room owner (RFC §10.6.5).

The tool waiting for the human runs on the turn's executor, where the request
settles first come first served (an answer the channel's handler took, a
hook's block, a timeout, a release); the owner keeps the request's record,
answers it (``kit.answer_input`` / ``kit.reject_input``) as long as it is open,
learns whether its answer was taken, and holds an answer that comes before
the request.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.models.pending_input import PendingInputEvent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler
from roomkit.turns._owner_inputs import OwnerInputs
from roomkit.turns.wire import InputAnswer, InputRequest
from tests.test_framework import SimpleChannel

ASK = AITool(
    name="ask_user",
    description="Ask the user.",
    parameters={"type": "object", "properties": {"question": {"type": "string"}}},
)


async def _kit(
    *,
    timeout: float = 5,
    organization_id: str | None = None,
) -> tuple[RoomKit, AIChannel, HumanInputToolHandler, list[PendingInputEvent], list[Any]]:
    """A kit whose agent asks the user once, then answers."""
    call = AIResponse(
        content="",
        tool_calls=[AIToolCall(id="c1", name="ask_user", arguments={"question": "Color?"})],
    )
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="Thanks.")])
    human = HumanInputToolHandler(tool_names={"ask_user"}, timeout=timeout, tool_definitions=[ASK])
    agent = AIChannel("agent", provider=provider, human_input_handler=human)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1", organization_id=organization_id)
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    asked: list[PendingInputEvent] = []
    reports: list[Any] = []

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC)
    async def on_input(event: PendingInputEvent, ctx: Any) -> HookResult:
        asked.append(event)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def on_call(event: ToolCallEvent, ctx: Any) -> None:
        reports.append(event)

    return kit, agent, human, asked, reports


async def _asked(asked: list[PendingInputEvent]) -> str:
    for _ in range(200):
        if asked:
            return asked[0].pending_id
        await asyncio.sleep(0.01)
    raise AssertionError("The human was never asked")


def _say(kit: RoomKit) -> asyncio.Task[Any]:
    message = InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="Hi"))
    return asyncio.create_task(kit.process_inbound(message, room_id="r1"))


async def test_the_owner_answers_a_turns_request() -> None:
    kit, agent, _, asked, reports = await _kit()
    turn = _say(kit)
    pending_id = await _asked(asked)

    assert await kit.answer_input(pending_id, "Blue") is True
    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.result == "Blue"
    # The request is answered once.
    assert await kit.answer_input(pending_id, "Red") is False
    assert await kit.reject_input(pending_id, "late") is False
    await kit.close()


async def test_the_owner_reads_an_open_request_as_its_hook_did() -> None:
    """A door answering a request tells who was asked first: the owner reads
    the request by its id, its room, channel, tool call and the actor of its
    turn, until it is answered (RFC §10.6.5)."""
    kit, _, _, asked, _ = await _kit()
    turn = _say(kit)
    pending_id = await _asked(asked)

    read = kit.input_request(pending_id)

    assert read is not None
    hooked = asked[0]
    for field in ("pending_id", "tool_name", "arguments", "room_id", "tool_call_id"):
        assert getattr(read, field) == getattr(hooked, field), field
    assert (read.channel_id, read.channel_type, read.actor_id) == (
        hooked.channel_id,
        hooked.channel_type,
        hooked.actor_id,
    )
    assert await kit.answer_input(pending_id, "Blue") is True
    assert kit.input_request(pending_id) is None
    await asyncio.wait_for(turn, 5)
    await kit.close()


async def test_an_open_request_is_read_within_its_organization_only() -> None:
    kit, _, _, asked, _ = await _kit(organization_id="org-a")
    turn = _say(kit)
    pending_id = await _asked(asked)

    assert kit.input_request(pending_id, organization_id="org-b") is None
    assert kit.input_request(pending_id, organization_id="org-a") is not None
    assert kit.input_request("no-such-request") is None
    assert await kit.answer_input(pending_id, "Blue") is True
    await asyncio.wait_for(turn, 5)
    await kit.close()


async def test_the_channels_handler_answers_where_the_tool_waits_first_come_first_served() -> None:
    kit, _, human, asked, reports = await _kit()
    turn = _say(kit)
    pending_id = await _asked(asked)

    assert human.handler.resolve(pending_id, "Green") is True
    assert human.handler.resolve(pending_id, "Red") is False
    # The kit's door, right after: the tool already has its answer.
    assert await kit.answer_input(pending_id, "Blue") is False
    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.result == "Green"
    await kit.close()


async def test_an_answer_the_handler_took_closes_the_owners_record_while_the_turn_runs() -> None:
    kit, _, human, asked, _ = await _kit()
    still_open: list[bool] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
    async def after_the_answer(event: ToolCallEvent, ctx: Any) -> None:
        # The turn still runs; the owner learnt the request settled.
        for _ in range(100):
            if event_id not in kit._turns.inputs._open:
                break
            await asyncio.sleep(0.005)
        still_open.append(event_id in kit._turns.inputs._open)

    turn = _say(kit)
    event_id = await _asked(asked)
    for _ in range(100):
        if event_id in kit._turns.inputs._open:
            break
        await asyncio.sleep(0.005)
    assert event_id in kit._turns.inputs._open

    assert human.handler.resolve(event_id, "Green") is True
    await asyncio.wait_for(turn, 5)

    assert still_open == [False]
    await kit.close()


async def test_a_block_after_an_answer_does_not_take_it_back() -> None:
    kit, _, human, _, reports = await _kit()

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC, priority=-1)
    async def answer_then_block(event: PendingInputEvent, ctx: Any) -> HookResult:
        human.handler.resolve(event.pending_id, "Blue")
        return HookResult.block("Too late to deny")

    await asyncio.wait_for(_say(kit), 5)

    [report] = reports
    assert report.result == "Blue"
    await kit.close()


async def test_the_kits_door_says_true_only_for_an_answer_the_tool_reads() -> None:
    kit, _, _, _, reports = await _kit()
    said: list[bool] = []

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC, priority=-1)
    async def answer_then_block(event: PendingInputEvent, ctx: Any) -> HookResult:
        said.append(await kit.answer_input(event.pending_id, "Blue"))
        return HookResult.block("Too late to deny")

    await asyncio.wait_for(_say(kit), 5)

    [report] = reports
    assert said == [True]
    assert report.result == "Blue"
    await kit.close()


async def test_a_request_still_open_when_its_turn_ends_is_withdrawn() -> None:
    outcomes: list[str] = []
    spawned: list[asyncio.Task[Any]] = []
    asked: list[str] = []
    human = HumanInputToolHandler(tool_names={"ask_user"}, timeout=2, tool_definitions=[ASK])

    async def waits_past_the_turn() -> None:
        pending = await human.handler.create("ask_user", {"question": "Later?"}, room_id="r1")
        asked.append(pending.pending_id)
        try:
            await human.handler.wait(pending.pending_id, timeout=2)
        except RuntimeError as exc:
            outcomes.append(str(exc))

    async def tool(name: str, arguments: dict[str, Any]) -> str:
        spawned.append(asyncio.create_task(waits_past_the_turn()))
        await asyncio.sleep(0.05)
        return "asked"

    other = AITool(name="later_ask", description="Ask.", parameters={"type": "object"})
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="later_ask", arguments={})])
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="Done.")])
    agent = AIChannel(
        "agent", provider=provider, human_input_handler=human, tools=[other], tool_handler=tool
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)

    await asyncio.wait_for(_say(kit), 5)
    await asyncio.wait_for(spawned[0], 5)

    # Withdrawn on both sides: the waiter reads it given up, the kit's door
    # answers it no more.
    assert outcomes == ["Released before an answer arrived"]
    assert await kit.answer_input(asked[0], "Too late") is False
    await kit.close()


async def test_a_detached_request_a_turn_raised_outlives_the_turn() -> None:
    """A runtime with a tool loop of its own raises a detached request, whose
    answer travels back another way and whose creator releases it
    (``create_detached``): the turn's end lets go of it, never withdraws it."""
    human = HumanInputHandler()
    raised: list[str] = []

    async def tool(name: str, arguments: dict[str, Any]) -> str:
        pending = await human.create_detached(
            "ask_user", {"question": "Ship it?"}, room_id="r1", channel_id="agent"
        )
        raised.append(pending.pending_id)
        return "asked; the answer comes back later"

    other = AITool(name="ask_later", description="Ask.", parameters={"type": "object"})
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="ask_later", arguments={})])
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="I asked.")])
    agent = AIChannel("agent", provider=provider, tools=[other], tool_handler=tool)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)

    await asyncio.wait_for(_say(kit), 5)

    # The owner's record went with the turn; the creator's way answers it.
    assert await kit.answer_input(raised[0], "From the kit") is False
    assert human.resolve(raised[0], "yes")
    assert await human.wait(raised[0], timeout=1) == "yes"
    await kit.close()


async def test_a_request_asked_after_its_turn_ended_is_the_handlers_alone() -> None:
    answered: list[str] = []
    spawned: list[asyncio.Task[Any]] = []
    turn_ended = asyncio.Event()
    human = HumanInputToolHandler(tool_names={"ask_user"}, timeout=2, tool_definitions=[ASK])

    async def later() -> None:
        await turn_ended.wait()  # asked once its turn is over
        pending = await human.handler.create(
            "ask_user", {"question": "Still there?"}, room_id="r1"
        )
        assert human.handler.resolve(pending.pending_id, "Yes")
        answered.append(await human.handler.wait(pending.pending_id, timeout=2))

    async def tool(name: str, arguments: dict[str, Any]) -> str:
        spawned.append(asyncio.create_task(later()))
        return "asked later"

    other = AITool(name="later_ask", description="Ask later.", parameters={"type": "object"})
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="later_ask", arguments={})])
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="Done.")])
    agent = AIChannel(
        "agent", provider=provider, human_input_handler=human, tools=[other], tool_handler=tool
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)

    await asyncio.wait_for(_say(kit), 5)
    turn_ended.set()
    await asyncio.wait_for(spawned[0], 5)

    assert answered == ["Yes"]
    await kit.close()


async def test_a_rejection_reaches_the_tool_as_a_refusal() -> None:
    kit, _, _, asked, reports = await _kit()
    turn = _say(kit)
    pending_id = await _asked(asked)

    assert await kit.reject_input(pending_id, "Not now") is True
    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.is_error
    assert "Not now" in str(report.result)
    await kit.close()


async def test_a_request_no_one_answers_in_time_is_closed_on_the_owner() -> None:
    kit, _, _, asked, reports = await _kit(timeout=0.02)
    turn = _say(kit)
    pending_id = await _asked(asked)

    await asyncio.wait_for(turn, 5)

    [report] = reports
    assert report.is_error
    assert await kit.answer_input(pending_id, "Too late") is False
    await kit.close()


async def test_a_request_a_hook_blocks_is_closed_on_the_owner() -> None:
    kit, _, _, _, reports = await _kit()
    denied: list[str] = []

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC, priority=-1)
    async def deny(event: PendingInputEvent, ctx: Any) -> HookResult:
        denied.append(event.pending_id)
        return HookResult.block("No humans today")

    await asyncio.wait_for(_say(kit), 5)

    [report] = reports
    assert report.is_error
    [pending_id] = denied
    # Settled on the executor: the owner takes no answer for it.
    assert await kit.answer_input(pending_id, "Too late") is False
    await kit.close()


# -- The owner's record ----------------------------------------------------------


class _Turn:
    """A turn of the owner's, as the record reads it."""

    def __init__(self, room_id: str = "r1") -> None:
        self.request = type("Request", (), {"room_id": room_id})()
        self.sent: list[InputAnswer] = []

    async def send_control(self, message: InputAnswer) -> None:
        self.sent.append(message)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _request(request_id: str) -> InputRequest:
    return InputRequest(request_id=request_id, tool_name="ask_user", arguments={})


async def test_an_answer_that_comes_before_its_request_is_held_for_it() -> None:
    inputs = OwnerInputs(hold=60)
    turn = _Turn()

    answering = asyncio.create_task(inputs.answer("p1", "Early"))
    await _settle()
    inputs.opened(turn, _request("p1"))  # type: ignore[arg-type]
    await _settle()
    assert turn.sent == [InputAnswer(request_id="p1", outcome="answered", value="Early")]
    assert not answering.done()

    inputs.taken(turn, "p1", True)  # type: ignore[arg-type]
    assert await answering is True


async def test_an_answer_held_past_its_bound_is_not_taken() -> None:
    inputs = OwnerInputs(hold=0.05)
    turn = _Turn()

    assert await inputs.answer("p1", "Early") is False
    inputs.opened(turn, _request("p1"))  # type: ignore[arg-type]
    await _settle()

    assert turn.sent == []


async def test_an_answer_the_request_settled_without_says_so() -> None:
    inputs = OwnerInputs(hold=60)
    turn = _Turn()
    inputs.opened(turn, _request("p1"))  # type: ignore[arg-type]

    answering = asyncio.create_task(inputs.answer("p1", "Second"))
    await _settle()
    # Settled first where the tool waits: the executor did not take it.
    inputs.taken(turn, "p1", False)  # type: ignore[arg-type]

    assert await answering is False
    assert await inputs.answer("p1", "Third") is False


async def test_an_answer_awaiting_its_verdict_when_the_turn_ends_was_not_taken() -> None:
    inputs = OwnerInputs(hold=60)
    turn = _Turn()
    inputs.opened(turn, _request("p1"))  # type: ignore[arg-type]

    answering = asyncio.create_task(inputs.answer("p1", "Answer"))
    await _settle()
    inputs.turn_ended(turn)  # type: ignore[arg-type]

    assert await answering is False


@pytest.mark.parametrize("ended", ["closed", "turn_ended"])
async def test_a_request_closed_or_whose_turn_ended_takes_no_answer(ended: str) -> None:
    inputs = OwnerInputs(hold=60)
    turn = _Turn()
    inputs.opened(turn, _request("p1"))  # type: ignore[arg-type]

    if ended == "closed":
        inputs.closed(turn, "p1")  # type: ignore[arg-type]
    else:
        inputs.turn_ended(turn)  # type: ignore[arg-type]

    assert await inputs.answer("p1", "Late") is False
    await _settle()
    assert turn.sent == []


async def test_a_request_belongs_to_the_turn_that_opened_it() -> None:
    inputs = OwnerInputs(hold=60)
    owner, other = _Turn(), _Turn()
    inputs.opened(owner, _request("p1"))  # type: ignore[arg-type]

    # Another turn opening, closing or judging the same id changes nothing.
    inputs.opened(other, _request("p1"))  # type: ignore[arg-type]
    inputs.closed(other, "p1")  # type: ignore[arg-type]
    answering = asyncio.create_task(inputs.answer("p1", "Mine"))
    await _settle()
    inputs.taken(other, "p1", False)  # type: ignore[arg-type]
    inputs.taken(owner, "p1", True)  # type: ignore[arg-type]

    assert await answering is True
    assert [a.value for a in owner.sent] == ["Mine"]
    assert other.sent == []


async def test_early_answers_are_held_up_to_a_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("roomkit.turns._owner_inputs.HELD_KEPT", 2)
    inputs = OwnerInputs(hold=60)
    answering = [asyncio.create_task(inputs.answer(rid, rid)) for rid in ("p1", "p2", "p3")]
    await _settle()

    # The oldest was let go, not taken; the others are still held.
    assert await answering[0] is False
    turn = _Turn()
    inputs.opened(turn, _request("p3"))  # type: ignore[arg-type]
    await _settle()
    assert [a.value for a in turn.sent] == ["p3"]
    for task in answering[1:]:
        task.cancel()
    await asyncio.gather(*answering[1:], return_exceptions=True)
