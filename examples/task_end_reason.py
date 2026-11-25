"""A task that did not complete tells its requester why.

The assistant hands a review to a reviewer agent whose turn is capped at one
round of tool calls. The reviewer reads a file, wants to read another, and its
turn is cut there: the task fails with no answer. The assistant is handed the
failure back, and reads why in it (RFC §23.3 step 8):

1. The hand-back says ``failed: it reached its limit of tool rounds before
   giving an answer (max_rounds)``: the runtime's words, then a stable code.
2. ``ON_TASK_COMPLETED``, the task's entry on the StatusBus and the hand-back's
   metadata carry the same code, ``task_end_reason``.
3. ``task_status`` answers with it too (``end_reason``, ``end_reason_text``),
   when the person asks why the review stopped.

The code comes from a closed list: a turn cut short (``max_rounds``,
``timeout``, ``budget_exceeded``...), a task cancelled (``cancelled``), a worker
that answered nothing (``empty_response``), and any other failure ``error``,
whose detail stays in the logs and the hooks. The
reviewer's last narration stays on the task's result, for the host: no model
reads it as an answer.

The models are scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/task_end_reason.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    AIChannel,
    ChannelCategory,
    EventType,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.providers.sms import MockSMSProvider
from roomkit.tasks import TASK_STATUS_TOOL, room_tasks

logger = setup_logging("example.task_end_reason")

PHONE = "+15145550100"
ROOM = "review-chat"


def _calls(call_id: str, name: str) -> AIResponse:
    call = AIToolCall(id=call_id, name=name, arguments={})
    return AIResponse(content="", finish_reason="tool_calls", tool_calls=[call])


async def _read_file(name: str, arguments: dict[str, object]) -> str:
    return "def placed(): ..."


def make_reviewer() -> Agent:
    """A reviewer that would read file after file, capped at one round."""
    return Agent(
        "reviewer",
        provider=MockAIProvider(ai_responses=[_calls(f"r{n}", "read_file") for n in range(3)]),
        role="Code reviewer",
        tools=[AITool(name="read_file", description="Read a file of the change.")],
        tool_handler=_read_file,
        max_tool_rounds=1,
        tool_search=False,
    )


def make_assistant() -> AIChannel:
    """The assistant: it says the hand-back, then reads ``task_status``."""
    return AIChannel(
        "assistant",
        provider=MockAIProvider(
            ai_responses=[
                AIResponse(content="The review stopped at its limit of tool rounds."),
                _calls("s1", TASK_STATUS_TOOL),
                AIResponse(content="It ran out of tool rounds before it could answer."),
            ]
        ),
        system_prompt="You are the assistant. You hand reviews to the reviewer.",
    )


def show_what_ends(kit: RoomKit) -> None:
    """Print what the host's hooks see of the task's end, and what is said."""

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event, ctx):
        meta = event.metadata
        print(f"  ⚙ ON_TASK_COMPLETED: {meta['task_status']}, why: {meta.get('task_end_reason')}")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def on_hand_back(event, ctx):
        if "task_id" in event.metadata:
            header = event.content.body.splitlines()[0]
            print(f"  ⚙ handed back ({event.metadata.get('task_end_reason')}): {header}")
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.MESSAGE})
    async def show(event, ctx):
        if event.source.channel_id == "assistant":
            print(f"  🗣 assistant: {event.content.body}")
        return HookResult.allow()


async def build_kit() -> RoomKit:
    """A room where the assistant answers, and a reviewer it can be handed."""
    kit = RoomKit()
    assistant = make_assistant()
    for channel in (SMSChannel("sms", provider=MockSMSProvider()), assistant, make_reviewer()):
        kit.register_channel(channel)
    # task_status, served by the room's owner wherever the assistant's turns run.
    kit.setup_task_tools(assistant)
    show_what_ends(kit)
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms", metadata={"phone_number": PHONE})
    await kit.attach_channel(ROOM, "assistant", category=ChannelCategory.INTELLIGENCE)
    return kit


async def main() -> None:
    kit = await build_kit()

    print("[host] delegates the review to the reviewer, capped at one tool round")
    task = await kit.delegate(ROOM, "reviewer", "Review the placement change.", notify="assistant")
    result = await task.wait(timeout=10)
    await asyncio.sleep(0.2)  # the hand-back's turn
    print(f"  ⚙ the task's result: {result.status}, error: {result.error}")

    print("[person] Why did the review stop?")
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id=PHONE,
            content=TextContent(body="Why did the review stop?"),
        ),
        room_id=ROOM,
    )

    [line] = room_tasks(await kit.status_bus.recent(50))
    print("\n  The task's line, as task_status answers it:")
    print("  " + json.dumps({k: line[k] for k in ("status", "end_reason", "end_reason_text")}))
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
