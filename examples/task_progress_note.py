"""An agent that knows how far its background task got, without a tool call.

Nova hands a counting task to a worker. While it runs, the worker's tool says how
far it got with ``post_task_progress`` (RFC §23.3). When Sylvain asks "How far is
the counter?", Nova's turn already carries the room's tasks in its notes (RFC
§23.4): the task, how long it has run, its latest progress, and that its result
has not come back, so Nova answers from it and gives no data it does not have.

The hook below logs the notes Nova's turn reads; the providers are mocks, so the
example runs without keys.

Run with:
    uv run python examples/task_progress_note.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    AIGenerationEvent,
    ChannelCategory,
    HookResult,
    HookTrigger,
    RoomKit,
    SMSChannel,
    TextContent,
    split_turn_notes,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.tasks import post_task_progress

logger = setup_logging("example.task_progress_note")

SECONDS = 3
STEP = 0.5  # the example counts fast: one "second" every half second


def make_counter(kit: RoomKit) -> AIChannel:
    """A worker whose tool counts, saying how far it got at each step."""

    async def count(name: str, arguments: dict[str, Any]) -> str:
        for done in range(1, SECONDS + 1):
            await asyncio.sleep(STEP)
            await post_task_progress(kit, f"{done}/{SECONDS} s")
        return f"Counted {SECONDS} seconds."

    return AIChannel(
        "counter",
        provider=MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="c1", name="count", arguments={})],
                ),
                AIResponse(content=f"Counted {SECONDS} seconds."),
            ]
        ),
        tool_handler=count,
        tools=[AITool(name="count", description="Count seconds.")],
        tool_search=False,
    )


async def main() -> None:
    kit = RoomKit()
    nova = AIChannel(
        "nova",
        provider=MockAIProvider(
            ["The counter is at 2 of 3 seconds.", f"The counter is done: {SECONDS} seconds."]
        ),
        system_prompt="You are Nova, an assistant.",
    )
    kit.register_channel(nova)
    kit.register_channel(make_counter(kit))
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    await kit.create_room(room_id="chat")
    await kit.attach_channel("chat", "nova", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("chat", "sms")

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def show_notes(event: AIGenerationEvent, ctx: object) -> HookResult:
        if event.channel_id != "nova":  # the worker's own turns, in its task's room
            return HookResult.allow()
        _, notes = split_turn_notes(str(event.ai_context.messages[-1].content))
        logger.info("  Nova's turn notes:\n%s", notes)
        return HookResult.allow()

    task = await kit.delegate("chat", "counter", f"count {SECONDS} s", notify="nova")
    await asyncio.sleep(STEP * 2.5)

    logger.info('Sylvain: "How far is the counter?"')
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id="sylvain",
            content=TextContent(body="How far is the counter?"),
        )
    )
    await task.wait(timeout=10)
    await asyncio.sleep(0.2)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
