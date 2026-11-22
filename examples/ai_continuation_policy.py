"""Go on when a model announces an action and stops: an AI channel's continuation policy.

A model answers "I will check the run." and stops, without calling the tool it
announced. The channel's continuation policy recognizes the announcement and
hands the model an instruction to go on; the next round calls the tool and
answers. Shows:
- AIChannel(continuation=...): the text of a round the model ended itself,
  without a call, in; the instruction to go on, or None, out
- The one stop reason it reads, whatever the provider: ``stop``, which each
  provider maps its vendor's own word onto
- The bound it shares with the empty round (max_empty_retries), and the
  ``unfinished`` end of a turn whose policy still asks once it has run out

Run with:
    uv run python examples/ai_continuation_policy.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    EventType,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider

logger = setup_logging("ai_continuation_policy")

STATUS = AITool(name="run_status", description="Read a run's status", parameters={})


def go_on_after_an_announcement(text: str) -> str | None:
    """Continue an answer that only announced a check; let any other answer stand."""
    if text.lower().startswith(("i will check", "let me check")):
        return "You announced a check but did not run it: run it now and report the result."
    return None


async def run_status(name: str, arguments: dict[str, Any]) -> str:
    return "finished at 12:07, no errors"


def scripted_model() -> MockAIProvider:
    """A model that announces a check and stops, then calls the tool once told to."""
    return MockAIProvider(
        ai_responses=[
            # The model ended its answer itself: the policy reads it.
            AIResponse(content="I will check the run.", finish_reason="stop"),
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="run_status", arguments={})],
            ),
            AIResponse(content="The run finished at 12:07 without errors.", finish_reason="stop"),
        ]
    )


def announcing_model() -> MockAIProvider:
    """A model that announces the check again when told to go on."""
    return MockAIProvider(
        ai_responses=[
            AIResponse(content="I will check the run.", finish_reason="stop"),
            AIResponse(content="I will check it right away.", finish_reason="stop"),
        ]
    )


async def run_turn(model: MockAIProvider) -> None:
    """One member question to an agent with the policy; log what the room keeps."""
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("member"))
    kit.register_channel(
        AIChannel(
            "agent",
            provider=model,
            tools=[STATUS],
            tool_handler=run_status,
            continuation=go_on_after_an_announcement,
            # One try, shared with an empty round: the default.
            max_empty_retries=1,
        )
    )
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "member")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(
            channel_id="member", sender_id="u1", content=TextContent(body="Did it run?")
        )
    )

    # The announcement and what follows it are two messages, never one run-on.
    for event in await kit.store.list_events("room"):
        if event.type == EventType.MESSAGE and event.source.channel_id == "agent":
            reason = event.metadata.get("loop_end_reason", "")
            logger.info("agent: %s %s", event.content.body, f"[{reason}]" if reason else "")
    logger.info("model rounds: %d", len(model.calls))
    await kit.close()


async def main() -> None:
    logger.info("A model that acts once told to go on:")
    await run_turn(scripted_model())
    logger.info("A model that announces again: the turn ends unfinished.")
    await run_turn(announcing_model())


if __name__ == "__main__":
    asyncio.run(main())
