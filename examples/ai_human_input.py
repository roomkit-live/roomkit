"""Human-in-the-loop tool calls with HumanInputToolHandler.

Demonstrates how to pause the AI tool loop when a tool needs
human input, notify the application, and resume with the
user's answer.

Shows:
- HumanInputToolHandler intercepting specific tool names
- ON_USER_INPUT_REQUIRED hook for notifications
- An answer arriving while the notification is still in flight —
  the tool resumes on the answer, not on the notification
- ``actor_id`` on the request: which participant's turn raised it
- create_detached() / release() for a runtime that owns its tool loop

Run with:
    uv run python examples/ai_human_input.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    HumanInputHandler,
    HumanInputToolHandler,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("example")

# A notification that takes its time — a slow WebSocket fan-out, a hook
# burning its budget.  The user answers long before it returns.
SLOW_NOTIFY_SECONDS = 1.0
USER_ANSWERS_AFTER = 0.1


async def main() -> None:
    kit = RoomKit()

    # --- Channels ------------------------------------------------

    ws = WebSocketChannel("ws-user")

    # MockAIProvider that first calls AskUserQuestion, then uses the answer.
    provider = MockAIProvider(
        ai_responses=[
            # Round 1: AI calls AskUserQuestion tool
            AIResponse(
                content="",
                finish_reason="tool_use",
                tool_calls=[
                    AIToolCall(
                        id="tc-1",
                        name="AskUserQuestion",
                        arguments={
                            "questions": [
                                {
                                    "question": "What color theme do you prefer?",
                                    "options": ["Dark", "Light", "System"],
                                }
                            ]
                        },
                    )
                ],
            ),
            # Round 2: AI uses the answer
            AIResponse(
                content="Great choice! I'll set up the dark theme for you.",
                finish_reason="stop",
            ),
        ]
    )

    # Human-input handler: intercepts AskUserQuestion calls
    human = HumanInputToolHandler(
        tool_names={"AskUserQuestion"},
        timeout=30,
        # The definition is what puts the tool in the turn's resolved toolset.
        # Name it here (or declare it on the channel) or the loop drops the
        # call as unoffered and the human is never asked.
        tool_definitions=[
            AITool(
                name="AskUserQuestion",
                description="Ask the user a question and wait for the answer.",
                parameters={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "options": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["question"],
                            },
                        }
                    },
                    "required": ["questions"],
                },
            )
        ],
    )

    ai = AIChannel(
        "ai-agent",
        provider=provider,
        system_prompt="You are a helpful assistant. Ask the user questions when needed.",
        human_input_handler=human,
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    # --- Hook: notify when human input is needed -----------------

    started = time.perf_counter()

    def elapsed() -> str:
        return f"t={time.perf_counter() - started:.2f}s"

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC)
    async def on_input_needed(event, ctx):
        logger.info(
            # ``actor_id`` is who to ask: the participant whose turn raised the
            # request. Without it a notification layer can only broadcast, and
            # whoever answers first answers for someone else.
            "%s | notification started: pending_id=%s tool=%s actor=%s args=%s",
            elapsed(),
            event.pending_id,
            event.tool_name,
            event.actor_id,
            event.arguments,
        )

        # Simulate user answering while the notification is still running
        async def _simulate_user_answer():
            await asyncio.sleep(USER_ANSWERS_AFTER)
            answer = json.dumps({"answers": [{"answer": "Dark"}]})
            logger.info("%s | user answered: %s", elapsed(), answer)
            human.handler.resolve(event.pending_id, answer)

        asyncio.create_task(_simulate_user_answer())

        # In a real app: broadcast to frontend via WebSocket
        # await ws_manager.broadcast(event.room_id, {...})
        await asyncio.sleep(SLOW_NOTIFY_SECONDS)
        logger.info("%s | notification finished", elapsed())
        return HookResult.allow()

    # --- Collect responses ---------------------------------------

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)
        if hasattr(event.content, "body"):
            # Lands right after the answer, not after the notification.
            logger.info("%s | AI reply delivered", elapsed())

    ws.register_connection("user-conn", on_recv, room_id="demo-room")

    # --- Room setup ----------------------------------------------

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-user")
    await kit.attach_channel("demo-room", "ai-agent", category=ChannelCategory.INTELLIGENCE)

    # --- Send a message that triggers the AI tool loop -----------

    logger.info("Sending user message...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Set up my workspace"),
        )
    )

    # Wait for the tool loop to complete (includes human input pause)
    await asyncio.sleep(2)

    # --- Show results --------------------------------------------

    logger.info("--- AI responses received ---")
    for event in inbox:
        if hasattr(event.content, "body"):
            logger.info("%s | AI: %s", elapsed(), event.content.body)

    # --- Cleanup -------------------------------------------------

    await kit.close()

    await detached_request()


async def detached_request() -> None:
    """A runtime that owns its tool loop: create_detached() / release().

    Claude Code and other external runtimes raise the request through
    RoomKit but carry the answer back their own way — nothing here calls
    wait(), so nothing here would retire the request.  Saying so at the
    call site is what makes the cleanup someone's job.
    """
    logger.info("--- detached request (external runtime path) ---")
    handler = HumanInputHandler()

    async def notify(event) -> bool:
        logger.info("notify: pending_id=%s tool=%s", event.pending_id, event.tool_name)
        return True

    handler._on_input_required = notify

    pending = await handler.create_detached(
        "AskUserQuestion",
        {"questions": [{"question": "Ship it?", "options": ["Yes", "No"]}]},
        room_id="demo-room",
    )
    await asyncio.sleep(0)  # the notification runs here, off the answering path
    handler.resolve(pending.pending_id, json.dumps({"answers": [{"answer": "Yes"}]}))
    handler.release(pending.pending_id)

    logger.info("active requests after release: %d", len(handler.pending))
    # The answer outlives the request: a late reader still reads it.
    logger.info("answer still readable: %s", await handler.wait(pending.pending_id, timeout=1))


if __name__ == "__main__":
    asyncio.run(main())
