"""Human-in-the-loop tool calls with HumanInputToolHandler.

Demonstrates how to pause the AI tool loop when a tool needs
human input, notify the application, and resume with the
user's answer.

Shows:
- HumanInputToolHandler intercepting specific tool names
- ON_USER_INPUT_REQUIRED hook for notifications
- Async resolution simulating a user answering
- Composition with other tool handlers

Run with:
    uv run python examples/ai_human_input.py
"""

from __future__ import annotations

import asyncio
import json
import logging

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    HumanInputToolHandler,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("example")


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

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC)
    async def on_input_needed(event, ctx):
        logger.info(
            "Human input required: pending_id=%s tool=%s args=%s",
            event.pending_id,
            event.tool_name,
            event.arguments,
        )
        # In a real app: broadcast to frontend via WebSocket
        # await ws_manager.broadcast(event.room_id, {...})

        # Simulate user answering after a short delay
        async def _simulate_user_answer():
            await asyncio.sleep(0.5)
            answer = json.dumps({"answers": [{"answer": "Dark"}]})
            logger.info("User answered: %s", answer)
            human.handler.resolve(event.pending_id, answer)

        asyncio.create_task(_simulate_user_answer())
        return HookResult.allow()

    # --- Collect responses ---------------------------------------

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

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
            logger.info("AI: %s", event.content.body)

    # --- Cleanup -------------------------------------------------

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
