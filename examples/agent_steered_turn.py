"""Write to an agent while it works: its running turn reads the message.

A person asks an agent to list the files; while its tool runs, they add
"Only the .py files." With ``steer_room_turns=True`` that second message is
stored in the room like any message, and handed to the turn already running
instead of starting a second turn beside it: the model reads it before its next
round and answers once, knowing both. A message that comes while the model is
writing its answer is read too: the same turn asks the model again.

Run with: uv run python examples/agent_steered_turn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
from typing import Any

from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIContext, AIResponse, AITool, AIToolCall, MockAIProvider

logger = setup_logging("agent_steered_turn")

LIST_FILES = AITool(
    name="list_files", description="List the project's files.", parameters={"type": "object"}
)


def _text(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "text", "") for part in content)


class Coder(MockAIProvider):
    """Lists the files first, then answers from what the person said last."""

    async def generate(self, context: AIContext) -> AIResponse:
        said = [_text(m.content) for m in context.messages if m.role == "user"]
        logger.info("The model reads: %s", said)
        if not any(m.role == "tool" for m in context.messages):
            return AIResponse(
                content="", tool_calls=[AIToolCall(id="c1", name="list_files", arguments={})]
            )
        if "Only the .py files." in said[-1]:
            return AIResponse(content="a.py, main.py")
        return AIResponse(content="a.py, main.py, notes.txt")


async def main() -> None:
    listing = asyncio.Event()
    go_on = asyncio.Event()

    async def list_files(name: str, arguments: dict[str, Any]) -> str:
        listing.set()
        await go_on.wait()  # a slow tool: the person writes meanwhile
        return "a.py main.py notes.txt"

    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(
        Agent(
            "coder",
            provider=Coder(),
            tools=[LIST_FILES],
            tool_handler=list_files,
            steer_room_turns=True,
        )
    )
    await kit.create_room(room_id="session")
    await kit.attach_channel("session", "ws")
    await kit.attach_channel("session", "coder", category=ChannelCategory.INTELLIGENCE)

    def said(body: str) -> InboundMessage:
        return InboundMessage(channel_id="ws", sender_id="dev", content=TextContent(body=body))

    turn = asyncio.create_task(kit.process_inbound(said("List the files."), room_id="session"))
    await listing.wait()
    logger.info("While the tool runs, the person writes: 'Only the .py files.'")
    await kit.process_inbound(said("Only the .py files."), room_id="session")
    go_on.set()
    await turn

    for event in await kit.store.list_events("session"):
        body = getattr(event.content, "body", None)
        if body and event.source.channel_id != "system":
            logger.info("%-6s %s", event.source.channel_id, body)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
