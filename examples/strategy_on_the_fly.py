"""Change a room's orchestration while it lives: install_strategy / uninstall_strategy.

A room starts as a plain chat with one assistant. Then the person brings a
team into it: a Discussion, whose agents read the conversation so far. Later
the room becomes a Swarm, where the assistant hands the person over to
billing. Each install attaches the strategy's agents; each uninstall takes
back what its install added (hooks, tools, turn runners, the agents it
attached, its state), and the timeline stays (RFC §19.7).

The models are scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/strategy_on_the_fly.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    Discussion,
    EventType,
    InboundMessage,
    RoomKit,
    Swarm,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

logger = setup_logging("example.strategy_on_the_fly")

ROOM = "support-room"


def scripted(*answers: AIResponse | str) -> MockAIProvider:
    return MockAIProvider(
        ai_responses=[a if isinstance(a, AIResponse) else AIResponse(content=a) for a in answers]
    )


def handoff(target: str) -> AIResponse:
    arguments = {"target": target, "reason": "a billing question", "summary": "invoice 40/20"}
    return AIResponse(
        content="",
        tool_calls=[AIToolCall(id="h1", name="handoff_conversation", arguments=arguments)],
    )


async def say(kit: RoomKit, body: str) -> None:
    logger.info("you: %s", body)
    await kit.process_inbound(
        InboundMessage(channel_id="you", sender_id="you", content=TextContent(body=body))
    )
    await asyncio.sleep(0.3)  # a discussion answers in turns, after the call returns


async def show_new(kit: RoomKit, seen: set[str]) -> None:
    for event in await kit.store.list_events(ROOM, limit=100):
        if event.id in seen or event.type != EventType.MESSAGE:
            continue
        seen.add(event.id)
        if event.source.channel_id != "you" and isinstance(event.content, TextContent):
            logger.info("  %s: %s", event.source.channel_id, event.content.body)


async def main() -> None:
    assistant = Agent(
        "assistant",
        provider=scripted(
            "Hi! What can I do for you?",
            "Let me bring in people who know the code.",
            handoff("billing"),
            "Billing will take it from here.",
        ),
        role="Assistant",
    )
    dev = Agent(
        "dev",
        provider=scripted("Release 4.2 changed the invoice rounding; fixing it now."),
        role="Developer",
    )
    billing = Agent("billing", provider=scripted("I refunded the 20 difference."), role="Billing")

    kit = RoomKit()
    kit.register_channel(WebSocketChannel("you"))
    kit.register_channel(assistant)
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "you")
    await kit.attach_channel(ROOM, "assistant", category=ChannelCategory.INTELLIGENCE)
    seen: set[str] = set()

    logger.info("1. a plain chat with one assistant")
    await say(kit, "hello")
    await say(kit, "my invoice says 40 instead of 20")
    await show_new(kit, seen)

    logger.info("2. the person brings a team in: a Discussion, installed on the live room")
    await kit.install_strategy(ROOM, Discussion([assistant, dev]))
    await say(kit, "@dev can you look at the invoice bug?")
    await show_new(kit, seen)
    logger.info("   dev's turn read the whole chat: %s", _read_by(dev))

    logger.info("3. back to support: the Discussion out, a Swarm in")
    await kit.uninstall_strategy(ROOM)
    await kit.install_strategy(ROOM, Swarm(agents=[assistant, billing], entry="assistant"))
    await say(kit, "and the refund?")
    await say(kit, "thanks, is it done?")
    await show_new(kit, seen)

    bound = sorted(b.channel_id for b in await kit.list_bindings(ROOM))
    logger.info("   bound now: %s (dev left with the Discussion)", bound)
    await kit.close()


def _read_by(agent: Agent) -> str:
    calls = agent._provider.calls  # type: ignore[attr-defined]
    text = " ".join(str(m.content) for m in calls[0].messages) if calls else ""
    return "yes" if "40 instead of 20" in text else "no"


if __name__ == "__main__":
    asyncio.run(main())
