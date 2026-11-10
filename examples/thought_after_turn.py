"""A thought that lands after its turn ended still reaches the room.

Nova listens in a room (her speak policy keeps her quiet) and thinks while
she listens (RFC §6.4). Her thinker takes longer than the turn: the turn
returns silent, and the thought lands afterwards. An ``ON_THOUGHT`` hook on
the executor posts it on the kit's status bus through ``current_turn_host()``
(RFC §10.6.5):

1. Nova's turns run on a host's executor; the kit holds her by her
   declaration alone.
2. The hook fires after the turn ended. Its host is still there for the verbs
   that write the room (``post_status``, ``send_event``,
   ``update_room_metadata``, ``add_observation``): they cross to the kit
   outside the turn's link, at the turn's chain depth (RFC §10.6.3).
3. The kit accepts it because Nova's executor took the room's last turn; work
   of an earlier turn, once a later one was sent, would be refused.

The models are scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/thought_after_turn.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    ChannelDeclaration,
    HookExecution,
    HookTrigger,
    InboundMessage,
    InMemoryStore,
    MockSpeakPolicy,
    MockThinker,
    RoomKit,
    SMSChannel,
    TextContent,
    Thought,
    ThoughtEvent,
    current_turn_host,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.providers.sms import MockSMSProvider
from roomkit.turns import InMemoryTurnBackend, TurnExecutor

logger = setup_logging("example.thought_after_turn")

PHONE = "+15145550100"
ROOM = "planning"
THOUGHT = Thought(text="They need the licence price: 1,200 $ a year per person.")


def build_nova(channel_id: str) -> AIChannel | None:
    """The host's factory: Nova listens, and thinks slower than her turn."""
    if channel_id != "nova":
        return None
    return AIChannel(
        "nova",
        provider=MockAIProvider(responses=["I can help with that."]),
        speak_policy=MockSpeakPolicy(["silent"]),
        thinker=MockThinker([THOUGHT], delay=0.3),
        think_wait=0.05,
    )


async def main() -> None:
    store = InMemoryStore()
    backend = InMemoryTurnBackend()
    executor = TurnExecutor(
        build_nova,
        hosts=lambda channel_id: channel_id == "nova",
    )

    @executor.hook(HookTrigger.ON_THOUGHT, execution=HookExecution.ASYNC)
    async def post_thought(event: ThoughtEvent, ctx: object) -> None:
        host = current_turn_host()
        if host is None:
            return
        logger.info("  thought after its turn (turn still running: %s)", host.open)
        await host.post_status("nova", "thought", "info", detail=event.thought.text)

    await backend.serve(executor)
    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(ChannelDeclaration(channel_id="nova"))

    async def show(entry: object) -> None:
        if getattr(entry, "action", None) == "thought":
            logger.info("  kit's status bus: %s", getattr(entry, "detail", ""))

    await kit.status_bus.subscribe(show)
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms", metadata={"phone_number": PHONE})
    await kit.attach_channel(ROOM, "nova", category=ChannelCategory.INTELLIGENCE)

    logger.info("[person] Right, we need to order the licences for the team.")
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id=PHONE,
            content=TextContent(body="Right, we need to order the licences for the team."),
        ),
        room_id=ROOM,
    )
    logger.info("the turn returned; Nova is still thinking")
    await asyncio.sleep(0.6)

    await kit.close()
    await executor.close()


if __name__ == "__main__":
    asyncio.run(main())
