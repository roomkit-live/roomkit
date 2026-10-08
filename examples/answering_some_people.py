"""An agent that listens to everyone in the room but answers only some people.

Nova is Sylvain's assistant in the living room. A television is on, and a
guest, Paul, is there too: Nova hears them all, but answers only Sylvain.
``AnswerOnly`` wraps her speak policy (RFC §6.4): a turn from anyone else is
left silent, reason ``only listened to``, without asking the policy it wraps.

1. The television asks its viewers a question: Nova stays silent.
2. Sylvain asks her: she answers. The policy she has (``AlwaysSpeak`` here)
   judges with Sylvain alone as the person present, not the television.
3. Paul asks her to put music on: she stays silent.

``AnswerOnly`` wraps any policy: ``ClassifierSpeakPolicy`` on judgments, or
one of your own. A speaker is matched by the name the room gives them, which
is not an access control.

The model is a scripted mock, so the example runs without keys.

Run with:
    uv run python examples/answering_some_people.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    AlwaysSpeak,
    AnswerOnly,
    ChannelCategory,
    HookExecution,
    HookTrigger,
    RoomKit,
    SMSChannel,
    SpeakDecisionEvent,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

logger = setup_logging("example.answering_some_people")

LINES = [
    ("TV", "And now a question for our viewers: what is the capital of Australia?"),
    ("Sylvain", "Nova, do you know that one?"),
    ("Paul", "Nova, put some music on."),
]


async def main() -> None:
    kit = RoomKit()
    nova = AIChannel(
        "nova",
        provider=MockAIProvider(responses=["Canberra, not Sydney."]),
        system_prompt="You are Nova, Sylvain's assistant.",
        speak_policy=AnswerOnly(AlwaysSpeak(), people=["Sylvain"]),
    )
    kit.register_channel(nova)
    kit.register_channel(SMSChannel("living-room", provider=MockSMSProvider()))
    await kit.create_room(room_id="home")
    await kit.attach_channel("home", "nova", category=ChannelCategory.INTELLIGENCE)
    # Several voices on one channel: a group binding (RFC §10.4).
    await kit.attach_channel("home", "living-room", group=True)

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: object) -> None:
        logger.info("  → %s (%s)", event.decision.mode, event.decision.reason)

    for name, body in LINES:
        logger.info('%s: "%s"', name, body)
        await kit.process_inbound(
            InboundMessage(
                channel_id="living-room",
                sender_id=name.lower(),
                content=TextContent(body=body),
                metadata={"sender_name": name},
            )
        )
        await asyncio.sleep(0.1)

    answers = [e for e in await kit.store.list_events("home") if e.source.channel_id == "nova"]
    for answer in answers:
        logger.info("Nova: %s", answer.content.body)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
