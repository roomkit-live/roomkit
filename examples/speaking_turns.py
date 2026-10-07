"""An agent that does not answer every turn: a speak policy on its AI channel.

Two people talk in one room with an agent, Nova. Nova speaks when someone names
her, offers when a question goes to nobody in particular, and otherwise listens
(RFC §6.4):

1. "Paul, do you have the September numbers?" goes to Paul: Nova stays silent.
   No turn runs, nothing is generated, and the message is stored all the same.
2. "Nova, can you sum up?" names her: she speaks.
3. "Does anyone know when the demo is?" asks nobody in particular: she offers,
   in one short sentence, what she could add, without giving it.

Every decision reaches ``ON_SPEAK_DECISION`` with its reason and what the policy
weighed, so what Nova left unanswered, and why, can be followed and measured.
The policy here reads keywords; the same seam takes one built on judgments.

The model is a scripted mock, so the example runs without keys.

Run with:
    uv run python examples/speaking_turns.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    RoomKit,
    SMSChannel,
    SpeakDecision,
    SpeakDecisionEvent,
    SpeakPolicy,
    SpeakTurn,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

logger = setup_logging("example.speaking_turns")


class SpeaksWhenNamed(SpeakPolicy):
    """Speaks when named, offers on a question to nobody, listens otherwise."""

    def __init__(self, name: str, people: tuple[str, ...]) -> None:
        self._name = name.lower()
        self._people = tuple(p.lower() for p in people)

    async def decide(self, turn: SpeakTurn) -> SpeakDecision:
        text = getattr(turn.event.content, "body", "").lower()
        named = self._name in text
        to_someone = any(p in text for p in self._people)
        question = text.rstrip().endswith("?")
        judgments = {
            "named": float(named),
            "to_someone": float(to_someone),
            "question": float(question),
        }
        if named:
            return SpeakDecision("speak", reason="named", judgments=judgments)
        if question and not to_someone:
            return SpeakDecision("offer", reason="a question to nobody", judgments=judgments)
        return SpeakDecision("silent", reason="not for me", judgments=judgments)


async def main() -> None:
    kit = RoomKit()
    model = MockAIProvider(
        responses=[
            "We talked about the September numbers, which Paul will send.",
            "I can look up the demo date if you like.",
        ]
    )
    nova = AIChannel(
        "nova",
        provider=model,
        system_prompt="You are Nova, the team's assistant.",
        speak_policy=SpeaksWhenNamed("Nova", people=("Paul", "Sylvain")),
    )
    kit.register_channel(nova)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    await kit.create_room(room_id="meeting")
    await kit.attach_channel("meeting", "nova", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("meeting", "sms")

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: object) -> None:
        body = getattr(event.event.content, "body", "")
        logger.info(
            '%-6s %-22s "%s" %s',
            event.decision.mode,
            event.decision.reason,
            body,
            event.decision.judgments,
        )

    lines = [
        ("sylvain", "Paul, do you have the September numbers?"),
        ("paul", "I'll send them tomorrow morning."),
        ("sylvain", "Nova, can you sum up?"),
        ("paul", "Does anyone know when the demo is?"),
    ]
    for sender, body in lines:
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id=sender, content=TextContent(body=body))
        )
        await asyncio.sleep(0.1)

    events = await kit.store.list_events("meeting")
    said = [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "sms"]
    answers = [e for e in events if e.source.channel_id == "nova"]
    logger.info("%d messages stored, %d answered by Nova:", len(said), len(answers))
    for answer in answers:
        logger.info("  Nova: %s", answer.content.body)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
