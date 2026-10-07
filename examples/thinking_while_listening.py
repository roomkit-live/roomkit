"""An agent that thinks while it listens, and raises its hand when it knows.

Two people meet in a room with an agent, Nova, who knows what the team's
licences cost. Her AI channel has a speak policy and a thinker (RFC §6.4): on a
message she leaves unanswered, the thinker writes what she has in mind (what she
makes of it, what she would say if given the turn); back within the wait with
something to say, the policy decides again with that thought.

1. "Right, we need to order the licences for the team." goes to nobody: Nova
   listens, and thinks she knows the price.
2. "Does anyone know how much it costs, per person?" asks nobody in particular,
   and her thought answers it: she offers.
3. "Nova, go ahead, tell us." asks her: she speaks, her thought in the turn's
   notes, and what she wanted to say is then said.

``ON_THOUGHT`` shows each thought, ``ON_SPEAK_DECISION`` each decision.

``CLASSIFIER`` (``mock``, ``jev``) chooses the policy's judgments and ``THINKER``
(``mock``, ``anthropic``) the thinker's model; the defaults need no key. With
``jev``: ``pip install roomkit[typesafe]`` and ``TYPESAFE_API_KEY``; with
``anthropic``: ``ANTHROPIC_API_KEY``. Nova's own answers come from a scripted
model.

Run with:
    uv run python examples/thinking_while_listening.py
    CLASSIFIER=jev THINKER=anthropic uv run python examples/thinking_while_listening.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import (
    ChannelCategory,
    Classifier,
    ClassifierSpeakPolicy,
    HookExecution,
    HookTrigger,
    JevClassifier,
    LLMThinker,
    MockClassifier,
    MockThinker,
    RoomKit,
    SMSChannel,
    SpeakDecisionEvent,
    TextContent,
    Thinker,
    Thought,
    ThoughtEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.sms.mock import MockSMSProvider

logger = setup_logging("example.thinking_while_listening")

NOVA = (
    "You are Nova, the team's assistant. You know that the design tool's licence costs "
    "$1,200 a year per person, and that the vendor gives 10% off beyond ten licences."
)

LINES = [
    ("Sylvain", "Right, we need to order the licences for the team."),
    ("Paul", "Does anyone know how much it costs, per person?"),
    ("Sylvain", "Nova, go ahead, tell us."),
]

PRICE = Thought(
    "They are about to order the licences; I know the price.",
    ("A licence costs $1,200 a year per person, with 10% off beyond ten.",),
)

# The judgments and thoughts a run on Jev and Claude Haiku gave, for the mocks:
# per policy call (the first line is asked again once the thought is in).
JUDGMENTS = [
    {"directness": 0.0, "unfinished": 0.06, "request": 0.48},
    {"directness": 0.0, "unfinished": 0.06, "request": 0.52, "answers": 0.46},
    {"directness": 0.07, "unfinished": 0.11, "request": 0.82, "answers": 0.96},
    {"directness": 2.99, "request": 0.96, "answered": 0.91},
]


def make_classifier() -> Classifier:
    if os.environ.get("CLASSIFIER", "mock") == "jev":
        return JevClassifier(require_env("TYPESAFE_API_KEY")["TYPESAFE_API_KEY"])
    return MockClassifier(JUDGMENTS)


def make_thinker() -> Thinker:
    if os.environ.get("THINKER", "mock") == "anthropic":
        key = require_env("ANTHROPIC_API_KEY")["ANTHROPIC_API_KEY"]
        config = AnthropicConfig(api_key=key, model="claude-haiku-4-5-20251001")
        return LLMThinker(AnthropicAIProvider(config))
    return MockThinker([PRICE])


async def main() -> None:
    kit = RoomKit()
    classifier = make_classifier()
    thinker = make_thinker()
    nova = AIChannel(
        "nova",
        provider=MockAIProvider(
            responses=[
                "I think I know the price: would you like me to tell you?",
                "$1,200 a year per person, and 10% off beyond ten licences.",
            ]
        ),
        system_prompt=NOVA,
        speak_policy=ClassifierSpeakPolicy(classifier, agent_name="Nova"),
        thinker=thinker,
        think_wait=4.0,
    )
    kit.register_channel(nova)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    await kit.create_room(room_id="meeting")
    await kit.attach_channel("meeting", "nova", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("meeting", "sms")
    # Both are in the meeting from the start: one person alone would make every
    # request Nova's.
    for name in ("Sylvain", "Paul"):
        await kit.ensure_participant("meeting", "sms", name.lower(), display_name=name)

    @kit.hook(HookTrigger.ON_THOUGHT, execution=HookExecution.ASYNC)
    async def on_thought(event: ThoughtEvent, ctx: object) -> None:
        logger.info("  Nova thinks: %s %s", event.thought.text, list(event.thought.want_to_say))

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: object) -> None:
        decision = event.decision
        judged = " ".join(f"{k}={v:.2f}" for k, v in decision.judgments.items() if v >= 0.05)
        logger.info("  → %s (%s) %s", decision.mode, decision.reason, judged)

    for name, body in LINES:
        logger.info('%s: "%s"', name, body)
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id=name.lower(),
                content=TextContent(body=body),
                metadata={"sender_name": name},
            )
        )
        await asyncio.sleep(0.2)

    answers = [e for e in await kit.store.list_events("meeting") if e.source.channel_id == "nova"]
    for answer in answers:
        logger.info("Nova: %s", answer.content.body)
    await kit.close()
    await classifier.close()
    await thinker.close()


if __name__ == "__main__":
    asyncio.run(main())
