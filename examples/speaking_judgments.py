"""A speak policy on judgments: whether Nova speaks, decided by a classifier.

Two people meet in a room with an agent, Nova. Her AI channel carries a
``ClassifierSpeakPolicy``: on each message, one classifier call answers narrow
questions about it (how directly is Nova brought in, did the speaker finish,
are they asking her to keep quiet...) and plain code composes the answers
(RFC §6.4):

1. "Paul, do you have the September numbers?" goes to Paul: silent.
2. "Nova, can you sum up where we are?" asks her: she speaks.
3. "Maybe Nova has the demo date somewhere." only wonders about her: she
   offers.
4. "Nova, don't answer, just listen for now." puts the room in the listening
   state the policy keeps: silent.
5. "Right, the budget, how much do we set it to?" asks the room: silent, the
   room listens.
6. "Nova, what did we spend on it last year?" is put to her: she answers, and
   the room goes on listening.
7. "So we keep the same figure." silent.
8. "Thanks Nova, feel free to chime in again." lets her talk again: the room
   is open, and the turn decided as usual.

Every decision reaches ``ON_SPEAK_DECISION`` with its reason and each judgment.
Told the languages Nova answers in, the policy also judges the speaker's, and
the turn's notes say it.

The classifier is chosen by ``CLASSIFIER``:

- ``mock`` (default): answers scripted from a run on Jev, no key needed;
- ``jev``: TypeSafe's Jev, calibrated (``pip install roomkit[typesafe]``,
  ``TYPESAFE_API_KEY``);
- ``openai``: OpenAI's Decisions API on ``gpt-6-luna``, with probabilities
  (``pip install roomkit[openai]``, ``OPENAI_API_KEY``);
- ``anthropic``: Claude Haiku under a JSON schema (``ANTHROPIC_API_KEY``),
  with probabilities 0 or 1; a generation takes about a second, so the channel
  waits up to 5 s for a decision rather than its default 2 s.

Nova's own answers come from a scripted model.

Run with:
    uv run python examples/speaking_judgments.py
    CLASSIFIER=jev TYPESAFE_API_KEY=... uv run python examples/speaking_judgments.py
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
    LLMClassifier,
    MockClassifier,
    OpenAIClassifier,
    RoomKit,
    SMSChannel,
    SpeakDecisionEvent,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.sms.mock import MockSMSProvider

logger = setup_logging("example.speaking_judgments")

LINES = [
    ("Sylvain", "Paul, do you have the September numbers?"),
    ("Paul", "Nova, can you sum up where we are?"),
    ("Sylvain", "Maybe Nova has the demo date somewhere."),
    ("Paul", "Nova, don't answer, just listen for now."),
    ("Sylvain", "Right, the budget, how much do we set it to?"),
    ("Paul", "Nova, what did we spend on it last year?"),
    ("Sylvain", "So we keep the same figure."),
    ("Paul", "Thanks Nova, feel free to chime in again."),
]

# What Jev answered on these lines, for the mock.
SCRIPT = [
    {"directness": 0.13, "request": 0.17, "language": "English"},
    {"directness": 2.98, "request": 0.95, "language": "English"},
    {"directness": 1.03, "unfinished": 0.31, "request": 0.28, "language": "English"},
    {
        "directness": 2.98,
        "deferred": 0.55,
        "hush": 0.94,
        "request": 0.45,
        "listen_request": 0.90,
        "language": "English",
    },
    {"request": 0.48, "asked_me": 0.30, "lift": 0.09, "language": "English"},
    {"directness": 2.93, "request": 0.88, "asked_me": 0.94, "lift": 0.12, "language": "English"},
    {"request": 0.09, "lift": 0.11, "language": "English"},
    {"directness": 2.91, "request": 0.63, "asked_me": 0.34, "lift": 0.85, "language": "English"},
]

LANGUAGES = {"English": "Answer in English only."}
"""One line per language Nova answers in, each written in its own language."""


def make_classifier(kind: str) -> Classifier:
    if kind == "jev":
        return JevClassifier(require_env("TYPESAFE_API_KEY")["TYPESAFE_API_KEY"])
    if kind == "openai":
        return OpenAIClassifier(require_env("OPENAI_API_KEY")["OPENAI_API_KEY"])
    if kind == "anthropic":
        key = require_env("ANTHROPIC_API_KEY")["ANTHROPIC_API_KEY"]
        config = AnthropicConfig(api_key=key, model="claude-haiku-5-5")
        return LLMClassifier(AnthropicAIProvider(config))
    return MockClassifier(SCRIPT)


async def main() -> None:
    kit = RoomKit()
    kind = os.environ.get("CLASSIFIER", "mock")
    classifier = make_classifier(kind)
    nova = AIChannel(
        "nova",
        provider=MockAIProvider(
            responses=[
                "We went through the September numbers: Paul sends them tomorrow.",
                "I can look up the demo date if you like.",
                "Last year the budget was 40,000 $.",
                "Happy to.",
            ]
        ),
        system_prompt="You are Nova, the team's assistant.",
        speak_policy=ClassifierSpeakPolicy(
            classifier, agent_name="Nova", agent_role="the team's assistant", languages=LANGUAGES
        ),
        speak_timeout=5.0 if kind == "anthropic" else 2.0,
    )
    kit.register_channel(nova)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    await kit.create_room(room_id="meeting")
    await kit.attach_channel("meeting", "nova", category=ChannelCategory.INTELLIGENCE)
    # Several people write on this one channel: a group binding (RFC §10.4).
    await kit.attach_channel("meeting", "sms", group=True)

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: object) -> None:
        decision = event.decision
        judged = " ".join(f"{k}={v:.2f}" for k, v in decision.judgments.items() if v >= 0.05)
        logger.info('"%s"', getattr(event.event.content, "body", ""))
        logger.info("  → %s (%s) %s %s", decision.mode, decision.reason, judged, decision.notes)

    for name, body in LINES:
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id=name.lower(),
                content=TextContent(body=body),
                metadata={"sender_name": name},
            )
        )
        await asyncio.sleep(0.1)

    answers = [e for e in await kit.store.list_events("meeting") if e.source.channel_id == "nova"]
    logger.info("Nova answered %d of %d lines:", len(answers), len(LINES))
    for answer in answers:
        logger.info("  Nova: %s", answer.content.body)
    await kit.close()
    await classifier.close()


if __name__ == "__main__":
    asyncio.run(main())
