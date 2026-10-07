"""A speak policy on judgments: whether Nova speaks, decided by a classifier.

Two people meet in a room with an agent, Nova. Her AI channel carries a
``ClassifierSpeakPolicy``: on each message, one classifier call answers narrow
questions about it (how directly is Nova brought in, did the speaker finish,
are they asking her to keep quiet, does a request to keep quiet still stand...)
and plain code composes the answers (RFC §6.4):

1. "Paul, tu as les chiffres de septembre ?" goes to Paul: silent.
2. "Nova, tu peux résumer où on en est ?" asks her: she speaks.
3. "Peut-être que Nova a la date de la démo quelque part." only wonders about
   her: she offers.
4. "Nova, ne réponds pas, écoute juste pour l'instant." asks for quiet: silent.
5. "Bon, le budget, on le passe à combien ?" goes to the room, not to her:
   silent, and the quiet asked for still stands.

Every decision reaches ``ON_SPEAK_DECISION`` with its reason and each judgment.
Told the languages Nova answers in, the policy also judges the speaker's, and
the turn's notes say it.

The classifier is chosen by ``CLASSIFIER``:

- ``mock`` (default): answers scripted from a run on Jev, no key needed;
- ``jev``: TypeSafe's Jev, calibrated (``pip install roomkit[typesafe]``,
  ``TYPESAFE_API_KEY``);
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
    ("Sylvain", "Paul, tu as les chiffres de septembre ?"),
    ("Paul", "Nova, tu peux résumer où on en est ?"),
    ("Sylvain", "Peut-être que Nova a la date de la démo quelque part."),
    ("Paul", "Nova, ne réponds pas, écoute juste pour l'instant."),
    ("Sylvain", "Bon, le budget, on le passe à combien ?"),
]

# What Jev answered on these lines, for the mock.
SCRIPT = [
    {"directness": 0.21, "request": 0.30, "language": "French"},
    {"directness": 2.95, "request": 0.95, "language": "French"},
    {"directness": 1.01, "unfinished": 0.32, "request": 0.21, "language": "French"},
    {
        "directness": 2.97,
        "deferred": 0.47,
        "hush": 0.95,
        "quiet_rule": 0.51,
        "request": 0.63,
        "language": "French",
    },
    {"quiet_rule": 0.95, "request": 0.47, "unfinished": 0.16, "language": "French"},
]

LANGUAGES = {
    "French": "Réponds en français uniquement.",
    "English": "Answer in English only.",
}


def make_classifier(kind: str) -> Classifier:
    if kind == "jev":
        return JevClassifier(require_env("TYPESAFE_API_KEY")["TYPESAFE_API_KEY"])
    if kind == "anthropic":
        key = require_env("ANTHROPIC_API_KEY")["ANTHROPIC_API_KEY"]
        config = AnthropicConfig(api_key=key, model="claude-haiku-4-5-20251001")
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
                "On a vu les chiffres de septembre : Paul les envoie demain.",
                "Je peux retrouver la date de la démo si vous voulez.",
            ]
        ),
        system_prompt="Tu es Nova, l'assistante de l'équipe.",
        speak_policy=ClassifierSpeakPolicy(
            classifier, agent_name="Nova", agent_role="the team's assistant", languages=LANGUAGES
        ),
        speak_timeout=5.0 if kind == "anthropic" else 2.0,
    )
    kit.register_channel(nova)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    await kit.create_room(room_id="meeting")
    await kit.attach_channel("meeting", "nova", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("meeting", "sms")

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: object) -> None:
        decision = event.decision
        judged = " ".join(f"{k}={v:.2f}" for k, v in decision.judgments.items() if v >= 0.05)
        logger.info("« %s »", getattr(event.event.content, "body", ""))
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
