"""Narrow judgments composed in code: a classifier decides whether Nova speaks.

Two people and an agent, Nova, talk in one room. On each message the example asks
a classifier three questions together, in one call (RFC §6.8):

- ``addressee`` (a choice): Nova, a person in the room, or nobody in particular;
- ``asks`` (yes/no): whether the message asks for something;
- ``urgency`` (a score): can wait, within the day, right now.

Then plain code composes the answers: Nova speaks when the message is hers, or
when an urgent request goes to nobody; she offers on a question to nobody; she
listens otherwise. Each judgment is printed with the decision, so a wrong
decision shows which judgment, or which rule, made it.

The classifier is chosen by ``CLASSIFIER``:

- ``mock`` (default): answers scripted from a run on Jev, no key needed;
- ``jev``: TypeSafe's Jev, calibrated probabilities, ~150 ms a call
  (``pip install roomkit[typesafe]``, ``TYPESAFE_API_KEY``);
- ``openai``: OpenAI's Decisions API on ``gpt-6-luna``, probabilities, ~150-450 ms
  a call (``pip install roomkit[openai]``, ``OPENAI_API_KEY``);
- ``anthropic``: Claude Haiku answering under a JSON schema, every probability 0
  or 1 (``ANTHROPIC_API_KEY``).

Run with:
    uv run python examples/classifier_judgments.py
    CLASSIFIER=jev TYPESAFE_API_KEY=... uv run python examples/classifier_judgments.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import (
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    Classifier,
    JevClassifier,
    LLMClassifier,
    MockClassifier,
    OpenAIClassifier,
    ScoreQuestion,
    YesNoQuestion,
)
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

logger = setup_logging("example.classifier_judgments")

PEOPLE = ["Sylvain", "Paul"]

QUESTIONS = {
    "addressee": ChoiceQuestion(
        "Whom is the last message addressed to?",
        {
            "nova": "Nova, the assistant: she is named, or the request is plainly hers",
            "person": "a person in the room, named or plainly meant",
            "nobody": "nobody in particular: the room, or whoever knows",
        },
    ),
    "asks": YesNoQuestion(
        "Does the last message ask for something: an answer, an action, a piece of help?",
        no="a statement, a reaction, or thinking aloud",
    ),
    "urgency": ScoreQuestion(
        "How urgent is what the last message asks for?",
        ("nothing asked, or it can wait", "within the day", "right now"),
    ),
}

TURNS = [
    ("Sylvain", "Paul, do you have the September numbers?"),
    ("Paul", "Nova, can you sum up where we are?"),
    ("Sylvain", "Does anyone know when the demo is?"),
    ("Paul", "There's smoke in the server room, someone has to call for help, now!"),
    ("Sylvain", "Well, I think we're making good progress."),
]

# What Jev answered on these turns, for the mock.
SCRIPT = [
    {"addressee": "person", "asks": 0.97, "urgency": 0.6},
    {"addressee": "nova", "asks": 0.97, "urgency": 1.0},
    {"addressee": "nobody", "asks": 0.96, "urgency": 0.6},
    {"addressee": "nobody", "asks": 0.92, "urgency": 2.0},
    {"addressee": "nobody", "asks": 0.06, "urgency": 0.1},
]


def decide(answers: Answers) -> tuple[str, str]:
    """The decision and its reason, from the judgments: the policy, in code."""
    addressee = answers.choice("addressee")
    asks = answers.yes("asks") >= 0.5
    if addressee == "nova":
        return "speak", "hers"
    if addressee == "nobody" and asks and answers.score("urgency") >= 1.5:
        return "speak", "urgent, and nobody was asked"
    if addressee == "nobody" and asks:
        return "offer", "a question to nobody"
    return "silent", "not hers" if addressee == "person" else "nothing asked"


def classifier_for(turn: int) -> Classifier:
    kind = os.environ.get("CLASSIFIER", "mock")
    if kind == "jev":
        return JevClassifier(require_env("TYPESAFE_API_KEY")["TYPESAFE_API_KEY"])
    if kind == "openai":
        return OpenAIClassifier(require_env("OPENAI_API_KEY")["OPENAI_API_KEY"])
    if kind == "anthropic":
        key = require_env("ANTHROPIC_API_KEY")["ANTHROPIC_API_KEY"]
        config = AnthropicConfig(api_key=key, model="claude-haiku-5-5")
        return LLMClassifier(AnthropicAIProvider(config))
    return MockClassifier(SCRIPT[turn])


def describe(answers: Answers) -> str:
    addressee = answers["addressee"]
    assert isinstance(addressee, ChoiceAnswer)
    spread = " ".join(f"{k}={v:.2f}" for k, v in addressee.probabilities.items())
    return (
        f"addressee {spread} | asks {answers.yes('asks'):.2f} | "
        f"urgency {answers.score('urgency'):.1f}"
    )


async def main() -> None:
    history: list[dict[str, str]] = []
    for index, (speaker, text) in enumerate(TURNS):
        state = {
            "agent": "Nova",
            "people": PEOPLE,
            "recent": history[-6:],
            "last": {"from": speaker, "text": text},
        }
        classifier = classifier_for(index)
        try:
            started = time.perf_counter()
            answers = await classifier.classify(state, QUESTIONS)
            elapsed = (time.perf_counter() - started) * 1000
        finally:
            await classifier.close()
        mode, reason = decide(answers)
        logger.info('%s: "%s"', speaker, text)
        logger.info("  %s (%.0f ms)", describe(answers), elapsed)
        logger.info("  → %s: %s", mode, reason)
        history.append({"from": speaker, "text": text})


if __name__ == "__main__":
    asyncio.run(main())
