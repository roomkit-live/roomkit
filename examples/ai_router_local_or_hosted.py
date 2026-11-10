"""Each turn on a local or a hosted model, picked by a router (RFC §6.4).

An ``AIChannel`` built with ``router=`` asks its router which provider serves
each turn, at the turn's start: the turn's context is built for that model, and
every tool round of the turn stays on it. Once a generation's retries are spent
before it answered anything, the router is asked again and may hand the
generation to another model.

The router here, ``LocalOrHosted``, decides in plain code:

- once the local model's retries are spent, the hosted one takes the generation;
- when the machine is loaded (the one-minute load average per core past
  ``MAX_LOAD``), the hosted model, so the local one is not queued behind work;
- otherwise from how much reasoning the message needs: ``simple`` goes to the
  local model, ``complex`` to the hosted one.

That last answer comes from a classifier (RFC §6.8). By default it is
``RulesClassifier``: a few rules on the message's length and words, no model at
all, which is often enough. ``CLASSIFIER=llamacpp`` puts a decision model that
llama.cpp runs locally in its place; the router does not change.

Both chat models are mock providers, so the example runs offline. In an
application, ``local`` would be, say, a llama.cpp or vLLM server through an
OpenAI-compatible provider, and ``hosted`` a provider such as Kimi or GLM.

Run with:
    uv run python examples/ai_router_local_or_hosted.py
    MAX_LOAD=0 uv run python examples/ai_router_local_or_hosted.py   # every turn hosted
    CLASSIFIER=llamacpp uv run python examples/ai_router_local_or_hosted.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    AIChannel,
    AIRouter,
    Answers,
    ChannelCategory,
    ChoiceAnswer,
    ChoiceQuestion,
    Classifier,
    InboundMessage,
    LlamaCppClassifier,
    RoomKit,
    RouteRequest,
    TextContent,
    WebSocketChannel,
)
from roomkit.classifiers import Question, State, check_answers
from roomkit.models import RetryPolicy
from roomkit.providers.ai import AIContext, AIProvider, AIResponse, MockAIProvider, ProviderError
from roomkit.providers.llamacpp import LlamaCppConfig

logger = setup_logging("example.ai_router_local_or_hosted")

COMPLEXITY: dict[str, Question] = {
    "complexity": ChoiceQuestion(
        "How much reasoning does the last message need?",
        {
            "simple": "a greeting, a fact, a short answer the model knows",
            "complex": "a plan, a comparison, code, or steps to reason through",
        },
    )
}

# Words that ask for reasoning, in the messages this example sends.
_REASONING = re.compile(r"\b(why|compare|plan|design|explain|debug|step by step)\b", re.I)


class RulesClassifier(Classifier):
    """How much reasoning a message needs, from rules: long, code, or words that
    ask for reasoning is ``complex``. No model: a script that decides, often
    enough to route a turn. It answers the ``complexity`` question only."""

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        text = state if isinstance(state, str) else ""
        complex_ = len(text) > 280 or "```" in text or _REASONING.search(text) is not None
        choice = "complex" if complex_ else "simple"
        answer = ChoiceAnswer(choice, {"simple": float(not complex_), "complex": float(complex_)})
        return check_answers(questions, Answers({"complexity": answer}))


class LocalOrHosted(AIRouter):
    """Each turn on *local* or *hosted*, from the machine's load and the
    *classifier*'s answer; *hosted* once *local* spent its retries."""

    def __init__(
        self, local: AIProvider, hosted: AIProvider, classifier: Classifier, max_load: float
    ) -> None:
        self.local = local
        self.hosted = hosted
        self.classifier = classifier
        self.max_load = max_load

    async def route(self, request: RouteRequest) -> AIProvider | None:
        if request.reason == "retry":
            return self.hosted if request.previous is self.local else None
        load = os.getloadavg()[0] / (os.cpu_count() or 1)
        if load > self.max_load:
            logger.info("  router: load %.2f per core > %.2f, hosted", load, self.max_load)
            return self.hosted
        answers = await self.classifier.classify(request.text, COMPLEXITY)
        complexity = answers.choice("complexity")
        logger.info("  router: load %.2f per core, %s", load, complexity)
        return self.hosted if complexity == "complex" else self.local

    async def close(self) -> None:
        await self.local.close()
        await self.hosted.close()
        await self.classifier.close()


class Model(MockAIProvider):
    """A scripted chat model named *model*, standing in for a real one."""

    def __init__(self, model: str, responses: list[str]) -> None:
        super().__init__(responses)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model


class FlakyLocal(Model):
    """The local model, which refuses its *busy_at*-th request as overloaded."""

    def __init__(self, model: str, responses: list[str], busy_at: int) -> None:
        super().__init__(model, responses)
        self._busy_at = busy_at
        self._asked = 0

    async def generate(self, context: AIContext) -> AIResponse:
        self._asked += 1
        if self._asked == self._busy_at:
            raise ProviderError("local server overloaded", retryable=True, status_code=503)
        return await super().generate(context)


MESSAGES = [
    "Hi! What time do you open on Saturday?",
    "Compare the two plans for me and explain which one fits a team of five.",
    "Thanks, and where do I park?",
]


def make_classifier() -> Classifier:
    if os.environ.get("CLASSIFIER") == "llamacpp":
        model = os.environ.get("LLAMACPP_MODEL", "LiquidAI/d1-2B-GGUF:Q8_0")
        return LlamaCppClassifier(LlamaCppConfig(model=model))
    return RulesClassifier()


async def main() -> None:
    classifier = make_classifier()
    if isinstance(classifier, LlamaCppClassifier):
        logger.info("Starting the local decision model (the first run downloads it)...")
        await classifier.start()
    # The local model is busy on its second request: the router hands that
    # generation to the hosted one.
    local = FlakyLocal(
        "qwen-local", ["We open at 9 on Saturday.", "Parking is behind the building."], busy_at=2
    )
    hosted = Model("kimi-hosted", ["Plan B fits five people: ...", "Parking is behind."])
    router = LocalOrHosted(local, hosted, classifier, float(os.environ.get("MAX_LOAD", "4")))

    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(
        AIChannel("assistant", router=router, retry_policy=RetryPolicy(max_retries=0))
    )
    await kit.create_room(room_id="shop")
    await kit.attach_channel("shop", "ws")
    await kit.attach_channel("shop", "assistant", category=ChannelCategory.INTELLIGENCE)
    try:
        for text in MESSAGES:
            logger.info('user: "%s"', text)
            await kit.process_inbound(
                InboundMessage(channel_id="ws", sender_id="user", content=TextContent(body=text))
            )
            answer = (await kit.store.list_events("shop"))[-1]
            # The model that served the turn is on its answer (RFC §6.4).
            logger.info(
                '  %s: "%s"', answer.metadata.get("ai_model"), getattr(answer.content, "body", "")
            )
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
