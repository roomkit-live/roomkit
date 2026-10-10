"""Who takes a message that names nobody: a discussion's dispatch policy (RFC §19.7.5 rule 18).

An on-call engineer chats with four agents (investigator, sre, dev, comms)
and names nobody. The same eight messages run twice:

1. Without a policy, each message asks every agent, one turn each (rule 8):
   most turns end in "(silent)", the agent that should answer may come
   third, and a thanks wakes the whole team.
2. With a ``ClassifierDispatchPolicy``, one classifier call per message
   picks who takes it, from each agent's identity and the conversation: the
   error rate goes to the SRE, the support tickets to comms, a thanks to
   nobody. ``ON_DISPATCH_DECISION`` reports each decision and what it cost.

Each message prints who took a turn and who answered, then the totals.

The classifier is chosen by ``CLASSIFIER``:

- ``mock`` (default): the probabilities Jev gave on these messages, no key
  needed;
- ``jev``: TypeSafe's Jev, calibrated, ~150 ms a decision
  (``pip install roomkit[typesafe]``, ``TYPESAFE_API_KEY``);
- ``anthropic``: Claude Haiku under a JSON schema (``ANTHROPIC_API_KEY``),
  about a second and a half a decision.

The agents' own answers come from a scripted model: an agent answers a
message in its part and says "(silent)" otherwise, as a real model told it
may stay silent does.

Run with:
    uv run python examples/discussion_dispatch.py
    CLASSIFIER=jev TYPESAFE_API_KEY=... uv run python examples/discussion_dispatch.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import (
    Agent,
    Classifier,
    ClassifierDispatchPolicy,
    Discussion,
    DispatchDecisionEvent,
    DispatchPolicy,
    EventStatus,
    HookExecution,
    HookTrigger,
    InboundMessage,
    JevClassifier,
    LLMClassifier,
    MockClassifier,
    RoomKit,
    SpeakQueueChange,
    SpeakQueueEvent,
    TextContent,
    WebSocketChannel,
)
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

logger = setup_logging("example.discussion_dispatch")
# The room's own records of each silent turn, and each classifier request,
# would drown the comparison.
for noisy in ("roomkit", "httpx", "httpx2", "typesafe_sdk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

ROOM = "incident-room"
SILENT = "(silent)"

TEAM = {
    "investigator": ("Investigator", "reads checkout-api's logs and request traces"),
    "sre": (
        "SRE",
        "reads production metrics and the deploy history (with diffs), can roll a deploy back",
    ),
    "dev": (
        "Developer",
        "reads the feature-flag history and the source code, and is the only one who can "
        "change a feature flag",
    ),
    "comms": (
        "Comms",
        "reads the support tickets, owns the public status page, writes the postmortem",
    ),
}

# Each message, and what the agents whose part it is answer to it.
MESSAGES: dict[str, dict[str, str]] = {
    "Checkout alerts started firing a few minutes ago, what is going on?": {
        "investigator": "Logs show 500s from checkout-api since 14:02, all on the tax call.",
    },
    "What does the error rate look like right now?": {
        "sre": "18% of checkouts fail, up from 0.3% an hour ago.",
    },
    "Customers are writing to support, what are they saying?": {
        "comms": "Twelve tickets: payment fails at the last step, mostly Canadian addresses.",
    },
    "Is the tax_engine_v2 flag still ramped to 30%?": {
        "dev": "Yes, 30% since 13:55; it was 5% this morning.",
    },
    "Thanks, that helps.": {},
    "Was anything deployed in the last hour?": {
        "sre": "checkout-api 4.12 at 13:58: it bumps the tax client.",
    },
    "The status page still says all systems operational.": {
        "comms": "Updating it now: degraded checkout, investigating.",
    },
    "Ok, I am grabbing a coffee, back in five.": {},
}

# What Jev answered on these messages, by candidate in the team's order.
JEV_RUN = [
    {"agent_0": 0.85, "agent_1": 0.81, "agent_2": 0.24, "agent_3": 0.13},
    {"agent_0": 0.65, "agent_1": 0.91, "agent_2": 0.08, "agent_3": 0.06},
    {"agent_0": 0.05, "agent_1": 0.05, "agent_2": 0.05, "agent_3": 0.95},
    {"agent_0": 0.06, "agent_1": 0.10, "agent_2": 0.94, "agent_3": 0.05},
    {"agent_0": 0.05, "agent_1": 0.05, "agent_2": 0.05, "agent_3": 0.06},
    {"agent_0": 0.09, "agent_1": 0.94, "agent_2": 0.09, "agent_3": 0.04},
    {"agent_0": 0.16, "agent_1": 0.11, "agent_2": 0.10, "agent_3": 0.89},
    {"agent_0": 0.04, "agent_1": 0.04, "agent_2": 0.03, "agent_3": 0.05},
]


class Scripted(MockAIProvider):
    """Answers the message a turn answers when it is the agent's part, else
    stays silent."""

    def __init__(self, who: str) -> None:
        super().__init__()
        self.who = who

    async def generate(self, context):  # type: ignore[no-untyped-def]
        heard = next(
            (
                text
                for m in reversed(context.messages)
                if m.role == "user"
                for text in MESSAGES
                if text in str(m.content)
            ),
            None,
        )
        answer = MESSAGES.get(heard, {}).get(self.who, SILENT) if heard else SILENT
        return AIResponse(content=answer)


def make_classifier(kind: str) -> Classifier:
    if kind == "jev":
        return JevClassifier(require_env("TYPESAFE_API_KEY")["TYPESAFE_API_KEY"])
    if kind == "anthropic":
        key = require_env("ANTHROPIC_API_KEY")["ANTHROPIC_API_KEY"]
        return LLMClassifier(
            AnthropicAIProvider(AnthropicConfig(api_key=key, model="claude-haiku-5-5"))
        )
    return MockClassifier(JEV_RUN)


class Scene:
    """One run of the eight messages, and what each cost."""

    def __init__(self, title: str, dispatch: DispatchPolicy | None) -> None:
        self.title = title
        self.kit = RoomKit()
        self.dispatch = dispatch
        self.turns: list[str] = []
        self.decided: dict[str, DispatchDecisionEvent] = {}
        self.totals = {"turns": 0, "silent": 0, "answers": 0}

    async def run(self) -> dict[str, int]:
        kit = self.kit
        kit.register_channel(WebSocketChannel("ops"))
        agents = [
            Agent(handle, provider=Scripted(handle), name=name, description=does)
            for handle, (name, does) in TEAM.items()
        ]
        strategy = Discussion(agents, silent_token=SILENT, dispatch=self.dispatch)
        await kit.create_room(room_id=ROOM, orchestration=strategy)
        await kit.attach_channel(ROOM, "ops")
        self._follow()
        logger.info("")
        logger.info("== %s", self.title)
        for text in MESSAGES:
            await self._say(text)
        await kit.close()
        return self.totals

    def _follow(self) -> None:
        @self.kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
        async def on_queue(event: SpeakQueueEvent, ctx: object) -> None:
            if event.change == SpeakQueueChange.TURN_GIVEN:
                self.turns.extend(event.channel_ids)

        @self.kit.hook(HookTrigger.ON_DISPATCH_DECISION, execution=HookExecution.ASYNC)
        async def on_decision(event: DispatchDecisionEvent, ctx: object) -> None:
            self.decided[event.event.id] = event

    async def _say(self, text: str) -> None:
        before = len(self.turns)
        result = await self.kit.process_inbound(
            InboundMessage(channel_id="ops", sender_id="oncall", content=TextContent(body=text))
        )
        await self._settle(result.event.id)
        turns = self.turns[before:]
        rows = await self._rows_after(result.event.index)
        answered = [e.source.channel_id for e in rows if e.status != EventStatus.BLOCKED]
        silent = len(rows) - len(answered)
        self.totals["turns"] += len(turns)
        self.totals["silent"] += silent
        self.totals["answers"] += len(answered)
        logger.info('"%s"', text)
        logger.info(
            "   %d turn(s) %-34s answered: %s",
            len(turns),
            "(" + ", ".join(turns) + ")" if turns else "",
            ", ".join(answered) or "nobody",
        )
        decision = self.decided.get(result.event.id)
        if decision is not None:
            judged = ", ".join(f"{a} {p:.2f}" for a, p in decision.decision.judgments.items())
            logger.info("   decided in %d ms: %s", decision.duration_ms, judged)

    async def _settle(self, event_id: str) -> None:
        """Until the message is decided (with a policy) and the queue is idle."""
        still, last = 0, None
        async with asyncio.timeout(30):
            while self.dispatch is not None and event_id not in self.decided:
                await asyncio.sleep(0.02)
            while still < 10:
                await asyncio.sleep(0.02)
                queue = self.kit.speak_queue(ROOM)
                idle = queue is not None and queue.speaking is None and not queue.queue
                still = still + 1 if idle and queue == last else 0
                last = queue

    async def _rows_after(self, index: int):  # type: ignore[no-untyped-def]
        events = await self.kit.store.list_events(
            ROOM, limit=200, event_filter=EventFilter(include_blocked=True)
        )
        return [e for e in events if e.index > index and e.source.channel_id in TEAM]


async def main() -> None:
    kind = os.environ.get("CLASSIFIER", "mock")
    classifier = make_classifier(kind)
    before = await Scene("Before: every agent takes every unaddressed message", None).run()
    policy = ClassifierDispatchPolicy(classifier)
    after = await Scene(f"After: a dispatch policy on the {kind} classifier", policy).run()
    await classifier.close()

    logger.info("")
    logger.info("%-8s %6s %8s %8s", "", "turns", "silent", "answers")
    for name, totals in (("before", before), ("after", after)):
        logger.info("%-8s %6d %8d %8d", name, totals["turns"], totals["silent"], totals["answers"])


if __name__ == "__main__":
    asyncio.run(main())
