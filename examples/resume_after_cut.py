"""An agent that knows it was cut off, and goes on when the person lets it.

Nova gives the weather. Sylvain says "Ok great, thanks" over her after a second,
as a voice barge-in would: the voice channel records the cut, naming the answer it
cut (RFC §12.3.13). On the next turn:

- the context Nova reads marks that answer as interrupted, rather than heard
  whole;
- her speak policy reads the cut (``SpeakTurn.cut``) and judges whether the
  turn leaves her free to go on: a thanks does, so she speaks with the reason
  ``resume after cut`` and a note asking her to go on from where she was cut.

A question over the cut ("Wait, and for Montreal?") wants the turn instead: the
turn decides as without a cut.

The voice channel's record is written here the way ``VoiceChannel`` writes it,
from a voice channel bound to the room (a record from anywhere else is not
trusted), so the example runs without audio. ``CLASSIFIER=jev`` (with ``TYPESAFE_API_KEY``)
judges on Jev; the default mock answers as Jev did.

Run with:
    uv run python examples/resume_after_cut.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import (
    AIGenerationEvent,
    ChannelCategory,
    Classifier,
    ClassifierSpeakPolicy,
    HookExecution,
    HookResult,
    HookTrigger,
    JevClassifier,
    MockClassifier,
    RoomKit,
    SMSChannel,
    SpeakDecisionEvent,
    TextContent,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType, Visibility
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.voice.backends.mock import MockVoiceBackend

logger = setup_logging("example.resume_after_cut")

FORECAST = "Tomorrow in Quebec City, 8 degrees and cloudy, with rain in the evening."

# Per policy call: the question, then the thanks spoken over the cut answer.
JUDGMENTS = [
    {"directness": 3.0, "request": 0.95},
    {"directness": 0.4, "request": 0.1, "resume": 0.84},
]


def make_classifier() -> Classifier:
    if os.environ.get("CLASSIFIER", "mock") == "jev":
        return JevClassifier(require_env("TYPESAFE_API_KEY")["TYPESAFE_API_KEY"])
    return MockClassifier(JUDGMENTS)


async def main() -> None:
    backend = MockVoiceBackend()
    kit = RoomKit(voice=backend)
    classifier = make_classifier()
    nova = AIChannel(
        "nova",
        provider=MockAIProvider(responses=[FORECAST, "So, as I was saying: rain in the evening."]),
        system_prompt="You are Nova, a voice assistant.",
        speak_policy=ClassifierSpeakPolicy(classifier, agent_name="Nova"),
    )
    kit.register_channel(nova)
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(VoiceChannel("voice", backend=backend))
    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "nova", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("call", "sms")
    await kit.attach_channel("call", "voice")
    await kit.ensure_participant("call", "sms", "sylvain", display_name="Sylvain")

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def on_decision(event: SpeakDecisionEvent, ctx: object) -> None:
        decision = event.decision
        logger.info("  → %s (%s) %s", decision.mode, decision.reason, list(decision.notes))

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def show_history(event: AIGenerationEvent, ctx: object) -> HookResult:
        said = [m.content for m in event.ai_context.messages if m.role == "assistant"]
        if said:
            logger.info("  Nova's last answer, as her context reads it: %s", said[-1])
        return HookResult.allow()

    async def say(text: str) -> None:
        logger.info('Sylvain: "%s"', text)
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id="sylvain",
                content=TextContent(body=text),
                metadata={"sender_name": "Sylvain"},
            )
        )
        await asyncio.sleep(0.1)

    await say("Nova, what's the weather tomorrow?")
    answer = next(
        e
        for e in reversed(await kit.store.list_events("call"))
        if e.source.channel_id == "nova" and e.type == EventType.MESSAGE
    )
    # What a VoiceChannel records when a barge-in cuts the answer (RFC §12.3.13).
    await kit.send_event(
        "call",
        "voice",
        TextContent(body=FORECAST),
        metadata={
            "interrupted": True,
            "played_ms": 1200,
            "answer_channel_id": "nova",
            "answer_responds_to": answer.responds_to,
        },
        visibility=Visibility.INTERNAL,
    )
    logger.info("  (Nova was cut off after 1.2 s)")
    await say("Ok great, thanks.")
    await kit.close()
    await classifier.close()


if __name__ == "__main__":
    asyncio.run(main())
