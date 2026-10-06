"""An agent cancels a background task the person no longer wants.

The person asks for the weather in Québec, the agent delegates it to a weather
agent and keeps talking. Half a second later the person changes their mind:
"Ah non, plutôt Montréal." The agent cancels the Québec task with its
``cancel_task`` tool and delegates Montréal (RFC §23.4):

1. The Québec task ends ``cancelled``: ``ON_TASK_COMPLETED`` fires, the
   StatusBus posts its end, and its result never comes back.
2. The agent is not handed the cancellation back: the tool's answer told it,
   and a second word of it would have it say so twice.
3. Only Montréal's result is handed back, and said.

Without ``cancel_task`` the Québec task ran to its end and its result was said
anyway, after the person had asked for Montréal.

The models are scripted mocks, so the example runs without keys: the agent's
mock reads the task id it cancels from the delegation, as a real model reads it
from ``delegate_task``'s answer.

Run with:
    uv run python examples/cancel_background_task.py
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
    HookResult,
    HookTrigger,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.providers.ai import AIContext, AIResponse
from roomkit.providers.ai.base import AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.tasks import (
    CANCEL_TASK_TOOL,
    CancelTaskTool,
    DelegateHandler,
    TaskStatusTool,
    build_delegate_tool,
    setup_delegation,
)

logger = setup_logging("example.cancel_background_task")

PHONE = "+15145550100"
ROOM = "weather-chat"
# Long enough for the person to change their mind while the weather agent works.
_LOOKUP_SECONDS = 1.0


class _WeatherAgentAI(MockAIProvider):
    """A weather agent that takes a while, and answers for the city it was given."""

    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(_LOOKUP_SECONDS)
        asked = " ".join(str(m.content) for m in context.messages)
        if "Montréal" in asked:
            return AIResponse(content="Demain à Montréal : 13 °C, pluie faible.")
        return AIResponse(content="Demain à Québec : 8 °C, averses.")


class _AssistantAI(MockAIProvider):
    """The agent, scripted turn by turn. Its third turn cancels the task its
    first one delegated, by the id the delegation gave."""

    def __init__(self, delegated: list[str]) -> None:
        super().__init__()
        self._delegated = delegated

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        turn = len(self.calls)
        if turn == 1:  # "la météo de demain à Québec ?"
            return _calls(_delegate("q", "Météo de demain à Québec"))
        if turn == 2:
            return AIResponse(content="Je regarde la météo de Québec.")
        if turn == 3:  # "Ah non, plutôt Montréal."
            cancel = AIToolCall(
                id="c", name=CANCEL_TASK_TOOL, arguments={"task_id": self._delegated[0]}
            )
            return _calls(cancel, _delegate("m", "Météo de demain à Montréal"))
        if turn == 4:
            return AIResponse(content="D'accord : j'arrête Québec et je regarde Montréal.")
        return AIResponse(content="Demain à Montréal, 13 °C et un peu de pluie.")


def _delegate(call_id: str, task: str) -> AIToolCall:
    return AIToolCall(id=call_id, name="delegate_task", arguments={"agent": "meteo", "task": task})


def _calls(*calls: AIToolCall) -> AIResponse:
    return AIResponse(content="", finish_reason="tool_calls", tool_calls=list(calls))


async def main() -> None:
    kit = RoomKit()
    delegated: list[str] = []
    handed_back: list[str] = []
    ended: dict[str, str] = {}

    sms = SMSChannel("sms", provider=MockSMSProvider())
    assistant = AIChannel(
        "assistant",
        provider=_AssistantAI(delegated),
        system_prompt="Tu es l'assistant météo. Tu délègues les recherches.",
        # Follow the room's tasks, and stop one the person no longer wants.
        tools=[TaskStatusTool(kit), CancelTaskTool(kit)],
    )
    weather = Agent("meteo", provider=_WeatherAgentAI(), role="Météo")
    for channel in (sms, assistant, weather):
        kit.register_channel(channel)
    setup_delegation(
        assistant,
        DelegateHandler(kit),
        tool=build_delegate_tool([("meteo", "la météo d'une ville")]),
    )

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event, ctx):
        delegated.append(event.metadata["task_id"])
        print(f"  ⚙ delegated {event.metadata['task_id']}: {event.metadata['task_input']}")

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event, ctx):
        ended[event.metadata["task_id"]] = str(event.metadata["task_status"])
        print(f"  ⚙ ended {event.metadata['task_id']}: {event.metadata['task_status']}")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def on_hand_back(event, ctx):
        if "task_id" in event.metadata:
            handed_back.append(event.metadata["task_id"])
            print(f"  ⚙ handed back {event.metadata['task_id']}: {event.metadata['task_status']}")
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.MESSAGE})
    async def show(event, ctx):
        if event.source.channel_id == "assistant":
            print(f"  🗣 assistant: {event.content.body}")
        return HookResult.allow()

    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms", metadata={"phone_number": PHONE})
    await kit.attach_channel(ROOM, "assistant", category=ChannelCategory.INTELLIGENCE)

    async def say(text: str) -> None:
        print(f"[person] {text}")
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id=PHONE, content=TextContent(body=text)),
            room_id=ROOM,
        )

    await say("Tu peux regarder la météo de demain à Québec ?")
    await asyncio.sleep(0.5)
    await say("Ah non, plutôt Montréal.")
    await asyncio.sleep(_LOOKUP_SECONDS + 1.0)  # Montréal's result comes back

    quebec, montreal = delegated
    print(f"\n  Québec task: {ended.get(quebec)}; its result handed back: {quebec in handed_back}")
    print(f"  Montréal task: {ended.get(montreal)}; handed back: {montreal in handed_back}")
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
