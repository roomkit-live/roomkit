"""An agent delegates with what the worker needs: its context and the conversation.

The agent's model calls ``delegate_task`` with a ``context`` and asks for the
room's last messages (RFC §23.4). The worker's child room starts empty, so this
is all it has: it reads the task, then both quoted as data.

- ``context`` needs nothing from the host.
- ``conversation`` needs ``DelegateHandler(max_conversation=N)``: with 0, the
  default, the tool has no such argument. The call asks for 8 here and is held
  to 2.
- The messages are those both the calling agent and the notified channel may
  read: the note the room shows the supervisor's phone alone stays out, though
  the handler notifies that phone.

The models are scripted mocks, so the example runs without keys.

Run with:
    uv run python examples/delegate_task_context.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    ChannelCategory,
    InboundMessage,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.providers.ai import AIResponse, AIToolCall, MockAIProvider
from roomkit.providers.sms import MockSMSProvider
from roomkit.tasks import DelegateHandler, setup_delegation

logger = setup_logging("delegate_task_context")


async def main() -> None:
    delegates = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="c1",
                        name="delegate_task",
                        arguments={
                            "agent": "fixer",
                            "task": "Fix the failing test.",
                            "context": {"branch": "fix-42", "test": "test_checkout"},
                            "conversation": 8,
                        },
                    )
                ],
            ),
            AIResponse(content="On it."),
            AIResponse(content="The test passes again."),
        ]
    )
    worker = MockAIProvider(responses=["Fixed."])

    kit = RoomKit()
    kit.register_channel(SMSChannel("phone", provider=MockSMSProvider()))
    kit.register_channel(SMSChannel("supervisor", provider=MockSMSProvider()))
    assistant = Agent("assistant", provider=delegates, tool_search=False)
    kit.register_channel(assistant)
    kit.register_channel(Agent("fixer", provider=worker, tool_search=False))

    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "phone", metadata={"phone_number": "+15551234567"})
    await kit.attach_channel("room-1", "supervisor", metadata={"phone_number": "+15557654321"})
    await kit.attach_channel("room-1", "assistant", category=ChannelCategory.INTELLIGENCE)

    setup_delegation(assistant, DelegateHandler(kit, notify="supervisor", max_conversation=2))

    await kit.send_event(
        "room-1",
        "supervisor",
        TextContent(body="Internal note."),
        visibility="supervisor",
        addressed_to=[],
    )
    await kit.send_event(
        "room-1", "phone", TextContent(body="It fails on main too."), addressed_to=[]
    )
    await kit.process_inbound(
        InboundMessage(
            channel_id="phone",
            sender_id="+15551234567",
            content=TextContent(body="The checkout test fails."),
        ),
        room_id="room-1",
    )

    while not worker.calls:
        await asyncio.sleep(0.01)
    logger.info("The worker's model reads:\n%s", worker.calls[0].messages[-1].content)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
