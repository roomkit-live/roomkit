"""One SMS number, many customers: each conversation stays its own.

A business with a single number receives texts from many people. The default
router (RFC §10.4) gives each new correspondent a room of their own and keeps
them there: the room created for a sender's first message has a binding that
names them, the next message is found through it, and a stranger is never let
into someone else's conversation. A group chat on a channel dedicated to its
room is the one case where several senders share the room, and it is declared
on the binding with ``group=True``.

Shows:
- two customers on one number each get a room, and the agent answering one
  reads nothing of the other
- a room the host opened for a customer keeps a stranger out before the
  customer has written
- ``attach_channel(..., group=True)``: a group channel shared by everyone

Runs with mock providers, no key needed:
    uv run python examples/shared_sms_number.py
"""

from __future__ import annotations

import asyncio

from shared import setup_logging

from roomkit import HookTrigger, InboundMessage, RoomKit, TextContent
from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

logger = setup_logging("shared_sms_number")

ALICE, BOB, CAROL = "+15550000001", "+15550000002", "+15550000003"


def build_kit() -> tuple[RoomKit, MockSMSProvider, MockAIProvider]:
    """A kit with one SMS number and an agent that joins every new room."""
    sms = MockSMSProvider()
    ai = MockAIProvider(responses=["Thanks, we are on it."])
    kit = RoomKit()
    kit.register_channel(SMSChannel("sms", provider=sms))
    kit.register_channel(AIChannel("assistant", provider=ai))

    @kit.hook(HookTrigger.ON_ROOM_CREATED)
    async def assistant_joins(event, ctx) -> None:
        await kit.attach_channel(ctx.room.id, "assistant")

    return kit, sms, ai


async def text(kit: RoomKit, sender: str, body: str) -> str:
    """Deliver one inbound SMS without naming a room; return where it landed."""
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id=sender, content=TextContent(body=body))
    )
    assert result.event is not None
    return result.event.room_id


def prompt_of(ai: MockAIProvider) -> str:
    """The conversation the agent read for its last answer."""
    parts = []
    for message in ai.calls[-1].messages:
        content = message.content
        parts.append(
            content
            if isinstance(content, str)
            else " ".join(getattr(part, "text", "") for part in content)
        )
    return " | ".join(parts)


async def two_customers() -> None:
    logger.info("--- Two customers on one number ---")
    kit, _sms, ai = build_kit()

    alice_room = await text(kit, ALICE, "My account number is 1234.")
    bob_room = await text(kit, BOB, "Hello, what are your opening hours?")
    logger.info("alice -> %s, bob -> %s", alice_room[:8], bob_room[:8])
    logger.info("the agent answering bob read: %s", prompt_of(ai))

    again = await text(kit, ALICE, "Any news?")
    logger.info("alice again -> %s (her own room: %s)", again[:8], again == alice_room)
    binding = await kit.store.get_binding(alice_room, "sms")
    logger.info("alice's binding names: %s", binding.participant_id if binding else None)
    await kit.close()


async def a_room_the_host_opened() -> None:
    logger.info("--- A room the host opened for alice ---")
    kit, sms, _ai = build_kit()
    await kit.create_room(room_id="alice-appointment")
    await kit.attach_channel("alice-appointment", "sms", metadata={"phone_number": ALICE})
    await kit.add_member("alice-appointment", "sms", ALICE)

    bob_room = await text(kit, BOB, "I would like an appointment.")
    sent_to_alice = [m for m in sms.sent if m["to"] == ALICE]
    logger.info("bob, writing first -> %s; sent to alice: %d", bob_room[:8], len(sent_to_alice))
    logger.info("alice -> %s", await text(kit, ALICE, "Confirming tomorrow."))
    await kit.close()


async def a_group_channel() -> None:
    logger.info("--- A group channel ---")
    kit, _sms, _ai = build_kit()
    await kit.create_room(room_id="team")
    await kit.attach_channel("team", "sms", group=True)

    for sender in (ALICE, BOB, CAROL):
        logger.info("%s -> %s", sender, await text(kit, sender, "Hi team"))
    await kit.close()


async def main() -> None:
    await two_customers()
    await a_room_the_host_opened()
    await a_group_channel()


if __name__ == "__main__":
    asyncio.run(main())
