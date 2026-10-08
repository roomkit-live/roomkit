"""One channel, several correspondents: who lands in whose room (RFC §10.4 step 3).

On a number shared by many customers, the one ACTIVE room bound to it is the
first customer's conversation. A second customer routed into it is stored
there, answered with the first one's history as context, and answered at the
first one's address. These tests drive the whole inbound path.
"""

from __future__ import annotations

from typing import Any

from roomkit import HookResult, HookTrigger, RoomKit
from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

ALICE, BOB, CAROL = "+15550000001", "+15550000002", "+15550000003"


class _Shop:
    """A kit with one SMS number and an agent that joins every room it opens."""

    def __init__(self) -> None:
        self.sms = MockSMSProvider()
        self.ai = MockAIProvider(responses=["AI reply"])
        self.kit = RoomKit()
        self.kit.register_channel(SMSChannel("sms", provider=self.sms))
        self.kit.register_channel(AIChannel("ai", provider=self.ai))

        @self.kit.hook(HookTrigger.ON_ROOM_CREATED)
        async def agent_joins(event: Any, ctx: Any) -> None:
            await self.kit.attach_channel(ctx.room.id, "ai")

    async def open_room_for(self, number: str, *, member: bool) -> None:
        """The host opens a customer's room itself, as the examples do."""
        await self.kit.create_room(room_id=f"room-{number}")
        await self.kit.attach_channel(f"room-{number}", "sms", metadata={"phone_number": number})
        if member:
            await self.kit.add_member(f"room-{number}", "sms", number)

    async def text(self, sender: str, body: str, room_id: str | None = None) -> InboundResult:
        return await self.kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id=sender, content=TextContent(body=body)),
            room_id=room_id,
        )

    async def room_of(self, sender: str, body: str = "hi") -> str:
        result = await self.text(sender, body)
        assert result.event is not None
        return result.event.room_id

    def sent_to(self, number: str) -> list[str]:
        return [m["event"].content.body for m in self.sms.sent if m["to"] == number]

    def last_prompt(self) -> list[str]:
        return [_text(m) for m in self.ai.calls[-1].messages]


def _text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(getattr(part, "text", "") for part in content)


class TestTwoCustomersOnOneNumber:
    async def test_each_customer_gets_their_own_room(self) -> None:
        shop = _Shop()

        alice_room = await shop.room_of(ALICE)
        bob_room = await shop.room_of(BOB)

        assert bob_room != alice_room
        assert await shop.room_of(ALICE) == alice_room
        assert await shop.room_of(BOB) == bob_room

    async def test_the_agent_answering_bob_reads_nothing_of_alice(self) -> None:
        shop = _Shop()

        await shop.text(ALICE, "alice: my account number is 1234")
        await shop.text(BOB, "bob: hello")

        prompt = " ".join(shop.last_prompt())
        assert "bob: hello" in prompt
        assert "1234" not in prompt

    async def test_the_room_created_for_a_sender_names_them(self) -> None:
        shop = _Shop()

        room_id = await shop.room_of(ALICE)

        binding = await shop.kit.store.get_binding(room_id, "sms")
        assert binding is not None
        assert binding.participant_id == ALICE


class TestARoomTheHostOpened:
    async def test_bob_is_kept_out_of_alices_room_and_nothing_reaches_her(self) -> None:
        shop = _Shop()
        await shop.open_room_for(ALICE, member=True)
        assert await shop.room_of(ALICE, "alice: my account number is 1234") == f"room-{ALICE}"
        before = len(shop.sent_to(ALICE))

        bob_room = await shop.room_of(BOB, "bob: hello")

        assert bob_room != f"room-{ALICE}"
        assert len(shop.sent_to(ALICE)) == before
        assert "1234" not in " ".join(shop.last_prompt())

    async def test_a_member_on_the_channel_holds_the_room_before_writing(self) -> None:
        """The host said whose room it is: bob, writing first, is not let in."""
        shop = _Shop()
        await shop.open_room_for(ALICE, member=True)

        assert await shop.room_of(BOB) != f"room-{ALICE}"
        assert await shop.room_of(ALICE) == f"room-{ALICE}"

    async def test_a_room_opened_for_no_one_keeps_its_first_sender(self) -> None:
        shop = _Shop()
        await shop.open_room_for(ALICE, member=False)

        assert await shop.room_of(ALICE) == f"room-{ALICE}"
        bob_room = await shop.room_of(BOB)

        assert bob_room != f"room-{ALICE}"
        # Two rooms are bound to the number now; alice is found by her binding.
        assert await shop.room_of(ALICE) == f"room-{ALICE}"
        binding = await shop.kit.store.get_binding(f"room-{ALICE}", "sms")
        assert binding is not None
        assert binding.participant_id == ALICE

    async def test_a_room_that_already_heard_two_people_admits_no_newcomer(self) -> None:
        """A room mixed before its binding could name anyone: what it received
        is the record (here, messages the host routed by room id)."""
        shop = _Shop()
        await shop.open_room_for(ALICE, member=False)
        await shop.text(ALICE, "hi", room_id=f"room-{ALICE}")
        await shop.text(CAROL, "hi", room_id=f"room-{ALICE}")

        assert await shop.room_of(BOB) != f"room-{ALICE}"


class TestWhatTheBindingRetains:
    async def test_a_blocked_first_message_retains_no_one(self) -> None:
        shop = _Shop()
        await shop.open_room_for(ALICE, member=False)

        @shop.kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def refuse_bob(event: Any, ctx: Any) -> HookResult:
            if event.source.participant_id == BOB:
                return HookResult.block("spam")
            return HookResult.allow()

        result = await shop.text(BOB, "spam")

        assert result.blocked
        binding = await shop.kit.store.get_binding(f"room-{ALICE}", "sms")
        assert binding is not None
        assert binding.participant_id is None
        assert await shop.room_of(ALICE) == f"room-{ALICE}"

    async def test_a_message_routed_by_room_id_retains_no_one(self) -> None:
        """The caller chose the room: the binding is left as the host set it."""
        shop = _Shop()
        await shop.open_room_for(ALICE, member=False)

        await shop.text(ALICE, "hi", room_id=f"room-{ALICE}")

        binding = await shop.kit.store.get_binding(f"room-{ALICE}", "sms")
        assert binding is not None
        assert binding.participant_id is None


class TestAGroupChannel:
    async def test_every_sender_on_a_group_binding_shares_the_room(self) -> None:
        shop = _Shop()
        await shop.kit.create_room(room_id="team")
        await shop.kit.attach_channel("team", "sms", group=True)

        assert await shop.room_of(ALICE) == "team"
        assert await shop.room_of(BOB) == "team"
        assert await shop.room_of(CAROL) == "team"
        binding = await shop.kit.store.get_binding("team", "sms")
        assert binding is not None
        assert binding.participant_id is None


class TestTwoNumbersOnOneKit:
    async def test_a_sender_of_one_number_stays_out_of_its_room_on_another(self) -> None:
        """A binding speaks for its own channel (RFC §10.4 step 1): the bank's
        customer writing to the clinic's number lands in the clinic's room."""
        kit = RoomKit()
        for room_id in ("bank", "clinic"):
            kit.register_channel(SMSChannel(f"sms-{room_id}", provider=MockSMSProvider()))
            await kit.create_room(room_id=room_id)
            await kit.attach_channel(room_id, f"sms-{room_id}")

        async def room_of(channel_id: str) -> str:
            result = await kit.process_inbound(
                InboundMessage(
                    channel_id=channel_id, sender_id=ALICE, content=TextContent(body="hi")
                )
            )
            assert result.event is not None
            return result.event.room_id

        assert await room_of("sms-bank") == "bank"
        assert await room_of("sms-clinic") == "clinic"
        assert await room_of("sms-bank") == "bank"
        assert await room_of("sms-clinic") == "clinic"
