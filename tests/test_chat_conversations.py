"""On a chat channel the conversation is the chat, not the sender (RFC §10.4).

Telegram, Teams, Discord and Buzz deliver to a chat. A user writing to the bot
privately and in a group holds two conversations, and a group's members hold
one together: a room found by its sender mixed them, and answered a private
message in the group or a group message in private.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from roomkit import HookTrigger, RoomKit
from roomkit.channels import TelegramChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.telegram.mock import MockTelegramProvider
from roomkit.providers.telegram.webhook import parse_telegram_webhook

ALICE, BOB = 1001, 1002
GROUP = -100500


async def _eventually(predicate: Callable[[], Any], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


class _Bot:
    """A Telegram bot whose agent joins every room the kit opens."""

    def __init__(self) -> None:
        self.telegram = MockTelegramProvider()
        self.kit = RoomKit()
        self.kit.register_channel(TelegramChannel("tg", provider=self.telegram))
        self.kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["ok"])))
        self._update = 0

        @self.kit.hook(HookTrigger.ON_ROOM_CREATED)
        async def agent_joins(event: Any, ctx: Any) -> None:
            await self.kit.attach_channel(ctx.room.id, "ai")

    async def write(self, user: int, text: str, chat: int | None = None) -> str:
        """*user* writes in *chat* (their private chat by default); returns the room."""
        chat = user if chat is None else chat
        self._update += 1
        update = {
            "update_id": self._update,
            "message": {
                "message_id": self._update,
                "from": {"id": user, "is_bot": False, "first_name": str(user)},
                "chat": {"id": chat, "type": "private" if chat == user else "supergroup"},
                "date": 0,
                "text": text,
            },
        }
        sent = len(self.telegram.sent)
        [message] = parse_telegram_webhook(update, channel_id="tg")
        result = await self.kit.process_inbound(message)
        assert result.event is not None
        await _eventually(lambda: len(self.telegram.sent) > sent)
        return result.event.room_id

    def last_reply_chat(self) -> str:
        return self.telegram.sent[-1]["to"]


async def test_a_group_message_is_answered_in_the_group_after_a_private_chat() -> None:
    bot = _Bot()

    private = await bot.write(ALICE, "hi")
    assert bot.last_reply_chat() == str(ALICE)
    group = await bot.write(ALICE, "hi everyone", chat=GROUP)

    assert group != private
    assert bot.last_reply_chat() == str(GROUP)
    await bot.kit.close()


async def test_a_private_message_is_answered_in_private_after_a_group() -> None:
    bot = _Bot()

    group = await bot.write(BOB, "hello", chat=GROUP)
    assert bot.last_reply_chat() == str(GROUP)
    private = await bot.write(BOB, "something private")

    assert private != group
    assert bot.last_reply_chat() == str(BOB)
    await bot.kit.close()


async def test_a_groups_members_share_its_room_and_their_private_chats_stay_apart() -> None:
    bot = _Bot()

    alice_in_group = await bot.write(ALICE, "hi", chat=GROUP)
    bob_in_group = await bot.write(BOB, "hello", chat=GROUP)
    alice_private = await bot.write(ALICE, "a private question")
    bob_private = await bot.write(BOB, "another one")

    assert alice_in_group == bob_in_group
    assert len({alice_in_group, alice_private, bob_private}) == 3
    events = await bot.kit.store.list_events(alice_in_group, limit=50)
    authors = {e.source.participant_id for e in events if e.source.channel_id == "tg"}
    assert authors == {str(ALICE), str(BOB)}
    await bot.kit.close()
