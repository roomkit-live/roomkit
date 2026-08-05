"""Telegram Bot example — send and receive messages via a Telegram bot.

This example simulates an inbound webhook with hardcoded JSON. In production,
you would run a web server (FastAPI, Starlette, etc.) that receives real
webhooks from Telegram and feeds them into RoomKit.

It covers two inbound shapes: a text message and a voice note. A media update
carries no body of its own — what it carries is a ``file_id``, which only the
side holding the bot token can turn into bytes. That is the provider:
``get_file()`` resolves the id to a path, ``download_file()`` fetches it. What
you then do with those bytes (transcribe them, store them) is your
application's call — RoomKit does not pick an ASR engine for you.

It also shows ``parse_telegram_message()``, the layer below the webhook parser,
which reads a message without deciding who sent it; ``parse_telegram_update()``,
which tells apart the three forms an Update takes; ``mentions_bot()``, which
says whether a group message was addressed to the bot; and the Bot API surface
the provider carries — ``get_me()`` and the webhook lifecycle — so an
application never writes a second HTTP client for a token the provider holds.

Setup:
    1. Create a bot via @BotFather on Telegram — it gives you the bot token.
    2. Deploy a web server with a public HTTPS URL.
    3. Register your webhook with Telegram:
         await provider.set_webhook("https://yourdomain.com/webhook/telegram",
                                    secret=..., allowed_updates=[...])
    4. In your webhook endpoint, parse the POST body and call:
         messages = parse_telegram_webhook(payload, channel_id="tg-main")
         for msg in messages:
             await kit.process_inbound(msg)

    Users interact with your bot by searching its username on Telegram,
    tapping Start, and sending messages. Telegram forwards each message
    to your webhook URL as a JSON POST.

Run this demo with:
    TELEGRAM_BOT_TOKEN=... uv run python examples/telegram_bot.py

    The simulated file_id is made up, so the demo only prints what the parser
    extracted. To exercise the real download path, send a voice note to your
    bot, read the file_id off the update, and set TELEGRAM_FILE_ID to it.

    Set TELEGRAM_WEBHOOK_URL to also register a webhook and tear it down again.
    That one writes to your real bot; the rest of the demo only reads.

Requires:
    pip install roomkit[telegram]
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import require_env

from roomkit import RoomKit, WebSocketChannel
from roomkit.channels import TelegramChannel
from roomkit.providers.telegram import (
    TelegramBotProvider,
    TelegramConfig,
    mentions_bot,
    parse_telegram_message,
    parse_telegram_update,
    parse_telegram_webhook,
)


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    env = require_env("TELEGRAM_BOT_TOKEN")

    config = TelegramConfig(bot_token=env["TELEGRAM_BOT_TOKEN"])
    provider = TelegramBotProvider(config)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()
    telegram = TelegramChannel("tg-main", provider=provider)
    ws = WebSocketChannel("ws-agent")
    kit.register_channel(telegram)
    kit.register_channel(ws)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel(
        "demo-room",
        "tg-main",
        metadata={"telegram_chat_id": "CHAT_ID"},
    )
    await kit.attach_channel("demo-room", "ws-agent")

    # --- Simulate inbound webhooks -------------------------------------------
    # In production you'd receive this JSON from your webhook endpoint.
    sender = {"id": 123456789, "is_bot": False, "first_name": "Alice"}
    chat = {"id": 123456789, "first_name": "Alice", "type": "private"}

    raw_webhooks = [
        {
            "update_id": 100000001,
            "message": {
                "message_id": 42,
                "from": sender,
                "chat": chat,
                "date": 1700000000,
                "text": "Hello from Telegram!",
            },
        },
        # A voice note: no text, no caption. The body is empty on purpose —
        # the file_id is the message.
        {
            "update_id": 100000002,
            "message": {
                "message_id": 43,
                "from": sender,
                "chat": chat,
                "date": 1700000001,
                "voice": {
                    "duration": 4,
                    "mime_type": "audio/ogg",
                    "file_id": "AwACAgIAAxkBAAIC-made-up-for-the-demo",
                    "file_unique_id": "AgADbQ4AAlOZAUo",
                    "file_size": 12834,
                },
            },
        },
    ]

    for raw_webhook in raw_webhooks:
        for inbound in parse_telegram_webhook(raw_webhook, channel_id="tg-main"):
            print(f"Parsed inbound from {inbound.sender_id}: {inbound.content.body!r}")  # type: ignore[union-attr]
            if "file_id" in inbound.metadata:
                print(
                    f"  {inbound.metadata['media_type']} file_id="
                    f"{inbound.metadata['file_id']} "
                    f"duration={inbound.metadata.get('duration')}s "
                    f"size={inbound.metadata.get('file_size')}B"
                )
            result = await kit.process_inbound(inbound)
            print(f"  Processed: blocked={result.blocked}")

    # --- Parsing without attribution -----------------------------------------
    # parse_telegram_webhook decides the sender is message.from.id. That rule is
    # not universal: under a one-bot-per-user model a DM belongs to the bot's
    # owner, not to the account that typed it. parse_telegram_message is the
    # layer below — it says what the message contains and leaves who sent it
    # to you, so reading a file_id never costs you your identity model.
    parts = parse_telegram_message(raw_webhooks[1]["message"])
    if parts:
        print(
            f"\nLow-level parse: media_type={parts.metadata['media_type']} "
            f"telegram sender={parts.sender_id} (yours to override)"
        )

    # --- Resolve a real file_id to bytes -------------------------------------
    # Two calls: getFile turns the id into a path (valid at least an hour),
    # then the path is downloaded. Telegram caps Bot API downloads at 20 MB and
    # refuses larger files at the getFile step, which surfaces here as None.
    file_id = os.environ.get("TELEGRAM_FILE_ID")
    if file_id:
        file_path = await provider.get_file(file_id)
        print(f"\ngetFile({file_id[:12]}…) -> {file_path}")
        if file_path:
            audio = await provider.download_file(file_path)
            print(f"  downloaded {len(audio or b'')} bytes")
            # Hand `audio` to whatever ASR your application uses — RoomKit
            # stops at the bytes.
    else:
        print("\nSet TELEGRAM_FILE_ID to a real file_id to exercise the download path.")

    # --- Who is this bot? ----------------------------------------------------
    # getMe is the call that tells a good token from a typo. Every Bot API call
    # answers the same way — a ProviderResult — so a refusal (telegram_401) and
    # an unreachable Telegram (timeout) are told apart without parsing strings.
    me = await provider.get_me()
    if me.success:
        bot = me.metadata["result"]
        bot_username, bot_id = bot.get("username"), bot.get("id")
        print(f"\ngetMe -> @{bot_username} (id={bot_id})")
    else:
        bot_username, bot_id = None, None
        print(f"\ngetMe failed: {me.error} — {me.metadata.get('description', '')}")

    # --- Webhook lifecycle ---------------------------------------------------
    # The secret is echoed back on every request in the
    # X-Telegram-Bot-Api-Secret-Token header, which provider.verify_signature
    # checks. allowed_updates is not optional in practice: Telegram's own
    # default leaves out callback_query, so a bot with buttons must ask for it.
    webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL")
    if webhook_url:
        registered = await provider.set_webhook(
            webhook_url,
            secret=secrets.token_hex(32),
            allowed_updates=["message", "edited_message", "callback_query"],
        )
        print(f"\nsetWebhook({webhook_url}) -> success={registered.success}")
        if not registered.success:
            # Telegram's own words say which rule was broken — HTTPS required,
            # host unreachable — where the error code alone would not.
            print(f"  {registered.error}: {registered.metadata.get('description', '')}")
        await provider.delete_webhook()
        print("  torn down again")
    else:
        print("\nSet TELEGRAM_WEBHOOK_URL to register a webhook and tear it down.")

    # --- The other two update forms ------------------------------------------
    # parse_telegram_update says which form arrived. A button press carries no
    # message of its own — what it carries is callback_data, posted by whoever
    # pressed the button, so treat what it names as a claim to check.
    press = parse_telegram_update(
        {
            "update_id": 100000003,
            "callback_query": {
                "id": "4382bfdwdsb323b2d9",
                "from": sender,
                "data": "approve:7f3a",
                "message": {"message_id": 44, "chat": chat, "text": "Approve the draft?"},
            },
        }
    )
    if press and press.callback:
        cb = press.callback
        print(f"\nButton press: data={cb.data!r} by {cb.sender_id} on message {cb.message_id}")
        print(f"  answer it with: await provider.answer_callback_query({cb.id!r}, 'Approved.')")

    # --- Was that group message for us? --------------------------------------
    # mentions_bot gives the fact. Whether a group answers only when addressed
    # is your policy, not the kit's. The emoji is the point: entity offsets
    # count UTF-16 code units, so a code-point slice would read the wrong
    # substring and miss the mention entirely.
    group_message = {
        "message_id": 45,
        "from": sender,
        "chat": {"id": -1001234567890, "type": "supergroup", "title": "Team"},
        "text": f"🎉 @{bot_username or 'your_bot'} can you summarise this?",
        "entities": [
            {"type": "mention", "offset": 3, "length": len(bot_username or "your_bot") + 1}
        ],
    }
    addressed = mentions_bot(group_message, bot_username=bot_username, bot_id=bot_id)
    print(f"\nGroup message addressed to the bot: {addressed}")

    # --- Show conversation history -------------------------------------------
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]

    await provider.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
