"""Telegram Bot example — send and receive messages via a Telegram bot.

This example simulates an inbound webhook with hardcoded JSON. In production,
you would run a web server (FastAPI, Starlette, etc.) that receives real
webhooks from Telegram and feeds them into RoomKit.

Setup:
    1. Create a bot via @BotFather on Telegram — it gives you the bot token.
    2. Deploy a web server with a public HTTPS URL.
    3. Register your webhook with Telegram:
         curl https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://yourdomain.com/webhook/telegram
    4. In your webhook endpoint, parse the POST body and call:
         messages = parse_telegram_webhook(payload, channel_id="tg-main")
         for msg in messages:
             await kit.process_inbound(msg)

    Users interact with your bot by searching its username on Telegram,
    tapping Start, and sending messages. Telegram forwards each message
    to your webhook URL as a JSON POST.

Run this demo with:
    TELEGRAM_BOT_TOKEN=... uv run python examples/telegram_bot.py

Requires:
    pip install roomkit[telegram]
"""

from __future__ import annotations

import asyncio
import os

from roomkit import (
    RoomKit,
    TelegramBotProvider,
    TelegramConfig,
    WebSocketChannel,
    parse_telegram_webhook,
)
from roomkit.channels import TelegramChannel


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN to run this example.")
        return

    config = TelegramConfig(bot_token=token)
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

    # --- Simulate inbound webhook --------------------------------------------
    # In production you'd receive this JSON from your webhook endpoint.
    raw_webhook = {
        "update_id": 100000001,
        "message": {
            "message_id": 42,
            "from": {
                "id": 123456789,
                "is_bot": False,
                "first_name": "Alice",
            },
            "chat": {
                "id": 123456789,
                "first_name": "Alice",
                "type": "private",
            },
            "date": 1700000000,
            "text": "Hello from Telegram!",
        },
    }

    inbound_messages = parse_telegram_webhook(raw_webhook, channel_id="tg-main")
    for inbound in inbound_messages:
        print(f"Parsed inbound from {inbound.sender_id}: {inbound.content.body}")  # type: ignore[union-attr]
        result = await kit.process_inbound(inbound)
        print(f"  Processed: blocked={result.blocked}")

    # --- Show conversation history -------------------------------------------
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]

    await provider.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
