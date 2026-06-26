"""Discord bot example — bidirectional messaging via the gateway.

A single ``discord.Client`` is owned by the gateway source and reused by the
provider for outbound sends (one connection, both directions).

Setup (Discord Developer Portal — https://discord.com/developers/applications):
    1. Create an application, then add a Bot to it and copy its token.
    2. Enable the **Message Content Intent** under Bot > Privileged Gateway
       Intents (otherwise inbound ``message.content`` arrives empty).
    3. Invite the bot to your server with the OAuth2 URL generator
       (scopes: ``bot``; permissions: View Channels, Send Messages,
       Read Message History, Add Reactions).

Requires:
    pip install roomkit[discord]

Run with:
    DISCORD_BOT_TOKEN=... uv run python examples/discord_bot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import require_env, run_until_stopped, setup_logging

from roomkit import (
    DiscordChannel,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, MediaContent
from roomkit.providers.discord import DiscordBotProvider, DiscordConfig
from roomkit.sources.discord import DiscordGatewaySource

logger = setup_logging("discord_bot")


async def main() -> None:
    env = require_env("DISCORD_BOT_TOKEN")
    channel_id = "discord-main"

    # --- Reaction lifecycle handler (not part of the message pipeline) -------
    async def on_discord_event(reaction: dict[str, str]) -> None:
        logger.info(
            "reaction %s %s on message %s",
            reaction["action"],
            reaction["emoji"],
            reaction["message_id"],
        )

    # --- Source + Provider + Channel -----------------------------------------
    config = DiscordConfig(bot_token=env["DISCORD_BOT_TOKEN"])
    source = DiscordGatewaySource(config, channel_id=channel_id, on_event=on_discord_event)
    provider = DiscordBotProvider(source)

    kit = RoomKit()
    kit.register_channel(DiscordChannel(channel_id, provider=provider))

    # --- Echo hook — reply in the same Discord channel -----------------------
    # The parser drops the bot's own messages, so echoing never loops.
    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="echo_reply")
    async def echo_reply(event: RoomEvent, ctx: RoomContext) -> HookResult:
        name = event.metadata.get("author_name") or event.source.participant_id or "unknown"
        discord_channel = event.metadata.get("channel_id")
        c = event.content
        if isinstance(c, TextContent):
            logger.info("[%s] %s", name, c.body)
            if discord_channel:
                await provider.send(_reply(f"echo: {c.body}"), to=discord_channel)
        elif isinstance(c, MediaContent):
            logger.info("[%s] %s %s", name, c.mime_type, c.url)
        else:
            logger.info("[%s] %s", name, type(c).__name__)
        return HookResult.allow()

    # --- Attach and run ------------------------------------------------------
    await kit.attach_source(channel_id, source, auto_restart=True)

    logger.info("Discord gateway source attached (channel=%s)", channel_id)
    logger.info("Waiting for messages... Press Ctrl+C to stop.")

    await run_until_stopped(kit)


def _reply(text: str) -> RoomEvent:
    """Build a minimal outbound RoomEvent carrying ``text``."""
    return RoomEvent(
        room_id="discord-echo",
        source=EventSource(channel_id="discord-main", channel_type=ChannelType.DISCORD),
        content=TextContent(body=text),
    )


if __name__ == "__main__":
    asyncio.run(main())
