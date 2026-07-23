"""Buzz bot example — a RoomKit agent on a Buzz (Nostr) relay.

One ``buzzkit.BuzzClient`` is owned by the source (inbound: NIP-42 auth +
real-time subscribe) and reused by the provider for outbound sends — one Nostr
identity, both directions.

Setup:
    The agent's Nostr key must be a member of the target Buzz community. Create
    an invite in the Buzz app (Community > Members > Create invite link) and
    claim it once for the agent key (``buzzkit``'s ``claim_invite``), then copy
    the channel UUID.

Requires:
    pip install roomkit[buzz]

Run with:
    BUZZ_RELAY_URL=wss://your-community.communities.buzz.xyz \
    BUZZ_NSEC=nsec1... \
    BUZZ_CHANNEL_ID=<relay-channel-uuid> \
    uv run python examples/buzz_bot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import require_env, run_until_stopped, setup_logging

from roomkit import (
    BuzzChannel,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource
from roomkit.providers.buzz import BuzzConfig, BuzzProvider
from roomkit.sources.buzz import BuzzRelaySource

logger = setup_logging("buzz_bot")


async def main() -> None:
    env = require_env("BUZZ_RELAY_URL", "BUZZ_NSEC", "BUZZ_CHANNEL_ID")
    channel_id = "buzz-main"
    relay_channel_id = env["BUZZ_CHANNEL_ID"]

    # --- Source + Provider + Channel -----------------------------------------
    config = BuzzConfig(relay_url=env["BUZZ_RELAY_URL"], private_key=env["BUZZ_NSEC"])
    source = BuzzRelaySource(config, channel_id=channel_id, relay_channel_id=relay_channel_id)
    provider = BuzzProvider(source)

    kit = RoomKit()
    kit.register_channel(BuzzChannel(channel_id, provider=provider))
    await kit.create_room(room_id="buzz-echo")
    await kit.attach_channel(
        "buzz-echo", channel_id, metadata={"buzz_channel_id": relay_channel_id}
    )

    # --- Echo hook — reply in the same Buzz channel --------------------------
    # The source drops the agent's own events, so echoing never loops.
    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="echo_reply")
    async def echo_reply(event: RoomEvent, ctx: RoomContext) -> HookResult:
        buzz_channel = event.metadata.get("buzz_channel_id") or relay_channel_id
        content = event.content
        if isinstance(content, TextContent):
            logger.info("[%s] %s", event.source.participant_id, content.body)
            await provider.send(_reply(f"echo: {content.body}"), to=buzz_channel)
        return HookResult.allow()

    # --- Attach and run ------------------------------------------------------
    await kit.attach_source(channel_id, source, auto_restart=True)

    logger.info("Buzz source attached (channel=%s relay_channel=%s)", channel_id, relay_channel_id)
    logger.info("Waiting for messages... Press Ctrl+C to stop.")

    await run_until_stopped(kit)


def _reply(text: str) -> RoomEvent:
    """Build a minimal outbound RoomEvent carrying ``text``."""
    return RoomEvent(
        room_id="buzz-echo",
        source=EventSource(channel_id="buzz-main", channel_type=ChannelType.BUZZ),
        content=TextContent(body=text),
    )


if __name__ == "__main__":
    asyncio.run(main())
