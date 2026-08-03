"""A first-class Buzz agent, powered by RoomKit's ``BuzzAgent`` runner.

What "first-class" means here is the lifecycle contract every Buzz agent is
expected to honor, however it was launched:

- **Presence is the status** — kind-20001 heartbeats while serving, an
  explicit ``offline`` on any deliberate stop (no stale green dot).
- **The owner's ``!shutdown`` works** — a kind-9 message mentioning the
  agent, authored by the owner proven by the NIP-OA auth tag, stops it
  gracefully instead of being answered by the AI. ``!cancel``/``!rotate``
  are consumed and surfaced, never fed to the model.
- **Opt-in inactivity bound** — with ``--exit-after-inactivity`` semantics:
  no traffic for that long, the agent reaps itself through the same
  graceful path.
- **Intentional exit is final and clean** — SIGTERM, ``!shutdown`` and the
  inactivity reaper all drain through ``kit.close()`` and the process exits
  ``0``, so a supervisor with restart-on-failure never resurrects a stop
  the owner meant.

The identity comes from the reserved environment triplet every Buzz launcher
uses, so this very script is deployable by a bash one-liner, a systemd unit,
or a container entrypoint.

Requires:
    pip install roomkit[buzz]  (buzzkit>=0.3.0)

Run with:
    BUZZ_RELAY_URL=wss://your-community.communities.buzz.xyz \
    BUZZ_PRIVATE_KEY=nsec1... \
    BUZZ_AUTH_TAG='["auth","<owner-pubkey>","","<sig>"]' \
    BUZZ_CHANNEL_ID=<relay-channel-uuid> \
    uv run python examples/buzz_agent.py

The auth tag is the owner's attestation over the agent's pubkey
(``buzzkit.compute_auth_tag(owner_nsec, agent_pubkey_hex)``); without it (or
an explicit ``owner_pubkey=``), owner commands stay inert — fail-closed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import os

from shared import setup_logging

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
from roomkit.models.event import ChannelData, EventSource
from roomkit.providers.buzz import BuzzAgent, BuzzConfig, BuzzProvider
from roomkit.sources.buzz import BuzzRelaySource

logger = setup_logging("buzz_agent")


async def main() -> int:
    channel_id = "buzz-main"
    relay_channel_id = os.environ["BUZZ_CHANNEL_ID"]

    # Identity from the reserved triplet (BUZZ_PRIVATE_KEY / BUZZ_RELAY_URL /
    # BUZZ_AUTH_TAG) — fail-closed: no key, no agent.
    config = BuzzConfig.from_env()
    source = BuzzRelaySource(config, channel_id, relay_channel_id=relay_channel_id)
    provider = BuzzProvider(source)

    # --- App wiring: rooms, channels, brains — all yours -------------------
    kit = RoomKit()
    kit.register_channel(BuzzChannel(channel_id, provider=provider))
    await kit.create_room(room_id="buzz-agent")
    await kit.attach_channel(
        "buzz-agent", channel_id, metadata={"buzz_channel_id": relay_channel_id}
    )

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="echo_reply")
    async def echo_reply(event: RoomEvent, ctx: RoomContext) -> HookResult:
        content = event.content
        if isinstance(content, TextContent):
            nostr_id = event.source.external_id
            thread_root = (event.channel_data.thread_id or nostr_id) if nostr_id else None
            await provider.send(
                RoomEvent(
                    room_id="buzz-agent",
                    source=EventSource(channel_id=channel_id, channel_type=ChannelType.BUZZ),
                    content=TextContent(body=f"echo: {content.body}"),
                    channel_data=ChannelData(thread_id=thread_root),
                ),
                to=event.metadata.get("buzz_channel_id") or relay_channel_id,
            )
        return HookResult.allow()

    # --- Lifecycle: the agent's, not yours ----------------------------------
    agent = BuzzAgent(
        kit,
        [source],
        exit_after_inactivity=float(os.environ.get("BUZZ_EXIT_AFTER_INACTIVITY", "0")) or None,
    )
    cause = await agent.run()  # blocks: owner !shutdown / SIGTERM / inactivity
    logger.info("Agent stopped: %s", cause)
    return 0  # every intentional path exits clean — supervisors must not restart it


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
