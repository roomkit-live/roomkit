"""Claude Code takes part in a discussion, over ACP (RFC §6.4, §19.7.5).

@dev is Claude Code, an external agent, in the same discussion as @sre, an
AI channel: it takes its turns like any agent, reads the discussion's notes,
and works in a read-only workspace holding a checkout service's source and
its support tickets. The run:

1. The on-call asks @dev why Canadian checkouts fail: Claude Code reads the
   source and answers in its turn.
2. The on-call starts a task for @dev with ``kit.start_task`` (the tickets,
   a longer read): it runs in a session of its own, in a child room, while
   @sre answers a quick question; the findings are posted as @dev's message
   when they are done (``hand_back="post"``).

``--console`` opens the discussion console instead, to lead the room
yourself (``/task @dev ...`` starts a task, ``/tasks`` lists them).

Requires:
    pip install "roomkit[acp,console]"
    Node.js 22+ with ``npx``, and Claude Code signed in once:
    npx -y @agentclientprotocol/claude-agent-acp@0.61.0 --cli auth login --claudeai

Run with:
    uv run python examples/discussion_acp.py
    uv run python examples/discussion_acp.py --console
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    ACPChannel,
    Agent,
    Discussion,
    EventStatus,
    EventType,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.console import DiscussionConsole
from roomkit.providers.ai import AIResponse, MockAIProvider
from roomkit.tools import PolicyExternalToolHandler, ToolPolicy

setup_logging("discussion_acp")
logging.getLogger("roomkit").setLevel(logging.WARNING)

CLAUDE_AGENT_ACP = "@agentclientprotocol/claude-agent-acp@0.61.0"
ROOM = "incident"

SOURCE = '''\
def jurisdiction(address):
    """Tax engine v2: the jurisdiction of a Canadian address."""
    if address.country == "CA" and not address.province:
        raise InvalidJurisdiction("CA address without a province")  # HTTP 422
    return Jurisdiction(address.country, address.province)
'''
FORM = """\
REQUIRED = {"CA": ["street", "city", "postal_code"]}  # province dropped on 2026-10-10 13:30
"""
TICKETS = """\
T-5521 14:07 Montréal QC: checkout fails, I left the province empty, it says optional
T-5524 14:09 Toronto ON: error at payment, no province field filled
T-5530 14:11 Vancouver BC: cannot pay, 'something went wrong'
T-5536 14:15 Seattle US: wrong delivery date shown
"""


READ_ONLY = ToolPolicy(allow=["Read*", "Glob*", "Grep*", "LS*", "read*", "glob*", "grep*"])
"""What Claude Code may do when it asks RoomKit: read its workspace, nothing else."""


def workspace() -> Path:
    root = Path(tempfile.mkdtemp(prefix="roomkit-discussion-acp-"))
    (root / "tax").mkdir()
    (root / "tax" / "engine_v2.py").write_text(SOURCE)
    (root / "web").mkdir()
    (root / "web" / "address_form.py").write_text(FORM)
    (root / "support").mkdir()
    (root / "support" / "tickets.txt").write_text(TICKETS)
    return root


async def build(*, console: bool = False) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("oncall"))
    dev = ACPChannel(
        "dev",
        command=["npx", "--no-install", CLAUDE_AGENT_ACP],
        cwd=workspace(),
        inherit_env=["HOME", "PATH", "USER", "XDG_CONFIG_HOME"],
        # RoomKit's gate: the policy, then BEFORE_TOOL_USE, each call reported.
        external_tool_handler=PolicyExternalToolHandler(policy=READ_ONLY),
        publish_tool_call_text=console,
        # Claude Code's own mode decides what it asks before acting: under a
        # user's "auto" mode it acts without asking, and RoomKit's gate never
        # hears of it. "default" asks before any action that is not a read. Set
        # on every session the channel opens: the room's, and a task's child room's.
        session_config={"mode": "default"},
        name="Developer",
        description="reads the checkout service's source and its support tickets",
        instructions=(
            "You are @dev, a developer in an incident chat with the on-call (@oncall) and "
            "@sre. Answer briefly, from the files in your workspace; change nothing."
        ),
    )
    sre = Agent(
        "sre",
        provider=MockAIProvider(ai_responses=[AIResponse(content="Checkout errors: 6%, flat.")]),
        name="SRE",
        description="reads production metrics",
    )
    await kit.create_room(room_id=ROOM, orchestration=Discussion([sre, dev]))
    await kit.attach_channel(ROOM, "oncall")
    return kit


async def say(kit: RoomKit, text: str) -> None:
    print(f"\n@oncall: {text}")
    await kit.process_inbound(
        InboundMessage(channel_id="oncall", sender_id="oncall", content=TextContent(body=text))
    )


async def follow(
    kit: RoomKit, start: float, seen: set[str], done: Callable[[RoomEvent], bool]
) -> None:
    """Print the agents' messages until one *done* accepts."""
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        await asyncio.sleep(0.2)
        for event in await kit.store.list_events(ROOM, limit=100):
            if (
                event.type != EventType.MESSAGE
                or event.status != EventStatus.DELIVERED
                or event.source.channel_id == "oncall"
                or event.id in seen
            ):
                continue
            seen.add(event.id)
            took = time.monotonic() - start
            print(f"\n[{took:5.1f} s] @{event.source.channel_id}: {event.content.body}")
            if done(event):
                return


async def turn_ended(kit: RoomKit, start: float, seen: set[str]) -> None:
    """Until no agent speaks: a coding agent writes between its tool rounds."""
    while (queue := kit.speak_queue(ROOM)) is not None and queue.speaking is not None:
        await follow(kit, start, seen, lambda _e: True)


async def scripted(kit: RoomKit) -> None:
    start, seen = time.monotonic(), set()
    await say(kit, "@dev why do Canadian checkouts fail? Look at the code.")
    await follow(kit, start, seen, lambda e: e.source.channel_id == "dev")
    await turn_ended(kit, start, seen)
    task = await kit.start_task(
        ROOM,
        "dev",
        "Go through support/tickets.txt: which customers are hit, and since when?",
        hand_back="post",
    )
    print(f"\n(task {task.id} started for @dev)")
    await say(kit, "@sre meanwhile, the checkout error rate?")
    await follow(kit, start, seen, lambda e: (e.metadata or {}).get("task_id") == task.id)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--console", action="store_true", help="lead the room yourself")
    args = parser.parse_args()
    if shutil.which("npx") is None:
        raise SystemExit("npx is needed to run Claude Code over ACP (Node.js 22+)")
    kit = await build(console=args.console)
    try:
        if args.console:
            await DiscussionConsole(
                kit, ROOM, channel_id="oncall", tasks=True, log_file="discussion_acp.log"
            ).run()
        else:
            await scripted(kit)
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
