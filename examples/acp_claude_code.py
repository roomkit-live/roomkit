"""Interactive Claude Code session through RoomKit's ACP channel.

This is the full RoomKit flow:

    terminal -> CLIChannel -> Room -> ACPChannel -> Claude Code

Claude Code runs as an ACP agent subprocess. Each tool permission is presented
in the terminal and, when approved, is granted once.

Requires:
    pip install "roomkit[acp,console]"
    Node.js 22+ with ``npx`` available

The CLI uses Rich's live Markdown renderer. It refreshes for every text delta
received from ACP while keeping headings, lists, links, and code blocks
formatted.

Authenticate once with a Claude subscription:
    npx -y @agentclientprotocol/claude-agent-acp@0.61.0 \\
        --cli auth login --claudeai

Alternatively, set ``ANTHROPIC_API_KEY``; the example forwards it explicitly
to the ACP subprocess.

Run with:
    uv run python examples/acp_claude_code.py
    uv run python examples/acp_claude_code.py --workspace /path/to/project
    uv run python examples/acp_claude_code.py --thinking-tokens 0  # faster, no reasoning
    uv run python examples/acp_claude_code.py --model sonnet       # pin at startup
    CONSOLE=1 uv run python examples/acp_claude_code.py  # branded console mode

Type a coding request at the prompt. ``/model`` shows the running model and
the available ones, ``/model sonnet`` switches. Type ``quit`` (or Ctrl+D) to
exit.

The host contributes context the agent cannot fetch (``context_contributor``,
see ``host_notes`` below). Ask ``when can we deploy this?`` and the answer
comes back with the Tuesday rule the agent was never told by anyone else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import console_enabled, existing_directory, non_negative_int, setup_logging

from roomkit import (
    ACPChannel,
    Channel,
    ChannelCategory,
    CLIChannel,
    HookExecution,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
)
from roomkit.console import terminal_input
from roomkit.tools import ExternalToolHandler, ToolDecision

CLAUDE_AGENT_ACP_VERSION = "0.61.0"
CLAUDE_AGENT_ACP_PACKAGE = f"@agentclientprotocol/claude-agent-acp@{CLAUDE_AGENT_ACP_VERSION}"

HOST_NOTES = {
    "deploy": "Deploys go out on Tuesdays only, and never after 15:00.",
    "release": "The changelog is hand-maintained; the release script does not write it.",
    "review": "Two approvals are required before anything lands on main.",
}
"""Facts this host holds and the agent has no way to fetch.

A real one would query the member's saved memories or a document corpus. The
shape is what matters: it is not on disk, not in the room, and the agent
cannot go and look.
"""


async def host_notes(context: RoomContext, trigger: RoomEvent) -> list[str]:
    """Select the notes this turn's request actually needs.

    Request-dependent on purpose. The ACP session keeps whatever it was
    already told, so a block that never changes is paid for again on every
    turn — standing instructions belong in the workspace's ``CLAUDE.md``, not
    here. A contributor that raised would cost its blocks, not the turn.
    """
    asked = Channel.extract_text(trigger).casefold()
    return [f"[Host note] {note}" for topic, note in HOST_NOTES.items() if topic in asked]


class TerminalPermissionHandler(ExternalToolHandler):
    """Ask the CLI user to approve each ACP tool call once."""

    async def process_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        room_id: str | None = None,
    ) -> ToolDecision:
        hook_allowed = await self._fire_before_hook(
            tool_name,
            tool_input,
            tool_call_id=tool_call_id,
            room_id=room_id,
        )
        if not hook_allowed:
            return ToolDecision(
                approved=False,
                reason="Denied by a RoomKit BEFORE_TOOL_USE hook",
            )

        arguments = json.dumps(tool_input, indent=2, ensure_ascii=False, default=str)
        prompt = f"\nClaude Code requests permission: {tool_name}\n{arguments}\nAllow once? [y/N] "
        try:
            # Suspends the pinned input bar (CONSOLE=1) for the read.
            answer = await terminal_input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ToolDecision(approved=False, reason="No terminal approval")

        approved = answer.strip().casefold() in {"y", "yes", "o", "oui"}
        return ToolDecision(
            approved=approved,
            reason="" if approved else "Rejected in the terminal",
        )

    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        *,
        is_error: bool = False,
        tool_call_id: str = "",
        job_id: str | None = None,
        room_id: str | None = None,
    ) -> None:
        await self._fire_on_tool_hook(
            tool_name,
            tool_input,
            result,
            tool_call_id=tool_call_id,
            room_id=room_id,
        )


async def main(args: argparse.Namespace) -> None:
    workspace = args.workspace

    agent_env: dict[str, str] = {}
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        agent_env["ANTHROPIC_API_KEY"] = api_key
    agent_env["MAX_THINKING_TOKENS"] = str(args.thinking_tokens)
    if args.model:
        # Highest-priority model pin for claude-agent-acp, read when the
        # session opens; /model switches it afterwards.
        agent_env["ANTHROPIC_MODEL"] = args.model

    kit = RoomKit()
    # console mode subsumes markdown; CONSOLE=1 adds the branded banner.
    cli = CLIChannel(
        "you",
        show_thinking=args.thinking_tokens > 0,
        agent_label=lambda _channel_id: "Claude Code",
        markdown=True,
        console=console_enabled(),
    )
    claude = ACPChannel(
        "claude-code",
        command=["npx", "-y", CLAUDE_AGENT_ACP_PACKAGE],
        cwd=workspace,
        env=agent_env or None,
        # The ACP SDK strips the environment to a minimal set; forward the
        # ssh-agent socket so the agent's git-over-SSH doesn't prompt for
        # key passphrases in this terminal.
        inherit_env=["SSH_AUTH_SOCK"],
        external_tool_handler=TerminalPermissionHandler(),
        # What only this process knows, added to the turn that needs it. The
        # blocks open the prompt, ahead of the room catch-up and the request.
        context_contributor=host_notes,
    )

    kit.register_channel(cli)
    kit.register_channel(claude)

    room_id = "claude-code-cli"
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, cli.channel_id)
    await kit.attach_channel(
        room_id,
        claude.channel_id,
        category=ChannelCategory.INTELLIGENCE,
    )

    billed_so_far = 0.0

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def turn_finished(event: Any, _ctx: Any) -> None:
        """What a host learns when the agent has finished answering.

        The trigger follows the channel's INTELLIGENCE category rather than
        its class, so an agent running its own tool loop in another process
        reports the end of a turn exactly like an in-process provider does —
        which is what makes post-processing (summaries, memory, metrics)
        possible for a conversation an agent held.

        Priced from ``cost``, which ACP reports as the session's running
        total — that one genuinely accumulates — so a turn's own price is the
        difference from the last reading. RoomKit relays the figure without
        differencing it, because which difference is wanted belongs to whoever
        is counting; this is one of them.

        The token counters explain the price rather than set it. A turn that
        answers in three words still reports tens of thousands, nearly all of
        it the agent's preamble and the project's context re-read from cache
        at a fraction of a fresh token's price. Summing ``total_tokens`` over
        a conversation therefore counts the same prefix once per turn and
        means nothing in money.
        """
        nonlocal billed_so_far
        spend = ""
        if event.usage:
            total = event.usage.get("cost")
            if total is not None:
                turn_cost = total - billed_so_far
                billed_so_far = total
                spend += f" · {turn_cost:.4f} {event.usage.get('currency', '')}".rstrip()
            spend += (
                f" · {event.usage.get('total_tokens', 0)} tokens"
                f" ({event.usage.get('input_tokens', 0)} in"
                f" / {event.usage.get('output_tokens', 0)} out"
                f" / {event.usage.get('cached_read_tokens', 0)} cached)"
            )
        print(f"\n[turn] {event.tool_calls_count} tool call(s) · {event.latency_ms}ms{spend}\n")

    async def switch_model(requested: str) -> None:
        """Handle ``/model`` here instead of letting it reach the agent.

        Claude Code answers its own ``/model`` locally and tells nobody, so
        the session config RoomKit tracks (and the status bar reading it)
        would go stale. Routing the switch through ACP keeps both honest.
        """
        options = next(
            (item for item in claude.config_options(room_id) if item.get("id") == "model"),
            None,
        )
        if not requested:
            current = claude.session_config(room_id).get("model", "unknown")
            choices = ", ".join(entry["value"] for entry in (options or {}).get("options", []))
            print(f"\nModel: {current}\nAvailable: {choices or 'unknown until the first turn'}\n")
            return
        try:
            values = await claude.set_config_option(room_id, "model", requested)
        except Exception as exc:  # the agent rejects unknown values
            print(f"\nCould not switch model: {exc}\n")
            return
        print(f"\nModel: {values.get('model')}\n")

    try:
        await cli.run(
            kit,
            room_id=room_id,
            # Awaited by the loop, in submission order: the switch lands
            # between turns instead of racing the one it would affect.
            commands={"/model": switch_model},
            welcome=(
                f"Claude Code via ACP {CLAUDE_AGENT_ACP_VERSION}\n"
                f"Workspace: {workspace}\n"
                f"Thinking budget: {args.thinking_tokens} tokens\n"
                "Tool permissions are requested in this terminal.\n"
                "Type a request, '/model [name]' to see or switch models, "
                "or 'quit' to exit."
            ),
        )
    finally:
        await kit.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive Claude Code through RoomKit's CLIChannel and ACPChannel."
    )
    parser.add_argument(
        "--workspace",
        type=existing_directory,
        default=Path.cwd(),
        help="Project directory exposed to Claude Code (default: current directory).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model the agent starts on (alias or id, e.g. 'sonnet'); /model switches it.",
    )
    parser.add_argument(
        "--thinking-tokens",
        type=non_negative_int,
        default=1024,
        help="Visible Claude reasoning budget; 0 disables it (default: 1024).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging("acp_claude_code")
    try:
        asyncio.run(main(_parse_args()))
    except KeyboardInterrupt:
        # Ctrl-C is how you leave; a traceback is not a goodbye.
        pass
