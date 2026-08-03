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

from roomkit import ACPChannel, ChannelCategory, CLIChannel, RoomKit
from roomkit.console import terminal_input
from roomkit.models.event import TextContent
from roomkit.tools import ExternalToolHandler, ToolDecision

CLAUDE_AGENT_ACP_VERSION = "0.61.0"
CLAUDE_AGENT_ACP_PACKAGE = f"@agentclientprotocol/claude-agent-acp@{CLAUDE_AGENT_ACP_VERSION}"


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

    def handle_line(line: str) -> TextContent | None:
        """Intercept ``/model`` before it reaches the agent as a prompt.

        Claude Code answers its own ``/model`` locally and tells nobody, so
        the session config RoomKit tracks (and the status bar reading it)
        would go stale. Routing the switch through ACP keeps both honest.
        """
        if not line.startswith("/model"):
            return TextContent(body=line)
        _, _, requested = line.partition(" ")
        asyncio.get_running_loop().create_task(switch_model(requested.strip()))
        return None

    async def switch_model(requested: str) -> None:
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
            content_factory=handle_line,
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
