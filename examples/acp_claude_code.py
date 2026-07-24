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

Type a coding request at the prompt. Type ``quit`` (or Ctrl+D) to exit.
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

from shared import setup_logging

from roomkit import ACPChannel, ChannelCategory, CLIChannel, RoomKit
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
            answer = await asyncio.to_thread(input, prompt)
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
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Workspace does not exist or is not a directory: {workspace}")

    agent_env: dict[str, str] = {}
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        agent_env["ANTHROPIC_API_KEY"] = api_key
    agent_env["MAX_THINKING_TOKENS"] = str(args.thinking_tokens)

    kit = RoomKit()
    cli = CLIChannel(
        "you",
        show_thinking=args.thinking_tokens > 0,
        agent_label=lambda _channel_id: "Claude Code",
        markdown=True,
    )
    claude = ACPChannel(
        "claude-code",
        command=["npx", "-y", CLAUDE_AGENT_ACP_PACKAGE],
        cwd=workspace,
        env=agent_env or None,
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

    try:
        await cli.run(
            kit,
            room_id=room_id,
            welcome=(
                f"Claude Code via ACP {CLAUDE_AGENT_ACP_VERSION}\n"
                f"Workspace: {workspace}\n"
                f"Thinking budget: {args.thinking_tokens} tokens\n"
                "Tool permissions are requested in this terminal.\n"
                "Type a request, or 'quit' to exit."
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
        type=Path,
        default=Path.cwd(),
        help="Project directory exposed to Claude Code (default: current directory).",
    )
    parser.add_argument(
        "--thinking-tokens",
        type=_non_negative_int,
        default=1024,
        help="Visible Claude reasoning budget; 0 disables it (default: 1024).",
    )
    return parser.parse_args()


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


if __name__ == "__main__":
    setup_logging("acp_claude_code")
    asyncio.run(main(_parse_args()))
