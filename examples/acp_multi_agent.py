"""Two coding agents in one Room — Claude Code and Codex, one console.

    terminal ──▶ CLIChannel ──▶ Room ──┬──▶ ACPChannel "claude-code"
                                       └──▶ ACPChannel "codex"

Both agents run in the same working directory, so the interesting flow needs
no shared conversation at all: ask one to write code, ask the other to review
what landed on disk.

Every submission names its recipient (``addressed_to``), so **only the
addressed agent runs** — and the room is created ``ADDRESSED_ONLY``, so an
agent's own output solicits nobody either. Two settings, no routing rules:
each agent's session holds what you asked *it*, and the two never answer
each other down to the chain-depth limit.

How you name an agent is this example's business, not RoomKit's: it accepts
``@codex review hello.py`` and ``/agent`` for a picker, and passes channel
ids.

Requires:
    pip install "roomkit[acp,console]"
    Node.js 22+ with ``npx`` available

Authenticate each agent once:
    npx -y @agentclientprotocol/claude-agent-acp@0.61.0 --cli auth login --claudeai
    npx -y @agentclientprotocol/codex-acp@1.1.9 --help   # then: codex login
    # or set ANTHROPIC_API_KEY / CODEX_API_KEY (OPENAI_API_KEY also works)

Run with:
    CONSOLE=1 uv run python examples/acp_multi_agent.py
    uv run python examples/acp_multi_agent.py --workspace /path/to/project

At the prompt:
    @codex what does this project do?     address an agent (and keep talking to it)
    @claude-code write hello.py           address the other one
    /agent                                keyboard menu: pick the addressed agent
    /agent codex                          switch without the menu
    /model sonnet                         switch the addressed agent's model
    quit                                  leave
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import console_enabled, existing_directory, setup_logging

from roomkit import (
    ACPChannel,
    AgentResponsePolicy,
    ChannelCategory,
    CLIChannel,
    RoomKit,
)
from roomkit.console import terminal_input, terminal_select
from roomkit.models.event import TextContent
from roomkit.tools import ExternalToolHandler, ToolDecision

CLAUDE_ACP = "@agentclientprotocol/claude-agent-acp@0.61.0"
CODEX_ACP = "@agentclientprotocol/codex-acp@1.1.9"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """One coding agent to put in the room."""

    channel_id: str
    package: str


AGENTS = (
    AgentSpec(channel_id="claude-code", package=CLAUDE_ACP),
    AgentSpec(channel_id="codex", package=CODEX_ACP),
)


class Addressed:
    """Who the next message is for — this console's own notion of focus.

    Plain application state: RoomKit is told the decision (a channel id, on
    every submission), never the syntax that produced it.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id


class TerminalPermissionHandler(ExternalToolHandler):
    """Ask the console user to approve each tool call, naming who asks.

    One handler instance per agent — the framework wires each to its channel,
    so ``self.channel_id`` answers "who wants to run this?". With two agents
    sharing one terminal, a prompt that cannot say that is a prompt you
    cannot answer.
    """

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
            return ToolDecision(approved=False, reason="Denied by a BEFORE_TOOL_USE hook")

        arguments = json.dumps(tool_input, indent=2, ensure_ascii=False, default=str)
        # The channel id is what you type to address it — say it the same way.
        who = f"@{self.channel_id}" if self.channel_id else "The agent"
        prompt = f"\n{who} requests permission: {tool_name}\n{arguments}\nAllow once? [y/N] "
        try:
            answer = await terminal_input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ToolDecision(approved=False, reason="No terminal approval")

        approved = answer.strip().casefold() in {"y", "yes", "o", "oui"}
        return ToolDecision(
            approved=approved, reason="" if approved else "Rejected in the terminal"
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


def _resolve_agent(name: str) -> str | None:
    """Match a typed mention against the agents in the room."""
    wanted = name.strip().casefold().replace(" ", "-")
    if not wanted:
        return None
    for spec in AGENTS:
        if spec.channel_id == wanted:
            return spec.channel_id
    # Prefix match, so "@claude" reaches "claude-code".
    matches = [spec.channel_id for spec in AGENTS if spec.channel_id.startswith(wanted)]
    return matches[0] if len(matches) == 1 else None


async def main(args: argparse.Namespace) -> None:
    workspace = args.workspace
    addressed = Addressed(AGENTS[0].channel_id)

    # ADDRESSED_ONLY: an agent's output solicits nobody it did not address.
    # Under the default (AGENT_CHAIN) the first answer would reach the other
    # agent, whose answer would come back, down to max_chain_depth.
    kit = RoomKit(agent_response_policy=AgentResponsePolicy.ADDRESSED_ONLY)
    cli = CLIChannel(
        "you",
        show_thinking=args.thinking_tokens > 0,
        markdown=True,
        console=console_enabled(),
    )
    kit.register_channel(cli)

    agents: dict[str, ACPChannel] = {}
    for spec in AGENTS:
        env: dict[str, str] = {}
        if spec.channel_id == "claude-code":
            if api_key := os.environ.get("ANTHROPIC_API_KEY"):
                env["ANTHROPIC_API_KEY"] = api_key
            env["MAX_THINKING_TOKENS"] = str(args.thinking_tokens)
        channel = ACPChannel(
            spec.channel_id,
            command=["npx", "-y", spec.package],
            cwd=workspace,
            env=env or None,
            # Both agents keep their credentials under $HOME (which the SDK
            # already passes); SSH_AUTH_SOCK is what git-over-SSH needs.
            inherit_env=["SSH_AUTH_SOCK"],
            external_tool_handler=TerminalPermissionHandler(),
        )
        kit.register_channel(channel)
        agents[spec.channel_id] = channel

    room_id = "coding-agents"
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, cli.channel_id)
    for spec in AGENTS:
        await kit.attach_channel(room_id, spec.channel_id, category=ChannelCategory.INTELLIGENCE)

    def agent_options() -> list[tuple[str, str]]:
        options = []
        for spec in AGENTS:
            model = agents[spec.channel_id].session_config(room_id).get("model")
            state = f" · {model}" if model else " · no session yet"
            options.append((spec.channel_id, f"@{spec.channel_id}{state}"))
        return options

    async def pick_agent(argument: str = "") -> None:
        """``/agent`` opens the menu; ``/agent codex`` switches directly."""
        if argument:
            target = _resolve_agent(argument)
            if target is None:
                print(f"\nNo such agent: {argument}. Try /agent for the list.\n")
                return
            addressed.agent_id = target
            print(f"\nNow talking to @{target}.\n")
            return
        chosen = await terminal_select(
            agent_options(),
            title="Address which agent?",
            default=addressed.agent_id,
        )
        if chosen is None:
            return
        addressed.agent_id = chosen
        print(f"\nNow talking to @{chosen}.\n")

    async def switch_model(requested: str) -> None:
        channel = agents[addressed.agent_id]
        if not requested:
            options = next(
                (item for item in channel.config_options(room_id) if item.get("id") == "model"),
                None,
            )
            current = channel.session_config(room_id).get("model", "unknown")
            choices = ", ".join(entry["value"] for entry in (options or {}).get("options", []))
            print(f"\n@{addressed.agent_id} model: {current}")
            print(f"Available: {choices or 'unknown until the first turn'}\n")
            return
        try:
            values = await channel.set_config_option(room_id, "model", requested)
        except Exception as exc:  # the agent rejects unknown values
            print(f"\nCould not switch model: {exc}\n")
            return
        print(f"\n@{addressed.agent_id} model: {values.get('model')}\n")

    def handle_line(line: str) -> TextContent | None:
        """@mentions, before anything reaches a Room. Commands are separate."""
        if line.startswith("@"):
            mention, _, rest = line[1:].partition(" ")
            target = _resolve_agent(mention)
            if target is None:
                print(f"\nNo such agent: @{mention}. Try /agent.\n")
                return None
            # Addressing is sticky: follow-ups stay with that agent until you
            # address another one, the way talking to a person works.
            addressed.agent_id = target
            body = rest.strip()
            if not body:
                print(f"\nNow talking to @{target}.\n")
                return None
            return TextContent(body=body)
        return TextContent(body=line)

    try:
        await cli.run(
            kit,
            room_id=room_id,
            sender_id="you",
            content_factory=handle_line,
            # Evaluated after content_factory, so an "@agent ..." line has
            # already moved the focus by the time we name the recipient.
            addressed_to=lambda _line: [addressed.agent_id],
            # Awaited by the loop, in submission order — which is what lets
            # pick_agent() open a menu without racing the loop for stdin.
            commands={"/agent": pick_agent, "/model": switch_model},
            welcome=(
                f"Two coding agents in one room: {', '.join(f'@{s.channel_id}' for s in AGENTS)}\n"
                f"Workspace: {workspace}\n"
                f"Addressing @{addressed.agent_id} — only the addressed agent runs.\n"
                "'@agent your request' to address one, '/agent' for the picker,\n"
                "'/model [name]' for the addressed one, 'quit' to exit."
            ),
        )
    finally:
        await kit.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claude Code and Codex in one RoomKit Room, addressed by mention."
    )
    parser.add_argument(
        "--workspace",
        type=existing_directory,
        default=Path.cwd(),
        help="Project directory exposed to both agents (default: current directory).",
    )
    parser.add_argument(
        "--thinking-tokens",
        type=int,
        default=0,
        help="Claude reasoning budget; 0 disables it (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging("acp_multi_agent")
    try:
        asyncio.run(main(_parse_args()))
    except KeyboardInterrupt:
        pass
