"""Two coding agents in one Room — Claude Code and Codex, one console.

    terminal ──▶ CLIChannel ──▶ Room ──┬──▶ ACPChannel "claude-code"
                                       └──▶ ACPChannel "codex"

Both agents run in the same working directory, so the interesting flow needs
no shared conversation at all: ask one to write code, ask the other to review
what landed on disk. You address an agent with a mention, and **only the
addressed agent runs** — a ``ConversationRouter`` stamps the routing decision
and RoomKit skips every other intelligence channel for that event.

Agents do not see each other's messages here (see ``_addressed_to`` below for
the single line that decides it). That is deliberate: it keeps each agent's
session to what you asked it, and it keeps two agents from answering each
other in a loop until the chain-depth limit stops them.

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
    /agents                               keyboard menu: pick the addressed agent
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
    ChannelCategory,
    CLIChannel,
    ConversationRouter,
    HookExecution,
    HookTrigger,
    RoomKit,
    RoutingConditions,
    RoutingRule,
)
from roomkit.console import terminal_input, terminal_select
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent, TextContent
from roomkit.orchestration import ConversationState
from roomkit.tools import ExternalToolHandler, ToolDecision

CLAUDE_ACP = "@agentclientprotocol/claude-agent-acp@0.61.0"
CODEX_ACP = "@agentclientprotocol/codex-acp@1.1.9"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """One coding agent to put in the room."""

    channel_id: str
    package: str

    @property
    def label(self) -> str:
        """Display name — the same derivation the CLI channel uses."""
        return self.channel_id.replace("-", " ").title()


AGENTS = (
    AgentSpec(channel_id="claude-code", package=CLAUDE_ACP),
    AgentSpec(channel_id="codex", package=CODEX_ACP),
)


class Addressed:
    """The agent the next message goes to.

    A mutable cell rather than ``ConversationState.active_agent_id``: the
    router consults ``active_agent_id`` *before* its rules (sticky affinity),
    so setting it would make every rule below unreachable — including the one
    that reads this cell. Switching it also means a store round trip, which
    cannot happen synchronously while a line is being submitted.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id


class TerminalPermissionHandler(ExternalToolHandler):
    """Ask the console user to approve each tool call, naming who asks.

    One handler per agent: with two agents sharing a terminal, a prompt that
    does not say who wants to run ``rm`` is a prompt you cannot answer.
    """

    def __init__(self, agent_label: str) -> None:
        self._agent = agent_label

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
        prompt = (
            f"\n{self._agent} requests permission: {tool_name}\n{arguments}\nAllow once? [y/N] "
        )
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


def _addressed_to(addressed: Addressed, agent_id: str) -> Any:
    """Routing predicate: does *agent_id* handle this event?

    Two decisions live here, and they are the whole orchestration policy of
    this example:

    1. An agent's own output never triggers another agent. Without this, the
       first answer would be delivered to the other agent, which would answer
       it, and so on until ``max_chain_depth`` (5) stopped the ping-pong.
       Returning False for intelligence-sourced events makes the router
       select nobody, which blocks every agent for that event.
    2. Everything a human types goes to the addressed agent, and only to it.
    """

    def predicate(event: RoomEvent, context: RoomContext, state: ConversationState) -> bool:
        source = context.get_binding(event.source.channel_id)
        if source is not None and source.category == ChannelCategory.INTELLIGENCE:
            return False
        return addressed.agent_id == agent_id

    return predicate


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

    kit = RoomKit()
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
            external_tool_handler=TerminalPermissionHandler(spec.label),
        )
        kit.register_channel(channel)
        agents[spec.channel_id] = channel

    # One rule per agent. No default_agent_id: when no rule matches (an
    # agent's own output), the router blocks every intelligence channel
    # instead of falling back to somebody.
    router = ConversationRouter(
        rules=[
            RoutingRule(
                agent_id=spec.channel_id,
                conditions=RoutingConditions(custom=_addressed_to(addressed, spec.channel_id)),
            )
            for spec in AGENTS
        ]
    )
    kit.hook(
        HookTrigger.BEFORE_BROADCAST,
        execution=HookExecution.SYNC,
        priority=-100,
    )(router.as_hook())

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

    async def pick_agent(_argument: str = "") -> None:
        """Choose the addressed agent from a keyboard menu."""
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
                print(f"\nNo such agent: @{mention}. Try /agents.\n")
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
            # Awaited by the loop, in submission order — which is what lets
            # pick_agent() open a menu without racing the loop for stdin.
            commands={"/agents": pick_agent, "/model": switch_model},
            welcome=(
                f"Two coding agents in one room: {', '.join(f'@{s.channel_id}' for s in AGENTS)}\n"
                f"Workspace: {workspace}\n"
                f"Addressing @{addressed.agent_id} — only the addressed agent runs.\n"
                "'@agent your request' to address one, '/agents' for the picker,\n"
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
