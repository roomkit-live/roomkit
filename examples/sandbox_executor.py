"""Sandbox Executor — give AI agents ad-hoc command execution in a container.

Demonstrates how to use the SandboxExecutor ABC with AIChannel.
When a sandbox is provided, RoomKit automatically injects tools for
file reading, search, git operations, and bash execution. The AI
decides when to use them based on the conversation.

This example uses a local mock executor. For production use, see
``roomkit-sandbox`` which provides a container-based executor using
Docker/Kubernetes with RTK (https://github.com/rtk-ai/rtk) for
token-optimized output.

Uses CLIChannel for interactive exploration. Try asking:
  - "List the files in the current directory"
  - "Read the contents of pyproject.toml"
  - "Search for all TODO comments in the codebase"
  - "Show me the git log"
  - "Clone the repo https://github.com/rtk-ai/rtk and show its README"

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/sandbox_executor.py
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import log_tool_call, require_env

from roomkit import (
    ChannelCategory,
    CLIChannel,
    HookTrigger,
    RoomKit,
    SandboxExecutor,
    SandboxResult,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.sandbox.tools import SANDBOX_TOOL_SCHEMAS


class LocalSandboxExecutor(SandboxExecutor):
    """Example executor that runs commands locally (NOT sandboxed).

    For production use, implement execution inside a Docker/Kubernetes
    container. See ``roomkit-sandbox`` for a ready-made solution.
    """

    def __init__(self, workdir: str = ".") -> None:
        self._workdir = workdir

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        args = arguments or {}
        try:
            cmd = self._build_command(command, args)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workdir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return SandboxResult(
                exit_code=proc.returncode or 0,
                output=stdout.decode(errors="replace"),
                error=stderr.decode(errors="replace"),
            )
        except Exception as exc:
            return SandboxResult(exit_code=1, error=str(exc))

    def tool_definitions(self) -> list[dict[str, Any]]:
        return SANDBOX_TOOL_SCHEMAS

    def _build_command(self, command: str, args: dict[str, Any]) -> list[str]:
        if command == "read":
            cmd = ["cat", "-n", args["path"]]
        elif command == "ls":
            cmd = ["ls", "-la", args.get("path", ".")]
        elif command == "grep":
            cmd = ["grep", "-rn", args["pattern"]]
            if args.get("path"):
                cmd.append(args["path"])
            else:
                cmd.append(".")
        elif command == "find":
            cmd = ["find", args.get("path", ".")]
            if args.get("name"):
                cmd.extend(["-name", args["name"]])
            if args.get("type"):
                cmd.extend(["-type", args["type"]])
        elif command == "git":
            cmd = ["git"] + shlex.split(args.get("args", "status"))
        elif command == "diff":
            cmd = ["diff", args["file_a"], args["file_b"]]
        elif command == "bash":
            cmd = ["bash", "-c", args["command"]]
        else:
            cmd = ["echo", f"Unknown command: {command}"]
        return cmd


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    # --- Create a sandbox executor ---
    sandbox = LocalSandboxExecutor(workdir=".")

    # --- Set up RoomKit ---
    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"])),
        system_prompt=(
            "You are a helpful developer assistant with access to a sandboxed "
            "development environment. You can read files, search code, run git "
            "commands, and execute bash commands. Use these tools to help the "
            "user explore and understand code."
        ),
        sandbox=sandbox,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    # Show sandbox tool invocations in the terminal
    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="sandbox")

    await kit.create_room(room_id="sandbox-room")
    await kit.attach_channel("sandbox-room", "cli")
    await kit.attach_channel("sandbox-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    tools = [t["name"] for t in sandbox.tool_definitions()]
    await cli.run(
        kit,
        room_id="sandbox-room",
        welcome=(
            "\nSandbox demo — the AI has access to: "
            + ", ".join(tools)
            + "\nAsk the AI to explore files, search code, or run commands.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
