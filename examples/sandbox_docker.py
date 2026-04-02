"""Sandbox with Docker — real container-based code execution.

Demonstrates using roomkit-sandbox with Docker to give an AI agent
sandboxed command execution via RTK. The agent runs commands inside
a lightweight Alpine container (37MB) with token-optimized output.

Prerequisites:
  - Docker running
  - Pull the sandbox image: docker pull ghcr.io/roomkit-live/sandbox:latest
  - pip install roomkit-sandbox[docker]

Try asking:
  - "List the files in /workspace"
  - "Clone https://github.com/rtk-ai/rtk and show the README"
  - "Search for 'fn main' in the cloned repo"
  - "Show the git log"
  - "Write a hello.py file and run it"

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/sandbox_docker.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import log_tool_call, require_env

from roomkit import ChannelCategory, CLIChannel, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    # Create a Docker-based sandbox executor
    from roomkit_sandbox import ContainerSandboxExecutor
    from roomkit_sandbox.backend import DockerSandboxBackend

    backend = DockerSandboxBackend(
        image="ghcr.io/roomkit-live/sandbox:latest",
        memory_limit="512m",
        cpu_count=1,
    )
    sandbox = ContainerSandboxExecutor(
        backend=backend,
        session_id="example-docker-sandbox",
    )

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"])),
        system_prompt=(
            "You are a helpful developer assistant with a sandboxed environment. "
            "You can read files, search code, run git commands, write files, and "
            "execute bash commands inside a Docker container. Use these tools to "
            "help the user explore and understand code."
        ),
        sandbox=sandbox,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

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
            "\nDocker Sandbox demo — tools: "
            + ", ".join(tools)
            + "\nCommands run inside a lightweight Alpine container with RTK.\n"
        ),
    )

    # Cleanup
    await sandbox.close()
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
