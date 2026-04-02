"""Sandbox with SmolBSD — VM-isolated code execution for local assistants.

Demonstrates using roomkit-sandbox with SmolBSD to give an AI agent
sandboxed command execution inside a NetBSD microVM. Provides true
VM isolation (~10ms boot) for local AI assistants.

Prerequisites:
  - SmolBSD installed: https://github.com/NetBSDfr/smolBSD
  - Incus running (Linux) or OrbStack VM (macOS)
  - Golden image created: sandbox-setup
  - pip install roomkit-sandbox

Try asking:
  - "List the files in /workspace"
  - "Clone a git repo and explore it"
  - "Write a script and run it"
  - "Show system information"

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/sandbox_smolbsd.py

Or with a local model (Ollama):
    OLLAMA_HOST=http://localhost:11434 uv run python examples/sandbox_smolbsd.py --ollama
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

    # Create a SmolBSD-based sandbox executor
    from roomkit_sandbox import ContainerSandboxExecutor
    from roomkit_sandbox.smolbsd_backend import SmolBSDSandboxBackend

    backend = SmolBSDSandboxBackend(
        stack="base",  # Use the base golden image
        workdir="/workspace",
    )
    sandbox = ContainerSandboxExecutor(
        backend=backend,
        session_id="example-smolbsd-sandbox",
    )

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"])),
        system_prompt=(
            "You are a helpful developer assistant with a sandboxed environment. "
            "You can read files, search code, run git commands, write files, and "
            "execute bash commands inside an isolated NetBSD microVM. Use these "
            "tools to help the user explore and understand code."
        ),
        sandbox=sandbox,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="smolbsd")

    await kit.create_room(room_id="sandbox-room")
    await kit.attach_channel("sandbox-room", "cli")
    await kit.attach_channel("sandbox-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    tools = [t["name"] for t in sandbox.tool_definitions()]
    await cli.run(
        kit,
        room_id="sandbox-room",
        welcome=(
            "\nSmolBSD Sandbox demo — tools: "
            + ", ".join(tools)
            + "\nCommands run inside a NetBSD microVM with true VM isolation.\n"
        ),
    )

    # Cleanup
    await sandbox.close()
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
