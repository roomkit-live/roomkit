"""OVHcloud Olala -- chat with Olala 1.0 mini, its reasoning shown, calling tools.

Olala 1.0 mini is OVHcloud's open-weights model (Apache 2.0, 6.8B parameters
with about 1B active per token, 32K context, reasoning and tool calling).
OVHcloud AI Endpoints serves it behind an OpenAI-compatible API, so RoomKit
talks to it with ``OpenAIAIProvider`` and a ``base_url``. (Ollama and upstream
llama.cpp do not load the model yet.)

Create a key in the OVHcloud control panel (Public Cloud > AI Endpoints > API
keys), then run:

    OVH_AI_ENDPOINTS_ACCESS_TOKEN=... uv run python examples/ovhcloud_olala.py

Try:
    - "Quelle heure est-il ?"
    - "Lance trois dés à six faces et donne-moi le total."

Environment variables:
    OVH_AI_ENDPOINTS_ACCESS_TOKEN   AI Endpoints key (required)
    OLALA_MODEL       Model name (default: Olala-v1.0-mini-7A1B)
    OLALA_REASONING_EFFORT   low (default), medium or high. AI Endpoints
                      cannot turn reasoning off: it refuses ``none``
    CONSOLE=1         Branded console (Markdown, pinned input bar)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import console_enabled, log_tool_call, require_env, setup_logging  # noqa: E402

from roomkit import (  # noqa: E402
    AIChannel,
    ChannelCategory,
    CLIChannel,
    HookResult,
    HookTrigger,
    RoomKit,
)
from roomkit.providers.ai import AITool  # noqa: E402
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig  # noqa: E402

setup_logging("ovhcloud_olala")

BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
MODEL = os.environ.get("OLALA_MODEL", "Olala-v1.0-mini-7A1B")
EFFORT = os.environ.get("OLALA_REASONING_EFFORT", "low")

TOOLS = [
    AITool(
        name="get_time",
        description="The current local date and time",
        parameters={"type": "object", "properties": {}},
    ),
    AITool(
        name="roll_dice",
        description="Roll dice and return each result",
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "sides": {"type": "integer", "minimum": 2, "maximum": 100},
            },
            "required": ["count", "sides"],
        },
    ),
]


async def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool the model called and return its result as JSON."""
    if name == "get_time":
        return json.dumps({"now": datetime.now().isoformat(timespec="minutes")})
    if name == "roll_dice":
        rolls = [random.randint(1, arguments["sides"]) for _ in range(arguments["count"])]
        return json.dumps({"rolls": rolls, "total": sum(rolls)})
    return json.dumps({"error": f"unknown tool {name}"})


async def main() -> None:
    env = require_env("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
    provider = OpenAIAIProvider(
        OpenAIConfig(
            api_key=env["OVH_AI_ENDPOINTS_ACCESS_TOKEN"],
            base_url=BASE_URL,
            model=MODEL,
            # Reasoning and the answer share this budget: at 1024 a long
            # reasoning leaves no token for the answer.
            max_tokens=4096,
            timeout=120.0,
            reasoning_effort=EFFORT,
            # Olala reasons on tool turns too; without this the effort is
            # left out of every turn that declares tools (RFC 6.7).
            supports_reasoning_effort_with_tools=True,
            # The model card recommends temperature 0.8 (below), top_p 0.9 and
            # top_k 40. top_k is not sent: AI Endpoints refuses the field.
            extra_body={"top_p": 0.9},
        )
    )

    kit = RoomKit()
    console = console_enabled()
    cli = CLIChannel("cli", show_thinking=True, console=console)
    ai = AIChannel(
        "olala",
        provider=provider,
        system_prompt="You are a helpful assistant. Use the tools when they help. Be concise.",
        temperature=0.8,
        tools=TOOLS,
        tool_handler=run_tool,
    )
    kit.register_channel(cli)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event: Any, _ctx: Any) -> HookResult:
        return log_tool_call(event, show_result=True)

    await kit.create_room(room_id="olala-room")
    await kit.attach_channel("olala-room", "cli")
    await kit.attach_channel("olala-room", "olala", category=ChannelCategory.INTELLIGENCE)

    try:
        await cli.run(
            kit,
            room_id="olala-room",
            welcome=(
                f"\nOlala chat -- model={MODEL} server={BASE_URL}\n"
                f"Reasoning effort: {EFFORT} (OLALA_REASONING_EFFORT)\n"
                'Try: "Lance trois dés à six faces et donne-moi le total."\n'
                "Type 'quit' or Ctrl+D to exit.\n"
            ),
        )
    finally:
        await provider.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
