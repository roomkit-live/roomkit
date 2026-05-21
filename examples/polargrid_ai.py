"""PolarGrid AI example — Canadian-hosted chat completions.

Wires :class:`roomkit.providers.polargrid.PolarGridAIProvider` into a
:class:`roomkit.CLIChannel`. PolarGrid serves OpenAI-shaped chat
completions from regional edges in Toronto, Vancouver, and Montreal —
useful when data residency on Canadian soil is a requirement.

Note: tool / function calling is not exposed by PolarGrid's chat
endpoint at the time of writing. Passing tools to ``AIChannel`` will
log a warning and continue with text-only output.

Run with:
    POLARGRID_API_KEY=pg_... POLARGRID_MODEL=qwen-3.5-9b \\
        uv run python examples/polargrid_ai.py

Optional:
    POLARGRID_REGION=toronto    # pin a region (toronto/vancouver/montreal
                                #   or yto-01/yvr-02/yul-01).
                                # Unset to auto-route to the fastest edge.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging  # noqa: E402

from roomkit import (
    ChannelCategory,
    CLIChannel,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.polargrid import PolarGridAIProvider, PolarGridConfig

setup_logging("polargrid_ai")


async def main() -> None:
    api_key = os.environ.get("POLARGRID_API_KEY")
    if not api_key:
        print(
            "POLARGRID_API_KEY is required. Get one at the PolarGrid Console "
            "and export it before running this example.",
            file=sys.stderr,
        )
        sys.exit(2)

    model = os.environ.get("POLARGRID_MODEL", "qwen-3.5-9b")
    region = os.environ.get("POLARGRID_REGION") or None

    provider = PolarGridAIProvider(
        PolarGridConfig(api_key=api_key, model=model, region=region)
    )

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Keep answers concise.",
    )
    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "cli")
    await kit.attach_channel("demo-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    try:
        await cli.run(
            kit,
            room_id="demo-room",
            welcome=(
                f"\nPolarGrid AI demo — model={model} "
                f"region={region or 'auto'}\n"
                "Type a message and press Enter. Use 'quit' or Ctrl+D to exit.\n"
            ),
        )
    finally:
        await provider.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
