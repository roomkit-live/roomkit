"""Interactive Mistral CLI — chat with reasoning (thinking) streamed live.

Wires a :class:`CLIChannel` (with ``show_thinking=True``) to an
:class:`AIChannel` backed by Mistral, so the model's reasoning trace renders
inline (💭) above each answer as it streams. This is the full RoomKit pipeline:
your input → room → AIChannel → Mistral → streamed thinking + answer → terminal.

Mistral's reasoning models (``mistral-medium-3-5``, ``mistral-small-latest``)
return reasoning as structured *thinking* chunks; the AIChannel's
``thinking_budget`` maps to Mistral's ``reasoning_effort`` so ``--no-think``
turns it off.

Requires:
    pip install roomkit[mistral]   (and roomkit[console] for colored output)

Environment:
    MISTRAL_API_KEY   — your Mistral API key
    MISTRAL_MODEL     — model id (default: mistral-medium-3-5)

Run with:
    MISTRAL_API_KEY=... uv run python examples/mistral_ai.py
    MISTRAL_API_KEY=... uv run python examples/mistral_ai.py --model magistral-medium-latest
    MISTRAL_API_KEY=... uv run python examples/mistral_ai.py --no-think

Type a message at the prompt. Type ``quit`` (or Ctrl+D) to exit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, setup_logging

from roomkit import CLIChannel, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import ChannelCategory
from roomkit.providers.mistral import MistralAIProvider, MistralConfig


async def main(args: argparse.Namespace) -> None:
    env = require_env("MISTRAL_API_KEY")

    provider = MistralAIProvider(MistralConfig(api_key=env["MISTRAL_API_KEY"], model=args.model))

    kit = RoomKit()

    cli = CLIChannel("you", show_thinking=not args.no_think)
    ai = AIChannel(
        "assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Think step by step, then answer concisely.",
        # >0 enables reasoning (mapped to Mistral reasoning_effort="high"); 0 disables it.
        thinking_budget=0 if args.no_think else 4096,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="mistral-cli")
    await kit.attach_channel("mistral-cli", "you")
    await kit.attach_channel("mistral-cli", "assistant", category=ChannelCategory.INTELLIGENCE)

    think_state = "off" if args.no_think else "on (💭 shown above each answer)"
    try:
        await cli.run(
            kit,
            room_id="mistral-cli",
            welcome=f"Mistral · {args.model} · thinking {think_state}\nType 'quit' to exit.",
        )
    finally:
        await provider.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Mistral CLI with live reasoning.")
    p.add_argument(
        "--model",
        default=os.environ.get("MISTRAL_MODEL", "mistral-medium-3-5"),
        help="Mistral model id. Env: MISTRAL_MODEL.",
    )
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Disable reasoning (reasoning_effort='none') and hide the thinking trace.",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("mistral_ai")
    asyncio.run(main(_parse_args()))
