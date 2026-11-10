"""Interactive Meta Muse CLI — chat with Spark, Wonder or Ember through the pipeline.

Wires a :class:`CLIChannel` to an :class:`AIChannel` backed by the Meta Model
API: your input → room → AIChannel → api.meta.ai → streamed answer → terminal.

Muse chat models serve an OpenAI-compatible Chat Completions API with a 1M-token
window, images in, tools, and reasoning it cannot turn off: ``--effort`` tunes
how long it thinks (``minimal`` answers a one-liner in ~3 s, ``low`` in ~5 s),
and ``none`` is sent as ``minimal``. The reasoning is counted, not returned.

The ``-contributor`` models cost a fraction of the standard ones because Meta
trains its models on their traffic: pick one with ``--model`` only knowingly.

Requires:
    pip install roomkit[meta]   (and roomkit[console] for colored output)

Environment:
    META_API_KEY  — your Meta Model API key
    META_MODEL    — model id (default: muse-spark-1.3)

Run with:
    META_API_KEY=... uv run python examples/meta_ai.py
    META_API_KEY=... uv run python examples/meta_ai.py --effort minimal
    META_API_KEY=... uv run python examples/meta_ai.py --model muse-ember-1.0 --effort max

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

from roomkit import AIChannel, ChannelCategory, CLIChannel, RoomKit
from roomkit.providers.meta import MetaAIProvider, MetaConfig


async def main(args: argparse.Namespace) -> None:
    env = require_env("META_API_KEY")

    provider = MetaAIProvider(
        MetaConfig(
            api_key=env["META_API_KEY"],
            model=args.model,
            reasoning_effort=args.effort,
        )
    )

    kit = RoomKit()

    cli = CLIChannel("you")
    ai = AIChannel(
        "assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Think step by step, then answer concisely.",
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="meta-cli")
    await kit.attach_channel("meta-cli", "you")
    await kit.attach_channel("meta-cli", "assistant", category=ChannelCategory.INTELLIGENCE)

    try:
        await cli.run(
            kit,
            room_id="meta-cli",
            welcome=f"Meta · {args.model} (effort: {args.effort})\nType 'quit' to exit.",
        )
    finally:
        await provider.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Meta Muse CLI.")
    p.add_argument(
        "--model",
        default=os.environ.get("META_MODEL", "muse-spark-1.3"),
        help="Muse chat model id (e.g. muse-ember-1.0). Env: META_MODEL.",
    )
    p.add_argument(
        "--effort",
        default="low",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        help="Reasoning effort (max: Wonder/Ember). Muse always reasons: none becomes minimal.",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("meta_ai")
    asyncio.run(main(_parse_args()))
