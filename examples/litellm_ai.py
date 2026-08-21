"""Interactive LiteLLM proxy CLI — chat through your AI gateway's models.

Wires a :class:`CLIChannel` to an :class:`AIChannel` backed by a LiteLLM proxy,
so you can talk to any model your gateway routes (Anthropic, OpenAI, Google,
local vLLM/Ollama, …) by its deployment alias, with the gateway enforcing
virtual keys and budgets. This is the full RoomKit pipeline: your input → room
→ AIChannel → LiteLLM proxy → upstream provider → streamed answer → terminal.

The proxy speaks the OpenAI Chat Completions API and normalises reasoning into
``reasoning_content``, so thinking models render their trace inline (💭) above
the answer when ``show_thinking`` is on.

Requires:
    pip install roomkit[litellm]   (and roomkit[console] for colored output)

    A running LiteLLM proxy. Quickest start (no upstream key needed — the
    proxy mocks the response):

        cat > /tmp/litellm-config.yaml <<'EOF'
        model_list:
          - model_name: mock-model
            litellm_params:
              model: openai/mock
              mock_response: "Hello from the gateway!"
        EOF
        uvx --from 'litellm[proxy]' litellm --config /tmp/litellm-config.yaml

Environment:
    LITELLM_API_KEY  — virtual key or master key (any value for a no-auth dev proxy)
    LITELLM_BASE_URL — proxy URL (default: http://localhost:4000)
    LITELLM_MODEL    — public model name as the proxy configures it

Run with:
    LITELLM_API_KEY=sk-... uv run python examples/litellm_ai.py --model mock-model
    LITELLM_API_KEY=sk-... uv run python examples/litellm_ai.py --list-models

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
from roomkit.providers.litellm import LiteLLMAIProvider, LiteLLMConfig


async def list_models(provider: LiteLLMAIProvider) -> None:
    """Print the deployment's models with the metadata /model/info reports."""
    for model in await provider.list_models():
        window = f"{model.context_window:,} tokens" if model.context_window else "window unknown"
        price = (
            f"${model.pricing.input_per_million:.2f}/M in, "
            f"${model.pricing.output_per_million:.2f}/M out"
            if model.pricing
            else "unpriced"
        )
        print(f"  {model.id:<40} {window:>20}   {price}")


async def main(args: argparse.Namespace) -> None:
    env = require_env("LITELLM_API_KEY")

    provider = LiteLLMAIProvider(
        LiteLLMConfig(
            api_key=env["LITELLM_API_KEY"],
            base_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000"),
            model=args.model,
        )
    )

    if args.list_models:
        try:
            await list_models(provider)
        finally:
            await provider.close()
        return

    kit = RoomKit()

    cli = CLIChannel("you", show_thinking=not args.no_think)
    ai = AIChannel(
        "assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Think step by step, then answer concisely.",
        # >0 requests reasoning through LiteLLM's normalised thinking budget;
        # 0 turns it off (reasoning_effort="none"). Works on any
        # reasoning-capable route the gateway fronts.
        thinking_budget=0 if args.no_think else 4096,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="litellm-cli")
    await kit.attach_channel("litellm-cli", "you")
    await kit.attach_channel("litellm-cli", "assistant", category=ChannelCategory.INTELLIGENCE)

    try:
        await cli.run(
            kit,
            room_id="litellm-cli",
            welcome=f"LiteLLM proxy · {args.model}\nType 'quit' to exit.",
        )
    finally:
        await provider.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive LiteLLM proxy CLI (your AI gateway).")
    p.add_argument(
        "--model",
        default=os.environ.get("LITELLM_MODEL", "mock-model"),
        help="Public model name as the proxy configures it. Env: LITELLM_MODEL.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List the deployment's models (context window, pricing) and exit.",
    )
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Disable reasoning and hide the thinking trace.",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("litellm_ai")
    asyncio.run(main(_parse_args()))
