"""Ollama AI example — AI-powered assistant via Ollama's native API.

Uses ``roomkit.providers.ollama.OllamaAIProvider`` rather than the
OpenAI-compatible shim, so the model's ``think`` parameter and the
streamed ``thinking`` field actually work. Wires the provider into a
:class:`roomkit.CLIChannel` with ``show_thinking=True`` so reasoning
streams inline (dim italic) above each answer, in arrival order with
the text.

See ``ollama_cli.py`` for a lower-level test bed that exercises
``think`` on/off, streaming on/off, and MCP tool calls without going
through RoomKit's channel pipeline.

Run with:
    OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=qwen3:8b \\
        uv run python examples/ollama_ai.py

Set ``OLLAMA_API_KEY`` to authenticate against a protected endpoint
(Ollama Cloud/Turbo, or a self-hosted server behind a Bearer-checking
reverse proxy); it is forwarded as ``Authorization: Bearer <key>``.

``OLLAMA_THINK`` is tri-state:
  * ``0`` / ``false`` / ``off`` / ``no``  → thinking disabled
  * ``low`` / ``medium`` / ``high``       → thinking on at that effort
                                            (reasoning models on Ollama 0.7+)
  * anything else (including unset)       → thinking on, model's default effort
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import env_bool, setup_logging  # noqa: E402

from roomkit import (
    ChannelCategory,
    CLIChannel,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig, ThinkEffort

setup_logging("ollama_ai")

_EFFORT_MAP: dict[str, ThinkEffort] = {"low": "low", "medium": "medium", "high": "high"}


def _describe_thinking(enabled: bool, effort: ThinkEffort | None) -> str:
    if not enabled:
        return "disabled"
    return f"enabled, effort={effort}" if effort else "enabled (model default effort)"


async def main() -> None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    think_on = env_bool("OLLAMA_THINK", default=True)
    effort: ThinkEffort | None = _EFFORT_MAP.get(
        os.environ.get("OLLAMA_THINK", "").strip().lower()
    )
    thinking_budget = 4096 if think_on else 0

    # api_key=None when unset → the SDK still honors OLLAMA_API_KEY itself.
    # Pass it explicitly here to show how a key from a secret manager would
    # flow into the provider config. `or None` collapses an exported-but-empty
    # var to the clean no-auth path instead of an empty Bearer header.
    api_key = os.environ.get("OLLAMA_API_KEY") or None
    provider = OllamaAIProvider(
        OllamaConfig(host=host, model=model, think=effort, api_key=api_key)
    )

    kit = RoomKit()

    cli = CLIChannel("cli", show_thinking=think_on)
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Keep answers concise.",
        thinking_budget=thinking_budget,
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
                f"\nOllama AI demo — model={model} host={host}\n"
                f"Thinking: {_describe_thinking(think_on, effort)}"
                " — OLLAMA_THINK=0|low|medium|high\n"
                "Type a message and press Enter. Use 'quit' or Ctrl+D to exit.\n"
            ),
        )
    finally:
        await provider.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
