"""vLLM + HuggingFace — chat with any local HuggingFace model.

Serve any HuggingFace model locally with vLLM and wire it into RoomKit.
This example uses Chocolatine-2-4B, a French-language model, for a
casual conversation about the great French pastry debate.

Start the vLLM server first:

    # GPU <= 12 GB (RTX 4070, 3060, etc.):
    uv run vllm serve jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1 \
        --port 8000 --enforce-eager --max-model-len 3072

    # GPU >= 24 GB (RTX 4090, A5000, etc.):
    uv run vllm serve jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1 --port 8000

    # Docker (NVIDIA GPU):
    docker run --gpus all -p 8000:8000 \
        vllm/vllm-openai:latest \
        --model jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1 \
        --enforce-eager --max-model-len 3072

Then run:
    uv run python examples/vllm_huggingface.py

Try:
    - "C'est quoi une chocolatine ?"
    - "Pain au chocolat ou chocolatine ?"
    - "Donne-moi ta meilleure recette de croissant"
    - "Quel est le meilleur fromage francais ?"

Environment variables (optional):
    VLLM_MODEL     Model name (default: jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1)
    VLLM_BASE_URL  Server URL (default: http://localhost:8000/v1)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from roomkit import ChannelCategory, CLIChannel, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.memory import SlidingWindowMemory
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider

MODEL = os.environ.get("VLLM_MODEL", "jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1")
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


async def main() -> None:
    # --- vLLM provider --------------------------------------------------------
    provider = create_vllm_provider(
        VLLMConfig(
            model=MODEL,
            base_url=BASE_URL,
            max_tokens=256,
            temperature=0.8,
        )
    )

    # --- RoomKit setup --------------------------------------------------------
    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt=(
            "Tu es un assistant francophone passionne de gastronomie francaise. "
            "Tu adores debattre de la question 'pain au chocolat vs chocolatine' "
            "et tu as un avis bien tranche (tu defends la chocolatine, evidemment). "
            "Tu connais les patisseries, les fromages, les vins et la cuisine "
            "regionale. Tu es drole, chaleureux et un peu chauvin sur la cuisine "
            "francaise. Reponds toujours en francais, de facon concise."
        ),
        # Small-model context management: keep only the last 4 exchanges
        # in context to stay within the tight token budget (~3072 tokens).
        # On overflow, AIChannel auto-compacts by summarizing older messages.
        memory=SlidingWindowMemory(max_events=4),
        max_context_events=4,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="chat-room")
    await kit.attach_channel("chat-room", "cli")
    await kit.attach_channel("chat-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    await cli.run(
        kit,
        room_id="chat-room",
        welcome=(
            f"\nLocal AI chat (vLLM + HuggingFace)\n"
            f"Model: {MODEL}\n"
            f"Server: {BASE_URL}\n\n"
            'Try: "Pain au chocolat ou chocolatine ?"\n'
            ' or: "Quel est le meilleur fromage francais ?"\n'
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
