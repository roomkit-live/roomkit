"""vLLM + HuggingFace -- assistant IA local avec sandbox conteneur.

Sert n'importe quel modele HuggingFace localement avec vLLM et le branche
dans RoomKit avec un sandbox Docker pour l'execution de commandes.
Cet exemple utilise Chocolatine-2-4B, un modele francophone.

Lancer le serveur vLLM d'abord :

    # GPU <= 12 Go (RTX 4070, 3060, etc.) :
    uv run vllm serve jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1 \
        --port 8000 --enforce-eager --max-model-len 4096

    # GPU >= 24 Go (RTX 4090, A5000, etc.) -- parametres par defaut :
    uv run vllm serve jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1 --port 8000

    # Docker (GPU NVIDIA) :
    docker run --gpus all -p 8000:8000 \
        vllm/vllm-openai:latest \
        --model jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1 \
        --enforce-eager --max-model-len 4096

Pre-requis sandbox :
    - Docker en cours d'execution
    - docker pull ghcr.io/roomkit-live/sandbox:latest
    - pip install roomkit-sandbox[docker]

Puis lancer l'exemple :
    uv run python examples/vllm_huggingface.py

Essayez :
    - "Liste les fichiers dans /workspace"
    - "Ecris un script Python hello.py et execute-le"
    - "Montre le contenu de /etc/os-release"
    - "Clone https://github.com/rtk-ai/rtk et montre le README"

Variables d'environnement (optionnel) :
    VLLM_MODEL     Nom du modele (defaut: jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1)
    VLLM_BASE_URL  URL du serveur (defaut: http://localhost:8000/v1)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import log_tool_call

from roomkit import ChannelCategory, CLIChannel, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider

MODEL = os.environ.get("VLLM_MODEL", "jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1")
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


async def main() -> None:
    from roomkit_sandbox import ContainerSandboxExecutor
    from roomkit_sandbox.docker_backend import DockerSandboxBackend

    # --- Sandbox conteneur ----------------------------------------------------
    backend = DockerSandboxBackend(
        image="ghcr.io/roomkit-live/sandbox:latest",
        memory_limit="512m",
        cpu_count=1,
    )
    sandbox = ContainerSandboxExecutor(
        backend=backend,
        session_id="vllm-hf-sandbox",
    )

    # --- Provider vLLM --------------------------------------------------------
    provider = create_vllm_provider(
        VLLMConfig(
            model=MODEL,
            base_url=BASE_URL,
            max_tokens=256,
            temperature=0.7,
        )
    )

    # --- Configuration RoomKit ------------------------------------------------
    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt=(
            "Tu es un assistant developpeur francophone avec acces a un "
            "environnement sandbox conteneurise. Tu peux lire des fichiers, "
            "chercher dans le code, executer des commandes git et bash. "
            "Utilise ces outils pour aider l'utilisateur. "
            "Reponds toujours en francais."
        ),
        sandbox=sandbox,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="sandbox")

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "cli")
    await kit.attach_channel("demo-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    tools = [t["name"] for t in sandbox.tool_definitions()]
    await cli.run(
        kit,
        room_id="demo-room",
        welcome=(
            f"\nAssistant IA local (vLLM + HuggingFace)\n"
            f"Modele : {MODEL}\n"
            f"Serveur : {BASE_URL}\n"
            f"Outils sandbox : {', '.join(tools)}\n\n"
            "Les commandes s'executent dans un conteneur Docker Alpine.\n"
            'Essayez : "Liste les fichiers dans /workspace"\n'
            '     ou : "Ecris un hello.py et execute-le"\n'
        ),
    )

    await sandbox.close()
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
