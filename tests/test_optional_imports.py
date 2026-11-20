"""``import roomkit`` needs only the base dependencies.

An optional SDK is imported where it is used, never at module level on a path
the package root reaches. The root imports the video vision providers, which
import the Gemini client helper: one ``import httpx`` at the top of that
module broke ``import roomkit`` for every install without the ``httpx`` extra.
The openai SDK 3.x runs on httpx2 and installs no ``httpx`` either, so the
clients RoomKit builds on it must not need one.
Nor must the helpers every example imports (``examples/shared``).
The checks run in a fresh interpreter, because this one has httpx loaded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_roomkit_imports_without_httpx() -> None:
    code = "import sys; sys.modules['httpx'] = None; import roomkit; print(roomkit.__version__)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr[-1200:]
    assert proc.stdout.strip(), "the interpreter printed no version"


def test_the_examples_helpers_import_without_httpx() -> None:
    """Every example starts with ``from shared import ...``; one run in an
    environment holding ``roomkit[openai]`` alone has no httpx."""
    code = "import sys; sys.modules['httpx'] = None; import shared; print(shared.setup_logging)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=Path(__file__).parents[1] / "examples",
    )
    assert proc.returncode == 0, proc.stderr[-1200:]


_CLIENTS_ON_THE_OPENAI_SDK = """
import sys
sys.modules['httpx'] = None

from roomkit.classifiers.openai import OpenAIClassifier
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig, AzureImageConfig
from roomkit.providers.azure.image import AzureImageProvider
from roomkit.providers.deepseek import DeepSeekAIProvider, DeepSeekConfig
from roomkit.providers.meta.config import MetaImageConfig
from roomkit.providers.meta.image import MetaImageProvider
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig, OpenAIImageConfig
from roomkit.providers.openai.image import OpenAIImageProvider
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.providers.xai.config import XAIImageConfig
from roomkit.providers.xai.image import XAIImageProvider
from roomkit.video.vision.openai import OpenAIVisionProvider

azure = "https://resource.openai.azure.com"
clients = [
    OpenAIAIProvider(OpenAIConfig(api_key="k", model="m"))._client,
    AzureAIProvider(AzureAIConfig(api_key="k", azure_endpoint=azure, model="m"))._client,
    OpenRouterAIProvider(OpenRouterConfig(api_key="k", model="m"))._client,
    create_vllm_provider(VLLMConfig(model="m"))._client,
    DeepSeekAIProvider(DeepSeekConfig(api_key="k", model="m"))._client,
    OpenAIImageProvider(OpenAIImageConfig(api_key="k", model="m"))._client,
    AzureImageProvider(AzureImageConfig(api_key="k", azure_endpoint=azure, model="m"))._client,
    XAIImageProvider(XAIImageConfig(api_key="k"))._client,
    MetaImageProvider(MetaImageConfig(api_key="k"))._client,
    OpenAIVisionProvider()._get_client(),
    OpenAIClassifier(api_key="k")._client,
]
print(len(clients), sorted({type(client.timeout).__name__ for client in clients}))
"""


def test_the_clients_on_the_openai_sdk_build_without_httpx() -> None:
    """A host that installed ``roomkit[openai]`` alone has httpx2 and no
    httpx: each client RoomKit builds on the SDK is built there, its
    connect/read timeout included (the classifier's is one number)."""
    proc = subprocess.run(
        [sys.executable, "-c", _CLIENTS_ON_THE_OPENAI_SDK],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1200:]
    assert proc.stdout.strip() == "11 ['Timeout', 'float']"
