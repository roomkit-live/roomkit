"""Curated catalog of popular Ollama models.

Hand-maintained, offline list returned by ``OllamaAIProvider.available_models``.
Ollama is an open-weights registry where models are pulled locally by name, so
this is a *popularity* snapshot of the public library (ollama.com/library), not
an account-specific set. For the models actually installed on a given server,
call ``OllamaAIProvider.list_models()`` — it queries the local ``/api/tags``.

Context windows are the documented per-model defaults; larger size variants of
the same family often support more. Ids use the canonical base pull name.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(id="llama3.1", display_name="Llama 3.1", context_window=128_000),
    ModelInfo(id="deepseek-r1", display_name="DeepSeek-R1", context_window=128_000),
    ModelInfo(id="llama3.2", display_name="Llama 3.2", context_window=128_000),
    ModelInfo(id="gemma3", display_name="Gemma 3", context_window=128_000, supports_vision=True),
    ModelInfo(id="qwen2.5", display_name="Qwen2.5", context_window=128_000),
    ModelInfo(id="qwen3", display_name="Qwen3", context_window=40_000),
    ModelInfo(id="mistral", display_name="Mistral 7B", context_window=32_000),
    ModelInfo(id="gemma2", display_name="Gemma 2", context_window=8_000),
    ModelInfo(id="llama3", display_name="Llama 3", context_window=8_000),
    ModelInfo(id="phi3", display_name="Phi-3", context_window=128_000),
    ModelInfo(id="qwen2.5-coder", display_name="Qwen2.5 Coder", context_window=32_000),
    ModelInfo(id="llava", display_name="LLaVA", context_window=32_000, supports_vision=True),
    ModelInfo(id="gemma4", display_name="Gemma 4", context_window=128_000, supports_vision=True),
    ModelInfo(id="qwen3.5", display_name="Qwen3.5", context_window=256_000, supports_vision=True),
    ModelInfo(id="gpt-oss", display_name="gpt-oss", context_window=128_000),
    ModelInfo(id="phi4", display_name="Phi-4", context_window=16_000),
    ModelInfo(id="qwen3-coder", display_name="Qwen3 Coder", context_window=256_000),
    ModelInfo(
        id="qwen3-vl", display_name="Qwen3-VL", context_window=256_000, supports_vision=True
    ),
    ModelInfo(
        id="llama3.2-vision",
        display_name="Llama 3.2 Vision",
        context_window=128_000,
        supports_vision=True,
    ),
    ModelInfo(id="llama3.3", display_name="Llama 3.3", context_window=128_000),
]
