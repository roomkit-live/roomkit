"""Ollama provider — native ``/api/chat`` access via ollama-python.

Use this when the OpenAI-compatible shim is too lossy: reasoning
models lose their ``thinking`` stream there, and the ``think``
parameter is silently ignored. For plain non-reasoning local models
either provider works; pick this one when you want to be explicit
about the runtime.
"""

from __future__ import annotations

from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig

__all__ = ["OllamaAIProvider", "OllamaConfig"]
