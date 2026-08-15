"""Qwen provider — Alibaba Model Studio's OpenAI-compatible chat API.

Distinct from the Qwen *voice* providers, which are separate packages talking
to local weights rather than to Model Studio: ``roomkit.voice.tts.qwen3`` and
``roomkit.voice.stt.qwen3``.
"""

from roomkit.providers.qwen.ai import QwenAIProvider
from roomkit.providers.qwen.config import QwenConfig

__all__ = ["QwenAIProvider", "QwenConfig"]
