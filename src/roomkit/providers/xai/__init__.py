"""xAI (Grok) provider — OpenAI-compatible chat, plus realtime speech-to-speech.

``XAIRealtimeProvider`` is deliberately not re-exported here: it needs the
``websockets`` package, so it stays behind an explicit
``from roomkit.providers.xai.realtime import XAIRealtimeProvider`` (or
``roomkit.voice.get_xai_realtime_provider()``). The chat provider only needs the
lazily-imported ``openai`` SDK, so it is safe to export.
"""

from roomkit.providers.xai.ai import XAIAIProvider
from roomkit.providers.xai.config import XAIConfig, XAIRealtimeConfig

__all__ = ["XAIAIProvider", "XAIConfig", "XAIRealtimeConfig"]
