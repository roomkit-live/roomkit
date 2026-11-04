"""MiniMax provider: OpenAI-compatible chat that reads its reasoning back."""

from roomkit.providers.minimax.ai import MiniMaxAIProvider
from roomkit.providers.minimax.config import MiniMaxConfig

__all__ = ["MiniMaxAIProvider", "MiniMaxConfig"]
