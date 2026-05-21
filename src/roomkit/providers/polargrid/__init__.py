"""PolarGrid provider — Canadian-hosted chat completions via polargrid-sdk.

Regional edges in Toronto, Vancouver, and Montreal make this provider
useful when data residency on Canadian soil is a requirement. The
chat-completions endpoint exposes an OpenAI-shaped surface but does
**not** support tool / function calling — ``context.tools`` is dropped
with a warning.
"""

from __future__ import annotations

from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig

__all__ = ["PolarGridAIProvider", "PolarGridConfig"]
