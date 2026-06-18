"""PolarGrid provider — Canadian-hosted chat completions via polargrid-sdk.

Regional edges in Toronto, Vancouver, and Montreal make this provider
useful when data residency on Canadian soil is a requirement. The
chat-completions endpoint exposes an OpenAI-shaped surface, including
tool / function calling as of polargrid-sdk 0.8.4 — ``context.tools``
are forwarded and tool calls are returned both non-streaming and
streaming.
"""

from __future__ import annotations

from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig
from roomkit.providers.polargrid.models import PolarGridRegion

__all__ = ["PolarGridAIProvider", "PolarGridConfig", "PolarGridRegion"]
