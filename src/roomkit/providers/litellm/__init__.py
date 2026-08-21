"""LiteLLM provider — OpenAI-compatible access to a self-hosted LiteLLM proxy.

Talks to the LiteLLM *proxy* (AI gateway) over its OpenAI-compatible API —
virtual keys, budgets, and routing stay on the gateway, and roomkit needs
only the ``openai`` SDK. The ``litellm`` Python package is deliberately not
a dependency: roomkit is already a provider abstraction, and running
LiteLLM's in-process one underneath it would trade the native providers'
fidelity for a second normalisation layer.
"""

from roomkit.providers.litellm.ai import LiteLLMAIProvider
from roomkit.providers.litellm.config import LiteLLMConfig

__all__ = ["LiteLLMAIProvider", "LiteLLMConfig"]
