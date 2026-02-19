"""Azure AI Studio provider — generates responses via Azure's OpenAI-compatible API."""

from __future__ import annotations

from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.openai.ai import OpenAIAIProvider


class AzureAIProvider(OpenAIAIProvider):
    """AI provider using Azure AI Studio's OpenAI-compatible Chat Completions API.

    Subclasses :class:`OpenAIAIProvider` — only client initialisation and provider
    name differ.  All message building, tool handling, response parsing, and
    streaming are inherited.
    """

    _config: AzureAIConfig  # type: ignore[assignment]

    def __init__(self, config: AzureAIConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for AzureAIProvider. "
                "Install it with: pip install roomkit[azure]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._client = _openai.AsyncAzureOpenAI(
            api_key=config.api_key.get_secret_value(),
            azure_endpoint=config.azure_endpoint,
            api_version=config.api_version,
            timeout=config.timeout,
        )

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "azure"
