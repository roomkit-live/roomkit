"""Azure AI Studio provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class AzureAIConfig(BaseModel):
    """Azure AI Studio provider configuration.

    Uses the OpenAI-compatible Chat Completions API exposed by Azure AI Foundry
    deployments (DeepSeek, GPT-4o, Mistral, etc.).

    Attributes:
        api_key: Azure API key for authentication.
        azure_endpoint: Azure AI Foundry project endpoint URL.
        api_version: Azure API version string.
        model: Deployment name (no default — user must specify).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        timeout: HTTP request timeout in seconds.
    """

    api_key: SecretStr
    azure_endpoint: str
    api_version: str = "2024-12-01-preview"
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 120.0
