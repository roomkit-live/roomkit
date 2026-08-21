"""Azure OpenAI image provider — draws via Azure's images endpoint (RFC §25)."""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.azure.config import AzureImageConfig
from roomkit.providers.image.base import parse_size
from roomkit.providers.openai.image import OpenAIImageProvider


class AzureImageProvider(OpenAIImageProvider):
    """Image provider using the images endpoint of an Azure OpenAI resource.

    Subclasses :class:`OpenAIImageProvider` the way the chat providers pair up
    — Azure serves the same ``gpt-image-*`` lineup through the same SDK, so
    request building, the generate/edit split, response mapping and usage
    accounting are all inherited. Three things are genuinely Azure's: the
    client (endpoint, key and API version instead of a bearer token), the
    catalog (none — deployments are user-named), and size validation (passed
    through to the vendor, because a deployment name does not say which
    model's size list applies).
    """

    _config: AzureImageConfig

    def __init__(self, config: AzureImageConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for AzureImageProvider. "
                "Install it with: pip install roomkit[azure]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._api_connection_error = _openai.APIConnectionError
        self._client = _openai.AsyncAzureOpenAI(
            api_key=config.api_key.get_secret_value(),
            azure_endpoint=config.azure_endpoint,
            api_version=config.api_version,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "azure"

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Azure exposes user-named deployments, not a fixed model catalog.

        The reason :meth:`AzureAIProvider.available_models` gives holds here
        too: deployment names are chosen per Azure resource, so there is no
        meaningful offline list. Rates for the underlying ``gpt-image-*``
        models are Azure's own, not OpenAI's, so the OpenAI image catalog is
        deliberately not inherited either.
        """
        return []

    @staticmethod
    def _validated_size(size: str) -> str:
        """Normalize a ``"WIDTHxHEIGHT"`` request and send it as-is.

        The parent refuses sizes off OpenAI's published list so the error can
        name the ones that work. Behind a deployment name that list is not
        knowable offline — ``gpt-image-2`` deployments take near-arbitrary
        geometries the ``gpt-image-1`` series refuses — so the size goes to
        the vendor, whose rejection still raises rather than substituting
        another geometry (RFC §25.2).
        """
        width, height = parse_size(size)
        return f"{width}x{height}"
