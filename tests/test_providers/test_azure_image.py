"""Tests for the Azure OpenAI image provider (RFC §25).

Offline: the ``openai`` module is replaced by a stub. Only what Azure changes
over the OpenAI parent is tested here — the client, the provider name, and the
empty catalog. Request building, editing, response mapping and usage
accounting are the parent's, covered by ``test_openai_image.py``.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import ProviderError
from roomkit.providers.azure.config import AzureImageConfig

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG).decode("ascii")


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    pass


def _mock_openai_module() -> MagicMock:
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    mod.APIConnectionError = _FakeAPIConnectionError
    return mod


def _provider(**overrides: Any) -> tuple[Any, MagicMock]:
    """Build the provider against a stubbed SDK, returning it and the module."""
    defaults: dict[str, Any] = {
        "api_key": "azure-key",
        "azure_endpoint": "https://paint.openai.azure.com",
        "model": "img-deploy",
    }
    defaults.update(overrides)
    mod = _mock_openai_module()
    with patch.dict("sys.modules", {"openai": mod}):
        from roomkit.providers.azure.image import AzureImageProvider

        return AzureImageProvider(AzureImageConfig(**defaults)), mod


def _response(count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=PNG_B64, revised_prompt=None) for _ in range(count)],
        output_format="png",
        usage=None,
    )


def test_the_client_is_azure_not_openai() -> None:
    provider, mod = _provider(api_version="2025-04-01-preview")

    mod.AsyncOpenAI.assert_not_called()
    kwargs = mod.AsyncAzureOpenAI.call_args.kwargs
    assert kwargs["api_key"] == "azure-key"
    assert kwargs["azure_endpoint"] == "https://paint.openai.azure.com"
    assert kwargs["api_version"] == "2025-04-01-preview"
    assert provider.model_name == "img-deploy"


async def test_generate_addresses_the_deployment() -> None:
    provider, _ = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    [result] = await provider.generate("an origami fox")

    assert result.decoded() == PNG
    kwargs = provider._client.images.generate.await_args.kwargs
    assert kwargs["model"] == "img-deploy"


async def test_sizes_pass_through_normalized() -> None:
    """A deployment name hides which model's size list applies, so Azure judges."""
    provider, _ = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    await provider.generate("a fox", size="2048X1152")

    assert provider._client.images.generate.await_args.kwargs["size"] == "2048x1152"


async def test_a_malformed_size_is_still_refused_locally() -> None:
    provider, _ = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        await provider.generate("a fox", size="huge")
    provider._client.images.generate.assert_not_awaited()


async def test_errors_name_azure_as_the_provider() -> None:
    provider, _ = _provider()
    provider._client.images.generate = AsyncMock(
        side_effect=_FakeAPIStatusError("boom", status_code=400)
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.provider == "azure"
    assert exc_info.value.status_code == 400


async def test_a_short_response_reports_azure_in_its_message() -> None:
    provider, _ = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response(count=1))

    with pytest.raises(ProviderError, match="azure returned 1 image"):
        await provider.generate("two foxes", n=2)


def test_the_catalog_is_empty_because_deployments_are_user_named() -> None:
    from roomkit.providers.azure.image import AzureImageProvider

    assert AzureImageProvider.available_models() == []


def test_missing_sdk_names_the_extra() -> None:
    with (
        patch.dict("sys.modules", {"openai": None}),
        pytest.raises(ImportError, match=r"roomkit\[azure\]"),
    ):
        from roomkit.providers.azure.image import AzureImageProvider

        AzureImageProvider(
            AzureImageConfig(api_key="k", azure_endpoint="https://x", model="img-deploy")
        )
