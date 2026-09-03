"""Every HTTP provider hands its client a timeout that splits connect from read.

A bare float applies the read budget to the TCP connect as well, so a host
that no longer accepts connections is only given up on once the kernel
exhausts its SYN retries (about two minutes), whatever the config says
(RMK-148). Each case builds a provider with distinctive values and reads back
the ``timeout`` its client or SDK constructor actually received, so the test
fails the day an adapter passes the float again.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig, AzureImageConfig
from roomkit.providers.azure.image import AzureImageProvider
from roomkit.providers.deepseek.ai import DeepSeekAIProvider
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.elasticemail.config import ElasticEmailConfig
from roomkit.providers.elasticemail.email import ElasticEmailProvider
from roomkit.providers.http.config import HTTPProviderConfig
from roomkit.providers.http.provider import WebhookHTTPProvider
from roomkit.providers.litellm.ai import LiteLLMAIProvider
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.messenger.config import MessengerConfig
from roomkit.providers.messenger.facebook import FacebookMessengerProvider
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig, OpenAIImageConfig
from roomkit.providers.openai.image import OpenAIImageProvider
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig, OpenRouterImageConfig
from roomkit.providers.openrouter.image import OpenRouterImageProvider
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig
from roomkit.providers.qwen.ai import QwenAIProvider
from roomkit.providers.qwen.config import QwenConfig
from roomkit.providers.sendgrid.config import SendGridConfig
from roomkit.providers.sendgrid.email import SendGridProvider
from roomkit.providers.sinch.config import SinchConfig
from roomkit.providers.sinch.sms import SinchSMSProvider
from roomkit.providers.telegram.api import TelegramBotAPI
from roomkit.providers.telegram.config import TelegramConfig
from roomkit.providers.telnyx.config import TelnyxConfig
from roomkit.providers.telnyx.rcs import TelnyxRCSConfig, TelnyxRCSProvider
from roomkit.providers.telnyx.sms import TelnyxSMSProvider
from roomkit.providers.twilio.config import TwilioConfig
from roomkit.providers.twilio.rcs import TwilioRCSConfig, TwilioRCSProvider
from roomkit.providers.twilio.sms import TwilioSMSProvider
from roomkit.providers.utils import http_timeout
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.providers.voicemeup.config import VoiceMeUpConfig
from roomkit.providers.voicemeup.sms import VoiceMeUpSMSProvider
from roomkit.providers.xai.ai import XAIAIProvider
from roomkit.providers.xai.config import XAIConfig, XAIImageConfig
from roomkit.providers.xai.image import XAIImageProvider

# Distinctive on purpose: equal values would let a swapped connect/read pass.
TIMEOUT = 42.0
CONNECT = 3.0
_TIMEOUTS: dict[str, float] = {"timeout": TIMEOUT, "connect_timeout": CONNECT}
_NUMBER = "+15550001111"

Builder = Callable[[], Awaitable[Any]]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"data": []}


class _RecordingAsyncClient:
    """Stand-in for ``httpx.AsyncClient`` that records its constructor kwargs.

    For the two providers that open a throwaway client inside a method
    (``GET /models``), where nothing holds the client afterwards.
    """

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).calls.append(kwargs)

    async def __aenter__(self) -> _RecordingAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse()


def _sdk_module() -> MagicMock:
    """A module stub whose constructors record what the provider passed."""
    mod = MagicMock()
    mod.PolarGrid.create = AsyncMock(return_value=MagicMock())
    return mod


async def _read_and_close(client: httpx.AsyncClient) -> httpx.Timeout:
    try:
        return client.timeout
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Builders: one per client construction site in ``providers/``
# ---------------------------------------------------------------------------


async def _twilio_sms() -> Any:
    cfg = TwilioConfig(account_sid="AC1", auth_token="t", from_number=_NUMBER, **_TIMEOUTS)
    return await _read_and_close(TwilioSMSProvider(cfg)._client)


async def _twilio_rcs() -> Any:
    cfg = TwilioRCSConfig(
        account_sid="AC1", auth_token="t", messaging_service_sid="MG1", **_TIMEOUTS
    )
    return await _read_and_close(TwilioRCSProvider(cfg)._client)


async def _telnyx_sms() -> Any:
    cfg = TelnyxConfig(api_key="k", from_number=_NUMBER, **_TIMEOUTS)
    return await _read_and_close(TelnyxSMSProvider(cfg)._client)


async def _telnyx_rcs() -> Any:
    cfg = TelnyxRCSConfig(api_key="k", agent_id="agent", **_TIMEOUTS)
    return await _read_and_close(TelnyxRCSProvider(cfg)._client)


async def _sinch_sms() -> Any:
    cfg = SinchConfig(service_plan_id="sp", api_token="t", from_number=_NUMBER, **_TIMEOUTS)
    return await _read_and_close(SinchSMSProvider(cfg)._client)


async def _voicemeup_sms() -> Any:
    cfg = VoiceMeUpConfig(username="u", auth_token="t", from_number=_NUMBER, **_TIMEOUTS)
    return await _read_and_close(VoiceMeUpSMSProvider(cfg)._client)


async def _telegram() -> Any:
    cfg = TelegramConfig(bot_token="t", **_TIMEOUTS)
    return await _read_and_close(TelegramBotAPI(cfg)._client)


async def _messenger() -> Any:
    cfg = MessengerConfig(page_access_token="t", **_TIMEOUTS)
    return await _read_and_close(FacebookMessengerProvider(cfg)._client)


async def _webhook_http() -> Any:
    cfg = HTTPProviderConfig(webhook_url="https://example.com/hook", **_TIMEOUTS)
    return await _read_and_close(WebhookHTTPProvider(cfg)._client)


async def _sendgrid() -> Any:
    cfg = SendGridConfig(api_key="k", from_email="a@example.com", **_TIMEOUTS)
    return await _read_and_close(SendGridProvider(cfg)._client)


async def _elasticemail() -> Any:
    cfg = ElasticEmailConfig(api_key="k", from_email="a@example.com", **_TIMEOUTS)
    return await _read_and_close(ElasticEmailProvider(cfg)._client)


async def _openrouter_image() -> Any:
    cfg = OpenRouterImageConfig(api_key="k", model="m", **_TIMEOUTS)
    return await _read_and_close(OpenRouterImageProvider(cfg)._http)


async def _openai() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        OpenAIAIProvider(OpenAIConfig(api_key="k", model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _azure() -> Any:
    mod = _sdk_module()
    cfg = AzureAIConfig(
        api_key="k", azure_endpoint="https://p.services.ai.azure.com", model="m", **_TIMEOUTS
    )
    with patch.dict("sys.modules", {"openai": mod}):
        AzureAIProvider(cfg)
    return mod.AsyncAzureOpenAI.call_args.kwargs["timeout"]


async def _openrouter() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        OpenRouterAIProvider(OpenRouterConfig(api_key="k", model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _litellm() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        LiteLLMAIProvider(LiteLLMConfig(api_key="k", model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _xai() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        XAIAIProvider(XAIConfig(api_key="k", model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _deepseek() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        DeepSeekAIProvider(DeepSeekConfig(api_key="k", model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _qwen() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        QwenAIProvider(QwenConfig(api_key="k", model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _vllm() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        create_vllm_provider(VLLMConfig(model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _ollama() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"ollama": mod}):
        OllamaAIProvider(OllamaConfig(model="m", **_TIMEOUTS))
    return mod.AsyncClient.call_args.kwargs["timeout"]


async def _anthropic() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"anthropic": mod}):
        AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-opus-5", **_TIMEOUTS))
    return mod.AsyncAnthropic.call_args.kwargs["timeout"]


async def _polargrid_pinned() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"polargrid": mod}):
        provider = PolarGridAIProvider(
            PolarGridConfig(api_key="pg_k", region="toronto", **_TIMEOUTS)
        )
    await provider._ensure_client()
    return mod.PolarGrid.call_args.kwargs["timeout"]


async def _polargrid_autorouted() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"polargrid": mod}):
        provider = PolarGridAIProvider(PolarGridConfig(api_key="pg_k", **_TIMEOUTS))
    await provider._ensure_client()
    return mod.PolarGrid.create.await_args.kwargs["timeout"]


async def _openai_image() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        OpenAIImageProvider(OpenAIImageConfig(api_key="k", model="m", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _azure_image() -> Any:
    mod = _sdk_module()
    cfg = AzureImageConfig(
        api_key="k", azure_endpoint="https://paint.openai.azure.com", model="m", **_TIMEOUTS
    )
    with patch.dict("sys.modules", {"openai": mod}):
        AzureImageProvider(cfg)
    return mod.AsyncAzureOpenAI.call_args.kwargs["timeout"]


async def _xai_image() -> Any:
    mod = _sdk_module()
    with patch.dict("sys.modules", {"openai": mod}):
        XAIImageProvider(XAIImageConfig(api_key="k", **_TIMEOUTS))
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _openrouter_models_fetch() -> Any:
    with patch.dict("sys.modules", {"openai": _sdk_module()}):
        provider = OpenRouterAIProvider(OpenRouterConfig(api_key="k", model="m", **_TIMEOUTS))
    _RecordingAsyncClient.calls.clear()
    with patch("httpx.AsyncClient", _RecordingAsyncClient):
        await provider._fetch_models_json()
    return _RecordingAsyncClient.calls[-1]["timeout"]


async def _litellm_model_info_fetch() -> Any:
    with patch.dict("sys.modules", {"openai": _sdk_module()}):
        provider = LiteLLMAIProvider(LiteLLMConfig(api_key="k", model="m", **_TIMEOUTS))
    _RecordingAsyncClient.calls.clear()
    with patch("httpx.AsyncClient", _RecordingAsyncClient):
        await provider._fetch_model_info()
    return _RecordingAsyncClient.calls[-1]["timeout"]


CASES: dict[str, Builder] = {
    "twilio-sms": _twilio_sms,
    "twilio-rcs": _twilio_rcs,
    "telnyx-sms": _telnyx_sms,
    "telnyx-rcs": _telnyx_rcs,
    "sinch-sms": _sinch_sms,
    "voicemeup-sms": _voicemeup_sms,
    "telegram": _telegram,
    "messenger": _messenger,
    "webhook-http": _webhook_http,
    "sendgrid": _sendgrid,
    "elasticemail": _elasticemail,
    "openrouter-image": _openrouter_image,
    "openai": _openai,
    "azure": _azure,
    "openrouter": _openrouter,
    "litellm": _litellm,
    "xai": _xai,
    "deepseek": _deepseek,
    "qwen": _qwen,
    "vllm": _vllm,
    "ollama": _ollama,
    "anthropic": _anthropic,
    "polargrid-pinned": _polargrid_pinned,
    "polargrid-autorouted": _polargrid_autorouted,
    "openai-image": _openai_image,
    "azure-image": _azure_image,
    "xai-image": _xai_image,
    "openrouter-models-fetch": _openrouter_models_fetch,
    "litellm-model-info-fetch": _litellm_model_info_fetch,
}


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", list(CASES.values()), ids=list(CASES))
async def test_client_timeout_splits_connect_from_read(build: Builder) -> None:
    timeout = await build()

    assert isinstance(timeout, httpx.Timeout), f"a bare {type(timeout).__name__} was passed"
    assert timeout.connect == CONNECT
    assert timeout.read == TIMEOUT
    assert timeout.write == TIMEOUT
    assert timeout.pool == TIMEOUT


def test_http_timeout_keeps_timeout_as_the_read_budget() -> None:
    timeout = http_timeout(OpenAIConfig(api_key="k", model="m", **_TIMEOUTS))

    assert timeout == httpx.Timeout(TIMEOUT, connect=CONNECT)


# ---------------------------------------------------------------------------
# The default: the SDKs' own 5 s, on every config
# ---------------------------------------------------------------------------

CONFIGS: list[tuple[type[Any], dict[str, Any]]] = [
    (OpenAIConfig, {"api_key": "k", "model": "m"}),
    (OpenAIImageConfig, {"api_key": "k", "model": "m"}),
    (
        AzureAIConfig,
        {"api_key": "k", "azure_endpoint": "https://p.services.ai.azure.com", "model": "m"},
    ),
    (
        AzureImageConfig,
        {"api_key": "k", "azure_endpoint": "https://paint.openai.azure.com", "model": "m"},
    ),
    (XAIImageConfig, {"api_key": "k"}),
    (OllamaConfig, {"model": "m"}),
    (VLLMConfig, {"model": "m"}),
    (PolarGridConfig, {"api_key": "pg_k"}),
    (AnthropicConfig, {"api_key": "k", "model": "claude-opus-5"}),
    (TelnyxConfig, {"api_key": "k", "from_number": _NUMBER}),
    (TelnyxRCSConfig, {"api_key": "k", "agent_id": "agent"}),
    (TwilioConfig, {"account_sid": "AC1", "auth_token": "t", "from_number": _NUMBER}),
    (TwilioRCSConfig, {"account_sid": "AC1", "auth_token": "t", "messaging_service_sid": "MG1"}),
    (SinchConfig, {"service_plan_id": "sp", "api_token": "t", "from_number": _NUMBER}),
    (VoiceMeUpConfig, {"username": "u", "auth_token": "t", "from_number": _NUMBER}),
    (TelegramConfig, {"bot_token": "t"}),
    (MessengerConfig, {"page_access_token": "t"}),
    (HTTPProviderConfig, {"webhook_url": "https://example.com/hook"}),
    (SendGridConfig, {"api_key": "k", "from_email": "a@example.com"}),
    (ElasticEmailConfig, {"api_key": "k", "from_email": "a@example.com"}),
]


@pytest.mark.parametrize(("config_cls", "kwargs"), CONFIGS, ids=[c.__name__ for c, _ in CONFIGS])
def test_connect_timeout_defaults_to_five_seconds(
    config_cls: type[Any], kwargs: dict[str, Any]
) -> None:
    config = config_cls(**kwargs)

    assert config.connect_timeout == 5.0
    assert config.timeout > config.connect_timeout
