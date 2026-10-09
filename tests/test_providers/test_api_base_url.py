"""A provider's API host is configurable, and its credentials never travel in clear."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from roomkit.providers.telegram import TelegramBotProvider, TelegramConfig
from roomkit.providers.twilio import TwilioConfig, TwilioRCSConfig, TwilioSMSProvider
from roomkit.providers.url_safety import validate_api_base_url
from tests.conftest import make_event

TWILIO = {"account_sid": "AC1", "auth_token": "secret", "from_number": "+15145551234"}


@pytest.mark.parametrize(
    "url",
    [
        "https://sandbox.example.com",
        "http://127.0.0.1:8401",
        "http://localhost:8401/",
        "http://[::1]:8401",
        "http://127.1:8401",
    ],
)
def test_https_or_this_machine_is_accepted(url: str) -> None:
    assert validate_api_base_url(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com",
        "http://10.0.0.5:8401",
        "https://user:pass@api.example.com",
        "api.twilio.com",
        "ftp://127.0.0.1",
    ],
    ids=["http-remote", "http-private", "credentials", "no-scheme", "other-scheme"],
)
def test_anything_else_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="API base URL"):
        validate_api_base_url(url)


def test_the_vendors_hosts_stay_the_default() -> None:
    assert TwilioConfig(**TWILIO).api_url == (
        "https://api.twilio.com/2010-04-01/Accounts/AC1/Messages.json"
    )
    rcs = TwilioRCSConfig(account_sid="AC1", auth_token="secret", messaging_service_sid="MG1")
    assert rcs.api_url.startswith("https://api.twilio.com/2010-04-01/")
    telegram = TelegramConfig(bot_token="111:AAA")
    assert telegram.base_url == "https://api.telegram.org/bot111:AAA"
    assert telegram.file_base_url == "https://api.telegram.org/file/bot111:AAA"


def test_a_config_refuses_a_host_its_token_would_reach_in_clear() -> None:
    with pytest.raises(ValidationError):
        TwilioConfig(**TWILIO, api_base_url="http://twilio.example.com")
    with pytest.raises(ValidationError):
        TelegramConfig(bot_token="111:AAA", api_base_url="http://bots.example.com")


async def test_twilio_sends_to_the_configured_host() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"sid": "SM1"})

    provider = TwilioSMSProvider(TwilioConfig(**TWILIO, api_base_url="http://127.0.0.1:8401/"))
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))  # noqa: SLF001

    result = await provider.send(make_event(body="Hi"), to="+15550000001")

    assert result.success
    assert str(seen[0].url) == "http://127.0.0.1:8401/2010-04-01/Accounts/AC1/Messages.json"
    await provider.close()


async def test_telegram_sends_to_the_configured_server() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body: dict[str, Any] = {"ok": True, "result": {"message_id": 7, "chat": {"id": 1}}}
        return httpx.Response(200, json=body)

    config = TelegramConfig(bot_token="111:AAA", api_base_url="https://bots.example.com")
    provider = TelegramBotProvider(config)
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))  # noqa: SLF001

    result = await provider.send(make_event(body="Hi"), to="1")

    assert result.success
    assert str(seen[0].url).startswith("https://bots.example.com/bot111:AAA/")
    await provider.close()
