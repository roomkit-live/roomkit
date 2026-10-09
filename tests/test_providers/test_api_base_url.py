"""A provider's API host is configurable, and its credentials never travel in clear."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from roomkit.providers.elasticemail import ElasticEmailConfig
from roomkit.providers.messenger import MessengerConfig
from roomkit.providers.sendgrid import SendGridConfig
from roomkit.providers.sinch import SinchConfig
from roomkit.providers.telegram import TelegramBotProvider, TelegramConfig
from roomkit.providers.telnyx import TelnyxConfig, TelnyxRCSConfig, TelnyxSMSProvider
from roomkit.providers.twilio import TwilioConfig, TwilioRCSConfig, TwilioSMSProvider
from roomkit.providers.url_safety import validate_api_base_url
from roomkit.providers.voicemeup import VoiceMeUpConfig
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


def test_every_messaging_provider_keeps_its_vendors_host() -> None:
    assert MessengerConfig(page_access_token="t", api_version="v21.0").base_url == (
        "https://graph.facebook.com/v21.0/me/messages"
    )
    sinch = SinchConfig(service_plan_id="P1", api_token="t", from_number="+1", region="eu")
    assert sinch.api_url == "https://eu.sms.api.sinch.com/xms/v1/P1/batches"
    voicemeup = VoiceMeUpConfig(username="u", auth_token="t", from_number="+1")
    assert voicemeup.base_url == "https://clients.voicemeup.com/api/v1.1/json/"
    sandbox = VoiceMeUpConfig(
        username="u", auth_token="t", from_number="+1", environment="sandbox"
    )
    assert sandbox.base_url == "https://dev-clients.voicemeup.com/api/v1.1/json/"
    assert TelnyxConfig(api_key="k", from_number="+1").api_base_url == "https://api.telnyx.com"
    assert TelnyxRCSConfig(api_key="k", agent_id="a").api_base_url == "https://api.telnyx.com"


def test_every_messaging_provider_takes_another_host() -> None:
    local = "http://127.0.0.1:8401"
    assert MessengerConfig(page_access_token="t", api_base_url=local).base_url.startswith(
        f"{local}/v"
    )
    sinch = SinchConfig(service_plan_id="P1", api_token="t", from_number="+1", api_base_url=local)
    assert sinch.api_url == f"{local}/xms/v1/P1/batches"
    voicemeup = VoiceMeUpConfig(username="u", auth_token="t", from_number="+1", api_base_url=local)
    assert voicemeup.base_url == f"{local}/api/v1.1/json/"
    assert SendGridConfig(
        api_key="k", from_email="a@x.test", base_url=f"{local}/send"
    ).base_url == (f"{local}/send")
    assert ElasticEmailConfig(api_key="k", from_email="a@x.test", base_url=local).base_url == local


@pytest.mark.parametrize(
    "build",
    [
        lambda url: MessengerConfig(page_access_token="t", api_base_url=url),
        lambda url: SinchConfig(
            service_plan_id="P", api_token="t", from_number="+1", api_base_url=url
        ),
        lambda url: VoiceMeUpConfig(
            username="u", auth_token="t", from_number="+1", api_base_url=url
        ),
        lambda url: TelnyxConfig(api_key="k", from_number="+1", api_base_url=url),
        lambda url: TelnyxRCSConfig(api_key="k", agent_id="a", api_base_url=url),
        lambda url: SendGridConfig(api_key="k", from_email="a@x.test", base_url=url),
        lambda url: ElasticEmailConfig(api_key="k", from_email="a@x.test", base_url=url),
    ],
    ids=["messenger", "sinch", "voicemeup", "telnyx", "telnyx-rcs", "sendgrid", "elasticemail"],
)
def test_no_messaging_provider_sends_its_credentials_in_clear(build: Any) -> None:
    with pytest.raises(ValidationError):
        build("http://api.example.com")


async def test_telnyx_sends_to_the_configured_host() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": {"id": "msg-1"}})

    provider = TelnyxSMSProvider(
        TelnyxConfig(api_key="k", from_number="+15145551234", api_base_url="http://127.0.0.1:8405")
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))  # noqa: SLF001

    result = await provider.send(make_event(body="Hi"), to="+15550000001")

    assert result.success
    assert str(seen[0].url) == "http://127.0.0.1:8405/v2/messages"
    await provider.close()
