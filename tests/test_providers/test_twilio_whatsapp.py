"""Tests for the Twilio WhatsApp provider: the Messages API with ``whatsapp:`` addresses."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import parse_qs

import httpx

from roomkit import HookTrigger, RoomKit
from roomkit.channels import WhatsAppChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.twilio import (
    TwilioConfig,
    TwilioWhatsAppProvider,
    parse_twilio_whatsapp_webhook,
)
from tests.conftest import make_event

ALICE = "+15550000001"


def _provider(requests: list[httpx.Request], status_code: int = 201, **config: Any):
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status_code >= 400:
            return httpx.Response(status_code, json={"code": 63016, "message": "Outside window"})
        return httpx.Response(status_code, json={"sid": "SM-wa-1", "status": "queued"})

    settings: dict[str, Any] = {
        "account_sid": "ACtest123",
        "auth_token": "secret123",
        "from_number": "+15145551234",
        **config,
    }
    provider = TwilioWhatsAppProvider(TwilioConfig(**settings))
    provider._messages._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))  # noqa: SLF001
    return provider


def _form(request: httpx.Request) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


async def test_both_addresses_go_out_as_whatsapp() -> None:
    requests: list[httpx.Request] = []
    provider = _provider(requests)

    result = await provider.send(make_event(body="Hello"), to=ALICE)

    assert result.success and result.provider_message_id == "SM-wa-1"
    form = _form(requests[0])
    assert form["To"] == f"whatsapp:{ALICE}"
    assert form["From"] == "whatsapp:+15145551234"
    assert form["Body"] == "Hello"
    assert str(requests[0].url).endswith("/Accounts/ACtest123/Messages.json")
    await provider.close()


async def test_an_address_already_written_for_whatsapp_keeps_one_prefix() -> None:
    requests: list[httpx.Request] = []
    provider = _provider(requests, from_number="whatsapp:+15145551234")

    await provider.send(make_event(body="Hello"), to=f"whatsapp:{ALICE}")

    form = _form(requests[0])
    assert form["To"] == f"whatsapp:{ALICE}"
    assert form["From"] == "whatsapp:+15145551234"
    await provider.close()


async def test_a_refusal_comes_back_as_a_failed_result() -> None:
    """Outside the 24-hour window Twilio refuses a free-form message."""
    provider = _provider([], status_code=400)

    result = await provider.send(make_event(body="Hello"), to=ALICE)

    assert not result.success
    await provider.close()


async def test_a_customer_writing_on_whatsapp_is_answered_on_whatsapp() -> None:
    requests: list[httpx.Request] = []
    provider = _provider(requests)
    kit = RoomKit()
    kit.register_channel(WhatsAppChannel("wa", provider=provider))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["Hi Alice"])))

    @kit.hook(HookTrigger.ON_ROOM_CREATED)
    async def agent_joins(event: Any, ctx: Any) -> None:
        await kit.attach_channel(ctx.room.id, "ai")

    payload = {
        "MessageSid": "SM-in-1",
        "From": f"whatsapp:{ALICE}",
        "To": "whatsapp:+15145551234",
        "Body": "Hello",
        "NumMedia": "0",
    }
    result = await kit.process_inbound(parse_twilio_whatsapp_webhook(payload, channel_id="wa"))
    assert result.event is not None
    for _ in range(200):
        if requests:
            break
        await asyncio.sleep(0.01)

    assert result.event.source.participant_id == ALICE
    assert _form(requests[0])["To"] == f"whatsapp:{ALICE}"
    assert _form(requests[0])["Body"] == "Hi Alice"
    await kit.close()
