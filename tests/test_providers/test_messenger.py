"""Tests for the Facebook Messenger provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from roomkit.models.event import TextContent
from roomkit.providers.messenger import (
    FacebookMessengerProvider,
    MessengerConfig,
    parse_messenger_webhook,
)
from tests.conftest import make_event


def _config(**overrides: Any) -> MessengerConfig:
    defaults: dict[str, Any] = {"page_access_token": "token-123"}
    defaults.update(overrides)
    return MessengerConfig(**defaults)


def _success_response(message_id: str = "mid.123") -> dict[str, Any]:
    return {"recipient_id": "user-1", "message_id": message_id}


def _error_response(
    code: int = 190, message: str = "Invalid OAuth access token"
) -> dict[str, Any]:
    return {"error": {"message": message, "type": "OAuthException", "code": code}}


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, response_data: dict[str, Any], status: int = 200) -> None:
        self._data = response_data
        self._status = status
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self._status, json=self._data, request=request)


class _TimeoutTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")


class TestMessengerConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.api_version == "v21.0"
        assert cfg.timeout == 30.0

    def test_base_url(self) -> None:
        cfg = _config(api_version="v22.0")
        assert cfg.base_url == "https://graph.facebook.com/v22.0/me/messages"


class TestFacebookMessengerProvider:
    @pytest.mark.asyncio
    async def test_send_success(self) -> None:
        transport = _MockTransport(_success_response("mid.42"))
        provider = FacebookMessengerProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello!")
        result = await provider.send(event, to="user-1")

        assert result.success is True
        assert result.provider_message_id == "mid.42"

        req = transport.requests[0]
        body = json.loads(req.content)
        assert body["recipient"]["id"] == "user-1"
        assert body["message"]["text"] == "Hello!"
        assert body["messaging_type"] == "RESPONSE"

    @pytest.mark.asyncio
    async def test_send_passes_access_token(self) -> None:
        transport = _MockTransport(_success_response())
        provider = FacebookMessengerProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="user-1")

        url = str(transport.requests[0].url)
        assert "access_token=token-123" in url

    @pytest.mark.asyncio
    async def test_send_empty_message(self) -> None:
        provider = FacebookMessengerProvider(_config())
        event = make_event(body="")
        result = await provider.send(event, to="user-1")

        assert result.success is False
        assert result.error == "empty_message"

    @pytest.mark.asyncio
    async def test_send_graph_api_error(self) -> None:
        transport = _MockTransport(_error_response(190, "Invalid token"), status=401)
        provider = FacebookMessengerProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="user-1")

        assert result.success is False
        assert result.error == "graph_190"
        assert result.metadata["message"] == "Invalid token"

    @pytest.mark.asyncio
    async def test_send_timeout(self) -> None:
        provider = FacebookMessengerProvider(_config())
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="user-1")

        assert result.success is False
        assert result.error == "timeout"


class TestParseMessengerWebhook:
    def test_parse_text_message(self) -> None:
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "page-1",
                    "time": 1700000000000,
                    "messaging": [
                        {
                            "sender": {"id": "sender-1"},
                            "recipient": {"id": "page-1"},
                            "timestamp": 1700000000000,
                            "message": {
                                "mid": "mid.abc",
                                "text": "Hello from Messenger",
                            },
                        }
                    ],
                }
            ],
        }
        messages = parse_messenger_webhook(payload, channel_id="msg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.channel_id == "msg-main"
        assert msg.sender_id == "sender-1"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello from Messenger"
        assert msg.external_id == "mid.abc"
        assert msg.idempotency_key == "mid.abc"
        assert msg.metadata["recipient_id"] == "page-1"

    def test_parse_skips_delivery_receipts(self) -> None:
        payload = {
            "entry": [
                {
                    "messaging": [
                        {
                            "sender": {"id": "s1"},
                            "recipient": {"id": "p1"},
                            "delivery": {"mids": ["mid.1"]},
                        }
                    ]
                }
            ]
        }
        messages = parse_messenger_webhook(payload, channel_id="msg-main")
        assert messages == []

    def test_parse_multiple_messages(self) -> None:
        payload = {
            "entry": [
                {
                    "messaging": [
                        {
                            "sender": {"id": "s1"},
                            "recipient": {"id": "p1"},
                            "timestamp": 1,
                            "message": {"mid": "m1", "text": "first"},
                        },
                        {
                            "sender": {"id": "s2"},
                            "recipient": {"id": "p1"},
                            "timestamp": 2,
                            "message": {"mid": "m2", "text": "second"},
                        },
                    ]
                }
            ]
        }
        messages = parse_messenger_webhook(payload, channel_id="msg-main")
        assert len(messages) == 2
        assert messages[0].content.body == "first"  # type: ignore[union-attr]
        assert messages[1].content.body == "second"  # type: ignore[union-attr]

    def test_parse_empty_payload(self) -> None:
        messages = parse_messenger_webhook({}, channel_id="msg-main")
        assert messages == []


class TestMessengerSignatureVerification:
    """Tests for FacebookMessengerProvider.verify_signature()."""

    def test_verify_valid_signature(self) -> None:
        import hashlib
        import hmac as hmac_mod

        provider = FacebookMessengerProvider(_config(app_secret="secret123"))
        payload = b'{"entry": []}'
        digest = hmac_mod.new(b"secret123", payload, hashlib.sha256).hexdigest()
        signature = f"sha256={digest}"

        assert provider.verify_signature(payload, signature) is True

    def test_verify_invalid_signature(self) -> None:
        provider = FacebookMessengerProvider(_config(app_secret="secret123"))
        payload = b'{"entry": []}'

        assert provider.verify_signature(payload, "sha256=bad_signature") is False

    def test_verify_no_app_secret(self) -> None:
        provider = FacebookMessengerProvider(_config())

        with pytest.raises(ValueError, match="app_secret must be provided"):
            provider.verify_signature(b"payload", "sha256=anything")

    def test_verify_missing_prefix(self) -> None:
        import hashlib
        import hmac as hmac_mod

        provider = FacebookMessengerProvider(_config(app_secret="secret123"))
        payload = b'{"entry": []}'
        digest = hmac_mod.new(b"secret123", payload, hashlib.sha256).hexdigest()

        # Valid digest but without the "sha256=" prefix
        assert provider.verify_signature(payload, digest) is False
