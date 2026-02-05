"""Tests for the Sinch SMS provider."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any

import httpx
import pytest

from roomkit.models.event import MediaContent, TextContent
from roomkit.providers.sinch import (
    SinchConfig,
    SinchSMSProvider,
    parse_sinch_webhook,
)
from tests.conftest import make_event, make_media_event


def _config(**overrides: Any) -> SinchConfig:
    defaults: dict[str, Any] = {
        "service_plan_id": "plan123",
        "api_token": "token123",
        "from_number": "+15145551234",
    }
    defaults.update(overrides)
    return SinchConfig(**defaults)


def _success_response(batch_id: str = "batch-123") -> dict[str, Any]:
    return {
        "id": batch_id,
        "to": ["+15145559999"],
        "from": "+15145551234",
        "canceled": False,
        "body": "Hello",
    }


class _MockTransport(httpx.AsyncBaseTransport):
    """Captures requests and returns a canned JSON response."""

    def __init__(
        self,
        response_data: dict[str, Any],
        status_code: int = 200,
    ) -> None:
        self._response_data = response_data
        self._status_code = status_code
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status_code,
            json=self._response_data,
            request=request,
        )


class _TimeoutTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSinchConfig:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.timeout == 10.0
        assert cfg.region == "us"
        assert cfg.webhook_secret is None

    def test_api_url(self) -> None:
        cfg = _config()
        assert cfg.api_url == "https://us.sms.api.sinch.com/xms/v1/plan123/batches"

    def test_api_url_eu_region(self) -> None:
        cfg = _config(region="eu")
        assert cfg.api_url == "https://eu.sms.api.sinch.com/xms/v1/plan123/batches"

    def test_config_with_webhook_secret(self) -> None:
        cfg = _config(webhook_secret="secret123")
        assert cfg.webhook_secret is not None


class TestSinchSMSProvider:
    @pytest.mark.asyncio
    async def test_send_sms_success(self) -> None:
        transport = _MockTransport(_success_response("batch-42"))
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello from RoomKit")
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        assert result.provider_message_id == "batch-42"

        req = transport.requests[0]
        assert "/batches" in str(req.url)
        assert req.headers["authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_send_sms_to_array(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="+15145559999")

        req = transport.requests[0]
        body = req.read().decode()
        # Sinch expects 'to' as an array
        assert '["' in body or '["' in body

    @pytest.mark.asyncio
    async def test_send_sms_empty_message(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "empty_message"
        assert len(transport.requests) == 0

    @pytest.mark.asyncio
    async def test_send_sms_auth_error(self) -> None:
        transport = _MockTransport({"message": "Unauthorized"}, status_code=401)
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "auth_error"

    @pytest.mark.asyncio
    async def test_send_sms_rate_limit(self) -> None:
        transport = _MockTransport({"message": "Rate limit"}, status_code=429)
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "rate_limit"

    @pytest.mark.asyncio
    async def test_send_sms_invalid_request(self) -> None:
        transport = _MockTransport(
            {"code": "invalid_parameter", "text": "Invalid phone number"},
            status_code=400,
        )
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="invalid")

        assert result.success is False
        assert result.error == "sinch_invalid_parameter"

    @pytest.mark.asyncio
    async def test_send_sms_timeout(self) -> None:
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_from_number_override(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="+15145559999", from_="+15140001111")

        req = transport.requests[0]
        body = req.read().decode()
        assert "+15140001111" in body

    @pytest.mark.asyncio
    async def test_send_mms_mt_media(self) -> None:
        transport = _MockTransport(_success_response("batch-mms"))
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
            caption="Check this!",
        )
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        req = transport.requests[0]
        body = req.read().decode()
        assert '"mt_media"' in body
        assert '"url"' in body
        assert "https://example.com/image.jpg" in body
        assert '"message"' in body
        assert "Check this!" in body

    @pytest.mark.asyncio
    async def test_send_mms_media_only(self) -> None:
        transport = _MockTransport(_success_response("batch-mms2"))
        cfg = _config()
        provider = SinchSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
        )
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        req = transport.requests[0]
        body = req.read().decode()
        assert '"mt_media"' in body
        # No message key when no caption
        assert '"message"' not in body


class TestSinchSignatureVerification:
    def test_verify_valid_signature(self) -> None:
        cfg = _config(webhook_secret="secret123")
        provider = SinchSMSProvider(cfg)

        payload = b'{"id": "msg-1", "from": "+15551234567", "body": "Hello"}'
        expected_sig = base64.b64encode(
            hmac.new(b"secret123", payload, hashlib.sha1).digest()
        ).decode()

        assert provider.verify_signature(payload, expected_sig) is True

    def test_verify_invalid_signature(self) -> None:
        cfg = _config(webhook_secret="secret123")
        provider = SinchSMSProvider(cfg)

        payload = b'{"id": "msg-1", "body": "Hello"}'
        assert provider.verify_signature(payload, "invalid") is False

    def test_verify_no_webhook_secret(self) -> None:
        cfg = _config()  # No webhook_secret
        provider = SinchSMSProvider(cfg)

        with pytest.raises(ValueError, match="webhook_secret must be provided"):
            provider.verify_signature(b"payload", "signature")


class TestParseSinchWebhook:
    def test_parse_webhook_inbound(self) -> None:
        payload = {
            "id": "msg-uuid-abc",
            "from": "+15145551111",
            "to": "12345",
            "body": "Hello from user",
            "received_at": "2026-01-28T12:00:00.000Z",
            "operator_id": "op-123",
            "client_reference": "ref-456",
        }
        msg = parse_sinch_webhook(payload, channel_id="sms-main")

        assert msg.channel_id == "sms-main"
        assert msg.sender_id == "+15145551111"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello from user"
        assert msg.external_id == "msg-uuid-abc"
        assert msg.idempotency_key == "msg-uuid-abc"
        assert msg.metadata["to"] == "12345"
        assert msg.metadata["received_at"] == "2026-01-28T12:00:00.000Z"
        assert msg.metadata["operator_id"] == "op-123"

    def test_parse_webhook_minimal(self) -> None:
        payload = {
            "from": "+15145551111",
            "body": "Hi",
        }
        msg = parse_sinch_webhook(payload, channel_id="sms-main")

        assert msg.sender_id == "+15145551111"
        assert msg.content.body == "Hi"
        assert msg.external_id is None

    def test_parse_webhook_with_media(self) -> None:
        payload = {
            "id": "msg-mms",
            "from": "+15145551111",
            "to": "12345",
            "body": "Look at this",
            "media": [
                {"url": "https://sinch.com/img.jpg", "mimeType": "image/jpeg"},
            ],
        }
        msg = parse_sinch_webhook(payload, channel_id="sms-main")

        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://sinch.com/img.jpg"
        assert msg.content.caption == "Look at this"

    def test_parse_webhook_media_only(self) -> None:
        payload = {
            "id": "msg-mms2",
            "from": "+15145551111",
            "to": "12345",
            "body": "",
            "media": [
                {"url": "https://sinch.com/img.jpg", "mimeType": "image/jpeg"},
            ],
        }
        msg = parse_sinch_webhook(payload, channel_id="sms-main")

        assert isinstance(msg.content, MediaContent)
        assert msg.content.caption is None
