"""Tests for the Twilio SMS provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from roomkit.models.event import CompositeContent, MediaContent, TextContent
from roomkit.providers.twilio import (
    TwilioConfig,
    TwilioSMSProvider,
    parse_twilio_webhook,
)
from tests.conftest import make_event, make_media_event


def _config(**overrides: Any) -> TwilioConfig:
    defaults: dict[str, Any] = {
        "account_sid": "ACtest123",
        "auth_token": "secret123",
        "from_number": "+15145551234",
    }
    defaults.update(overrides)
    return TwilioConfig(**defaults)


def _success_response(sid: str = "SM123456") -> dict[str, Any]:
    return {
        "sid": sid,
        "status": "queued",
        "date_created": "2026-01-28T12:00:00Z",
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


class TestTwilioConfig:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.timeout == 10.0
        assert cfg.messaging_service_sid is None

    def test_api_url(self) -> None:
        cfg = _config()
        assert cfg.api_url == "https://api.twilio.com/2010-04-01/Accounts/ACtest123/Messages.json"

    def test_config_with_messaging_service(self) -> None:
        cfg = _config(messaging_service_sid="MG123")
        assert cfg.messaging_service_sid == "MG123"


class TestTwilioSMSProvider:
    @pytest.mark.asyncio
    async def test_send_sms_success(self) -> None:
        transport = _MockTransport(_success_response("SM-42"))
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello from RoomKit")
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        assert result.provider_message_id == "SM-42"

        req = transport.requests[0]
        assert "Messages.json" in str(req.url)
        # Check basic auth is used
        assert req.headers.get("authorization") is not None

    @pytest.mark.asyncio
    async def test_send_sms_with_messaging_service(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config(messaging_service_sid="MG123")
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="+15145559999")

        req = transport.requests[0]
        body = req.read().decode()
        assert "MessagingServiceSid=MG123" in body
        # Should NOT have From when using MessagingServiceSid
        assert "From=" not in body

    @pytest.mark.asyncio
    async def test_send_sms_empty_message(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
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
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "auth_error"

    @pytest.mark.asyncio
    async def test_send_sms_rate_limit(self) -> None:
        transport = _MockTransport({"message": "Rate limit"}, status_code=429)
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "rate_limit"

    @pytest.mark.asyncio
    async def test_send_sms_invalid_number(self) -> None:
        transport = _MockTransport(
            {"code": 21211, "message": "Invalid phone number"},
            status_code=400,
        )
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="invalid")

        assert result.success is False
        assert result.error == "twilio_21211"

    @pytest.mark.asyncio
    async def test_send_sms_timeout(self) -> None:
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_from_number_override(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="+15145559999", from_="+15140001111")

        req = transport.requests[0]
        body = req.read().decode()
        # URL-encoded: + becomes %2B
        assert "From=%2B15140001111" in body or "From=+15140001111" in body

    @pytest.mark.asyncio
    async def test_send_mms_with_media(self) -> None:
        transport = _MockTransport(_success_response("SM-MMS"))
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
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
        assert "MediaUrl0=" in body
        assert "Body=" in body

    @pytest.mark.asyncio
    async def test_send_mms_media_only(self) -> None:
        transport = _MockTransport(_success_response("SM-MMS2"))
        cfg = _config()
        provider = TwilioSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
        )
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        req = transport.requests[0]
        body = req.read().decode()
        assert "MediaUrl0=" in body
        # No Body param when no caption
        assert "Body=" not in body


class TestTwilioSignatureVerification:
    def test_verify_valid_signature(self) -> None:
        cfg = _config(auth_token="12345")
        provider = TwilioSMSProvider(cfg)

        # Pre-computed signature for this URL + params + auth_token
        url = "https://example.com/webhook"
        payload = b"Body=Hello&From=%2B15551234567&To=%2B15559876543"

        # Compute expected signature manually for test
        import base64
        import hashlib
        import hmac

        params = {"Body": "Hello", "From": "+15551234567", "To": "+15559876543"}
        validation_string = url
        for key in sorted(params.keys()):
            validation_string += key + params[key]
        expected_sig = base64.b64encode(
            hmac.new(b"12345", validation_string.encode(), hashlib.sha1).digest()
        ).decode()

        assert provider.verify_signature(payload, expected_sig, url=url) is True

    def test_verify_invalid_signature(self) -> None:
        cfg = _config()
        provider = TwilioSMSProvider(cfg)

        payload = b"Body=Hello&From=%2B15551234567"
        assert provider.verify_signature(payload, "invalid", url="https://example.com") is False

    def test_verify_no_url(self) -> None:
        cfg = _config()
        provider = TwilioSMSProvider(cfg)

        assert provider.verify_signature(b"payload", "signature") is False


class TestParseTwilioWebhook:
    def test_parse_webhook_inbound(self) -> None:
        payload = {
            "MessageSid": "SM123456",
            "From": "+15145551111",
            "To": "+15145552222",
            "Body": "Hello from user",
            "AccountSid": "AC123",
            "NumMedia": "0",
            "FromCity": "Montreal",
            "FromState": "QC",
            "FromCountry": "CA",
        }
        msg = parse_twilio_webhook(payload, channel_id="sms-main")

        assert msg.channel_id == "sms-main"
        assert msg.sender_id == "+15145551111"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello from user"
        assert msg.external_id == "SM123456"
        assert msg.idempotency_key == "SM123456"
        assert msg.metadata["to"] == "+15145552222"
        assert msg.metadata["account_sid"] == "AC123"
        assert msg.metadata["from_city"] == "Montreal"

    def test_parse_webhook_minimal(self) -> None:
        payload = {
            "From": "+15145551111",
            "Body": "Hi",
        }
        msg = parse_twilio_webhook(payload, channel_id="sms-main")

        assert msg.sender_id == "+15145551111"
        assert msg.content.body == "Hi"
        assert msg.external_id is None

    def test_parse_webhook_with_media(self) -> None:
        payload = {
            "MessageSid": "SM-MMS-1",
            "From": "+15145551111",
            "To": "+15145552222",
            "Body": "Look at this",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/media/img.jpg",
            "MediaContentType0": "image/jpeg",
        }
        msg = parse_twilio_webhook(payload, channel_id="sms-main")

        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://api.twilio.com/media/img.jpg"
        assert msg.content.caption == "Look at this"

    def test_parse_webhook_multiple_media(self) -> None:
        payload = {
            "MessageSid": "SM-MMS-2",
            "From": "+15145551111",
            "To": "+15145552222",
            "Body": "Photos",
            "NumMedia": "2",
            "MediaUrl0": "https://api.twilio.com/media/a.jpg",
            "MediaContentType0": "image/jpeg",
            "MediaUrl1": "https://api.twilio.com/media/b.png",
            "MediaContentType1": "image/png",
        }
        msg = parse_twilio_webhook(payload, channel_id="sms-main")

        assert isinstance(msg.content, CompositeContent)
        assert len(msg.content.parts) == 3  # text + 2 media

    def test_parse_webhook_media_only(self) -> None:
        payload = {
            "MessageSid": "SM-MMS-3",
            "From": "+15145551111",
            "To": "+15145552222",
            "Body": "",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/media/img.jpg",
            "MediaContentType0": "image/jpeg",
        }
        msg = parse_twilio_webhook(payload, channel_id="sms-main")

        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://api.twilio.com/media/img.jpg"
        assert msg.content.caption is None
