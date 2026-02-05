"""Tests for the Telnyx SMS provider."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from roomkit.models.event import MediaContent, TextContent
from roomkit.providers.telnyx import (
    TelnyxConfig,
    TelnyxSMSProvider,
    parse_telnyx_webhook,
)
from roomkit.providers.telnyx.sms import _is_telnyx_inbound
from tests.conftest import make_event, make_media_event

# Check if PyNaCl is available for signature tests
try:
    from nacl.signing import SigningKey

    HAS_NACL = True
except ImportError:
    HAS_NACL = False


def _config(**overrides: Any) -> TelnyxConfig:
    defaults: dict[str, Any] = {
        "api_key": "KEY_test123",
        "from_number": "+15145551234",
    }
    defaults.update(overrides)
    return TelnyxConfig(**defaults)


def _success_response(message_id: str = "msg-uuid-123") -> dict[str, Any]:
    return {
        "data": {
            "id": message_id,
            "record_type": "message",
            "direction": "outbound",
            "type": "SMS",
        }
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


class TestTelnyxConfig:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.timeout == 10.0
        assert cfg.messaging_profile_id is None

    def test_config_with_messaging_profile(self) -> None:
        cfg = _config(messaging_profile_id="profile-123")
        assert cfg.messaging_profile_id == "profile-123"


class TestTelnyxSMSProvider:
    @pytest.mark.asyncio
    async def test_send_sms_success(self) -> None:
        transport = _MockTransport(_success_response("msg-42"))
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello from RoomKit")
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        assert result.provider_message_id == "msg-42"

        req = transport.requests[0]
        assert req.url == "https://api.telnyx.com/v2/messages"
        assert req.headers["Authorization"] == "Bearer KEY_test123"

    @pytest.mark.asyncio
    async def test_send_sms_with_messaging_profile(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config(messaging_profile_id="profile-abc")
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="+15145559999")

        req = transport.requests[0]
        body = req.read().decode()
        assert "profile-abc" in body

    @pytest.mark.asyncio
    async def test_send_sms_empty_message(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "empty_message"
        assert len(transport.requests) == 0

    @pytest.mark.asyncio
    async def test_send_sms_auth_error(self) -> None:
        transport = _MockTransport({"errors": [{"detail": "Unauthorized"}]}, status_code=401)
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "auth_error"

    @pytest.mark.asyncio
    async def test_send_sms_rate_limit(self) -> None:
        err = {"errors": [{"detail": "Rate limit exceeded"}]}
        transport = _MockTransport(err, status_code=429)
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "rate_limit"

    @pytest.mark.asyncio
    async def test_send_sms_invalid_number(self) -> None:
        transport = _MockTransport(
            {"errors": [{"detail": "Invalid phone number"}]},
            status_code=400,
        )
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="invalid")

        assert result.success is False
        assert result.error == "invalid_request"

    @pytest.mark.asyncio
    async def test_send_sms_timeout(self) -> None:
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_from_number_override(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="+15145559999", from_="+15140001111")

        req = transport.requests[0]
        body = req.read().decode()
        assert "+15140001111" in body

    @pytest.mark.asyncio
    async def test_send_mms_with_media(self) -> None:
        transport = _MockTransport(_success_response("msg-mms"))
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
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
        assert "media_urls" in body
        assert "https://example.com/image.jpg" in body

    @pytest.mark.asyncio
    async def test_send_mms_media_only(self) -> None:
        transport = _MockTransport(_success_response("msg-mms2"))
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
        )
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        req = transport.requests[0]
        body = req.read().decode()
        assert "media_urls" in body
        # No text when no caption
        assert '"text"' not in body


class TestIsTelnyxInbound:
    def test_inbound_message(self) -> None:
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {"direction": "inbound"},
            }
        }
        assert _is_telnyx_inbound(payload) is True

    def test_outbound_sent(self) -> None:
        payload = {
            "data": {
                "event_type": "message.sent",
                "payload": {"direction": "outbound"},
            }
        }
        assert _is_telnyx_inbound(payload) is False

    def test_outbound_finalized(self) -> None:
        payload = {
            "data": {
                "event_type": "message.finalized",
                "payload": {"direction": "outbound"},
            }
        }
        assert _is_telnyx_inbound(payload) is False

    def test_received_but_outbound_direction(self) -> None:
        """Edge case: message.received but direction is outbound (unlikely)."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {"direction": "outbound"},
            }
        }
        assert _is_telnyx_inbound(payload) is False


class TestParseTelnyxWebhook:
    def test_parse_webhook_inbound(self) -> None:
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-uuid-abc",
                    "direction": "inbound",
                    "from": {"phone_number": "+15145551111"},
                    "to": [{"phone_number": "+15145552222"}],
                    "text": "Hello from user",
                    "received_at": "2026-01-28T12:00:00Z",
                },
            }
        }
        msg = parse_telnyx_webhook(payload, channel_id="sms-main")

        assert msg.channel_id == "sms-main"
        assert msg.sender_id == "+15145551111"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello from user"
        assert msg.external_id == "msg-uuid-abc"
        assert msg.idempotency_key == "msg-uuid-abc"
        assert msg.metadata["destination_number"] == "+15145552222"
        assert msg.metadata["received_at"] == "2026-01-28T12:00:00Z"

    def test_parse_webhook_empty_text(self) -> None:
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-xyz",
                    "direction": "inbound",
                    "from": {"phone_number": "+15145551111"},
                    "to": [{"phone_number": "+15145552222"}],
                },
            }
        }
        msg = parse_telnyx_webhook(payload, channel_id="sms-main")

        assert msg.content.body == ""

    def test_parse_webhook_with_media(self) -> None:
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-mms",
                    "direction": "inbound",
                    "from": {"phone_number": "+15145551111"},
                    "to": [{"phone_number": "+15145552222"}],
                    "text": "Look at this",
                    "media": [
                        {"url": "https://telnyx.com/img.jpg", "content_type": "image/jpeg"},
                    ],
                },
            }
        }
        msg = parse_telnyx_webhook(payload, channel_id="sms-main")

        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://telnyx.com/img.jpg"
        assert msg.content.caption == "Look at this"

    def test_parse_webhook_media_only(self) -> None:
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-mms2",
                    "direction": "inbound",
                    "from": {"phone_number": "+15145551111"},
                    "to": [{"phone_number": "+15145552222"}],
                    "text": "",
                    "media": [
                        {"url": "https://telnyx.com/img.jpg", "content_type": "image/jpeg"},
                    ],
                },
            }
        }
        msg = parse_telnyx_webhook(payload, channel_id="sms-main")

        assert isinstance(msg.content, MediaContent)
        assert msg.content.caption is None

    def test_parse_webhook_rejects_outbound(self) -> None:
        """Outbound webhooks should raise ValueError by default."""
        payload = {
            "data": {
                "event_type": "message.sent",
                "payload": {
                    "id": "msg-out",
                    "direction": "outbound",
                    "from": {"phone_number": "+15145552222"},
                    "to": [{"phone_number": "+15145551111", "status": "sent"}],
                    "text": "Outbound message",
                },
            }
        }
        with pytest.raises(ValueError, match="Not an inbound message"):
            parse_telnyx_webhook(payload, channel_id="sms-main")

    def test_parse_webhook_strict_false_allows_outbound(self) -> None:
        """strict=False allows parsing outbound webhooks."""
        payload = {
            "data": {
                "event_type": "message.sent",
                "payload": {
                    "id": "msg-out",
                    "direction": "outbound",
                    "from": {"phone_number": "+15145552222"},
                    "to": [{"phone_number": "+15145551111"}],
                    "text": "Outbound message",
                },
            }
        }
        msg = parse_telnyx_webhook(payload, channel_id="sms-main", strict=False)

        assert msg.sender_id == "+15145552222"
        assert msg.content.body == "Outbound message"


@pytest.mark.skipif(not HAS_NACL, reason="PyNaCl not installed")
class TestTelnyxSignatureVerification:
    def _generate_keypair(self) -> tuple[str, str]:
        """Generate a test ED25519 keypair, returns (public_key_b64, private_key)."""
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key
        public_key_b64 = base64.b64encode(bytes(verify_key)).decode()
        return public_key_b64, signing_key

    def _sign_payload(self, signing_key: Any, timestamp: str, payload: bytes) -> str:
        """Sign a payload like Telnyx does."""
        signed_payload = f"{timestamp}|".encode() + payload
        signature = signing_key.sign(signed_payload).signature
        return base64.b64encode(signature).decode()

    def test_verify_valid_signature(self) -> None:
        public_key_b64, signing_key = self._generate_keypair()
        cfg = _config()
        provider = TelnyxSMSProvider(cfg, public_key=public_key_b64)

        payload = b'{"data": {"payload": {"text": "Hello"}}}'
        timestamp = "1706443200"
        signature = self._sign_payload(signing_key, timestamp, payload)

        assert provider.verify_signature(payload, signature, timestamp) is True

    def test_verify_invalid_signature(self) -> None:
        public_key_b64, _ = self._generate_keypair()
        cfg = _config()
        provider = TelnyxSMSProvider(cfg, public_key=public_key_b64)

        payload = b'{"data": {"payload": {"text": "Hello"}}}'
        timestamp = "1706443200"
        invalid_signature = base64.b64encode(b"invalid" * 8).decode()

        assert provider.verify_signature(payload, invalid_signature, timestamp) is False

    def test_verify_tampered_payload(self) -> None:
        public_key_b64, signing_key = self._generate_keypair()
        cfg = _config()
        provider = TelnyxSMSProvider(cfg, public_key=public_key_b64)

        original_payload = b'{"data": {"payload": {"text": "Hello"}}}'
        timestamp = "1706443200"
        signature = self._sign_payload(signing_key, timestamp, original_payload)

        tampered_payload = b'{"data": {"payload": {"text": "Tampered"}}}'
        assert provider.verify_signature(tampered_payload, signature, timestamp) is False

    def test_verify_wrong_timestamp(self) -> None:
        public_key_b64, signing_key = self._generate_keypair()
        cfg = _config()
        provider = TelnyxSMSProvider(cfg, public_key=public_key_b64)

        payload = b'{"data": {"payload": {"text": "Hello"}}}'
        timestamp = "1706443200"
        signature = self._sign_payload(signing_key, timestamp, payload)

        wrong_timestamp = "1706443201"
        assert provider.verify_signature(payload, signature, wrong_timestamp) is False

    def test_verify_no_public_key(self) -> None:
        cfg = _config()
        provider = TelnyxSMSProvider(cfg)  # No public_key

        with pytest.raises(ValueError, match="public_key must be provided"):
            provider.verify_signature(b"payload", "signature", "timestamp")

    def test_verify_no_timestamp(self) -> None:
        public_key_b64, _ = self._generate_keypair()
        cfg = _config()
        provider = TelnyxSMSProvider(cfg, public_key=public_key_b64)

        assert provider.verify_signature(b"payload", "signature", None) is False
