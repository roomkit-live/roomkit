"""Tests for the VoiceMeUp SMS provider."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import MediaContent, TextContent
from roomkit.providers.voicemeup import (
    VoiceMeUpConfig,
    VoiceMeUpSMSProvider,
    configure_voicemeup_mms,
    parse_voicemeup_webhook,
)
from roomkit.providers.voicemeup.sms import _mms_buffer
from tests.conftest import make_event, make_media_event


def _config(**overrides: Any) -> VoiceMeUpConfig:
    defaults: dict[str, Any] = {
        "username": "testuser",
        "auth_token": "secret123",
        "from_number": "+15145551234",
    }
    defaults.update(overrides)
    return VoiceMeUpConfig(**defaults)


def _success_response(sms_hash: str = "abc123") -> dict[str, Any]:
    return {
        "response_details": {
            "response_status": "success",
            "response_messages": {
                "message": [
                    {"code": "queued_sms_hash", "_content": sms_hash},
                ]
            },
        }
    }


def _error_response(code: str, description: str = "") -> dict[str, Any]:
    return {
        "response_details": {
            "response_status": "error",
            "response_messages": {
                "message": [
                    {"code": code, "_content": description or code},
                ]
            },
        }
    }


class _MockTransport(httpx.AsyncBaseTransport):
    """Captures requests and returns a canned JSON response."""

    def __init__(self, response_data: dict[str, Any] | list[dict[str, Any]]) -> None:
        if isinstance(response_data, list):
            self._responses = list(response_data)
        else:
            self._responses = [response_data]
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        data = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return httpx.Response(200, json=data, request=request)


class _TimeoutTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVoiceMeUpConfig:
    def test_production_url(self) -> None:
        cfg = _config()
        assert cfg.base_url == "https://clients.voicemeup.com/api/v1.1/json/"

    def test_sandbox_url(self) -> None:
        cfg = _config(environment="sandbox")
        assert cfg.base_url == "https://dev-clients.voicemeup.com/api/v1.1/json/"


class TestVoiceMeUpSMSProvider:
    @pytest.mark.asyncio
    async def test_send_sms_success(self) -> None:
        transport = _MockTransport(_success_response("hash-42"))
        cfg = _config()
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello from RoomKit")
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        assert result.provider_message_id == "hash-42"

        req = transport.requests[0]
        assert "queue_sms" in str(req.url)
        # Auth params in URL, message data in POST body
        body_str = req.content.decode()
        assert "source_number=15145551234" in body_str
        assert "destination_number=15145559999" in body_str

    @pytest.mark.asyncio
    async def test_send_sms_phone_formatting(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config(from_number="+15145551234")
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="+15145559999")

        req = transport.requests[0]
        body_str = req.content.decode()
        assert "source_number=15145551234" in body_str
        assert "destination_number=15145559999" in body_str
        # No '+' in form data (stripped by _strip_plus)
        assert "%2B" not in body_str

    @pytest.mark.asyncio
    async def test_send_sms_long_message_split(self) -> None:
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        long_body = "A" * 2500
        event = make_event(body=long_body)
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        assert len(transport.requests) == 3  # 1000 + 1000 + 500

    @pytest.mark.asyncio
    async def test_send_sms_error_duplicate(self) -> None:
        transport = _MockTransport(_error_response("duplicate_entry", "SMS already queued"))
        cfg = _config()
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="dup")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "duplicate_entry"

    @pytest.mark.asyncio
    async def test_send_sms_error_permission(self) -> None:
        transport = _MockTransport(_error_response("permission_refused"))
        cfg = _config()
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="nope")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "permission_refused"

    @pytest.mark.asyncio
    async def test_send_sms_network_error(self) -> None:
        cfg = _config()
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_send_mms_with_caption(self) -> None:
        """VoiceMeUp sends MMS with attachment and message."""
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
            caption="Check this!",
        )
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        req = transport.requests[0]
        body_str = req.content.decode()
        assert "message=Check" in body_str
        assert "attachment=https" in body_str  # singular, in POST body

    @pytest.mark.asyncio
    async def test_send_mms_no_caption(self) -> None:
        """VoiceMeUp sends MMS with attachment only (no message)."""
        transport = _MockTransport(_success_response())
        cfg = _config()
        provider = VoiceMeUpSMSProvider(cfg)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
        )
        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        req = transport.requests[0]
        body_str = req.content.decode()
        assert "attachment=https" in body_str
        assert "message=" not in body_str  # No message param when no caption


class TestParseWebhook:
    def test_parse_webhook_inbound(self) -> None:
        payload = {
            "message": "Hello from user",
            "source_number": "+15145551111",
            "destination_number": "+15145552222",
            "direction": "inbound",
            "sms_hash": "hash-abc",
            "datetime_transmission": "2026-01-27T10:30:00Z",
        }
        msg = parse_voicemeup_webhook(payload, channel_id="sms-main")

        assert msg is not None
        assert msg.channel_id == "sms-main"
        assert msg.sender_id == "+15145551111"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello from user"
        assert msg.external_id == "hash-abc"
        assert msg.idempotency_key == "hash-abc"
        assert msg.metadata["destination_number"] == "+15145552222"
        assert msg.metadata["direction"] == "inbound"
        assert msg.metadata["has_attachment"] is False

    def test_parse_webhook_with_attachment(self) -> None:
        """Real VoiceMeUp MMS payload format."""
        payload = {
            "sms_log_id": "5236238",
            "sms_hash": "ce650cb8-1c2d-a751-d397-31b901ec4efd",
            "datetime": "2026-01-28 10:29:05",
            "date": "2026-01-28",
            "time": "10:29:02",
            "direction": "inbound",
            "source_number": "14188050723",
            "destination_number": "14184805945",
            "client_did_id": "87289",
            "message": "",
            "attachment": "https://clients.voicemeup.com/file/ce650cb8.mms.jpg",
            "attachment_size": "93002",
            "attachment_mime_type": "image/jpeg",
            "status": "enabled",
            "transmission_status": "completed",
            "datetime_transmission": "2026-01-28 10:29:02",
        }
        msg = parse_voicemeup_webhook(payload, channel_id="sms-main")

        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://clients.voicemeup.com/file/ce650cb8.mms.jpg"
        assert msg.content.mime_type == "image/jpeg"
        assert msg.content.caption is None  # Empty message
        assert msg.metadata["has_attachment"] is True

    def test_parse_webhook_with_attachment_and_caption(self) -> None:
        payload = {
            "message": "Look at this",
            "source_number": "+15145551111",
            "destination_number": "+15145552222",
            "direction": "inbound",
            "sms_hash": "hash-mms",
            "attachment": "https://cdn.voicemeup.com/img.jpg",
            "attachment_mime_type": "image/jpeg",
        }
        msg = parse_voicemeup_webhook(payload, channel_id="sms-main")

        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://cdn.voicemeup.com/img.jpg"
        assert msg.content.caption == "Look at this"
        assert msg.metadata["has_attachment"] is True

    def test_parse_webhook_legacy_attachment_url_field(self) -> None:
        """Backwards compatibility with attachment_url field name."""
        payload = {
            "message": "",
            "source_number": "+15145551111",
            "destination_number": "+15145552222",
            "sms_hash": "hash-mms2",
            "attachment_url": "https://cdn.voicemeup.com/img.jpg",
            "attachment_type": "image/jpeg",
        }
        msg = parse_voicemeup_webhook(payload, channel_id="sms-main")

        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://cdn.voicemeup.com/img.jpg"
        assert msg.content.mime_type == "image/jpeg"


class TestMMSAggregation:
    """Tests for automatic MMS webhook aggregation."""

    @pytest.fixture(autouse=True)
    def clear_buffer(self) -> None:
        """Clear MMS buffer before each test."""
        _mms_buffer.clear()

    def _make_mms_html_payload(
        self,
        text: str = "Hello",
        source: str = "14188050723",
        dest: str = "14184805945",
        timestamp: str = "2026-01-28 13:00:14",
    ) -> dict[str, Any]:
        """Create a VoiceMeUp MMS metadata webhook (first part)."""
        return {
            "sms_log_id": "5237355",
            "sms_hash": "c3efc8a3-f901-52ac-9868-93f6e89d10f6",
            "datetime": "2026-01-28 13:00:16",
            "date": "2026-01-28",
            "time": "13:00:14",
            "direction": "inbound",
            "source_number": source,
            "destination_number": dest,
            "message": text,
            "attachment": "https://clients.voicemeup.com/file/c3efc8a3.mms.html",
            "attachment_size": "343",
            "attachment_mime_type": "text/html",
            "datetime_transmission": timestamp,
        }

    def _make_mms_image_payload(
        self,
        source: str = "14188050723",
        dest: str = "14184805945",
        timestamp: str = "2026-01-28 13:00:14",
    ) -> dict[str, Any]:
        """Create a VoiceMeUp MMS image webhook (second part)."""
        return {
            "sms_log_id": "5237356",
            "sms_hash": "249fad7f-dc67-09bb-52d6-cd752856bae0",
            "datetime": "2026-01-28 13:00:17",
            "date": "2026-01-28",
            "time": "13:00:14",
            "direction": "inbound",
            "source_number": source,
            "destination_number": dest,
            "message": "",
            "attachment": "https://clients.voicemeup.com/file/249fad7f.mms.jpg",
            "attachment_size": "93002",
            "attachment_mime_type": "image/jpeg",
            "datetime_transmission": timestamp,
        }

    def _make_sms_payload(self, text: str = "Plain SMS") -> dict[str, Any]:
        """Create a regular SMS webhook (no attachment)."""
        return {
            "sms_hash": "sms-hash-123",
            "direction": "inbound",
            "source_number": "14188050723",
            "destination_number": "14184805945",
            "message": text,
            "datetime_transmission": "2026-01-28 14:00:00",
        }

    def test_regular_sms_passes_through(self) -> None:
        """Regular SMS (no attachment) should return immediately."""
        payload = self._make_sms_payload("Hello world")
        result = parse_voicemeup_webhook(payload, channel_id="sms")

        assert result is not None
        assert isinstance(result.content, TextContent)
        assert result.content.body == "Hello world"

    def test_mms_html_buffers(self) -> None:
        """First MMS webhook (.mms.html) should buffer and return None."""
        payload = self._make_mms_html_payload(text="Check this image")
        result = parse_voicemeup_webhook(payload, channel_id="sms")

        assert result is None
        assert len(_mms_buffer) == 1

    def test_mms_merge_text_and_image(self) -> None:
        """Second MMS webhook should merge with buffered text."""
        # First webhook: text + .mms.html
        html_payload = self._make_mms_html_payload(text="Look at this photo!")
        result1 = parse_voicemeup_webhook(html_payload, channel_id="sms")
        assert result1 is None

        # Second webhook: image (same source/dest/timestamp)
        image_payload = self._make_mms_image_payload()
        result2 = parse_voicemeup_webhook(image_payload, channel_id="sms")

        assert result2 is not None
        assert isinstance(result2.content, MediaContent)
        assert result2.content.url == "https://clients.voicemeup.com/file/249fad7f.mms.jpg"
        assert result2.content.caption == "Look at this photo!"
        assert len(_mms_buffer) == 0  # Cleared after merge

    def test_mms_merge_combines_sms_hash(self) -> None:
        """Merged MMS should combine both sms_hash values."""
        html_payload = self._make_mms_html_payload()
        parse_voicemeup_webhook(html_payload, channel_id="sms")

        image_payload = self._make_mms_image_payload()
        result = parse_voicemeup_webhook(image_payload, channel_id="sms")

        assert result is not None
        # Combined hash for traceability
        assert "c3efc8a3-f901-52ac-9868-93f6e89d10f6" in result.external_id
        assert "249fad7f-dc67-09bb-52d6-cd752856bae0" in result.external_id

    def test_mms_different_sender_no_merge(self) -> None:
        """MMS from different sender should not merge."""
        # First webhook from sender A
        html_payload = self._make_mms_html_payload(source="14181111111")
        parse_voicemeup_webhook(html_payload, channel_id="sms")

        # Second webhook from sender B (different source)
        image_payload = self._make_mms_image_payload(source="14182222222")
        result = parse_voicemeup_webhook(image_payload, channel_id="sms")

        # Should NOT merge — image becomes standalone MMS
        assert result is not None
        assert isinstance(result.content, MediaContent)
        assert result.content.caption is None  # No text merged
        assert len(_mms_buffer) == 1  # First still pending

    def test_mms_different_timestamp_no_merge(self) -> None:
        """MMS with different timestamp should not merge."""
        html_payload = self._make_mms_html_payload(timestamp="2026-01-28 13:00:14")
        parse_voicemeup_webhook(html_payload, channel_id="sms")

        image_payload = self._make_mms_image_payload(timestamp="2026-01-28 13:05:00")
        result = parse_voicemeup_webhook(image_payload, channel_id="sms")

        assert result is not None
        assert result.content.caption is None  # No merge
        assert len(_mms_buffer) == 1

    @pytest.mark.asyncio
    async def test_timeout_emits_text_only(self) -> None:
        """If second webhook never arrives, emit text-only after timeout."""
        received: list[InboundMessage] = []

        async def on_timeout(msg: InboundMessage) -> None:
            received.append(msg)

        configure_voicemeup_mms(timeout_seconds=0.1, on_timeout=on_timeout)

        html_payload = self._make_mms_html_payload(text="Where is my image?")
        result = parse_voicemeup_webhook(html_payload, channel_id="sms")
        assert result is None

        # Wait for timeout
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert isinstance(received[0].content, TextContent)
        assert received[0].content.body == "Where is my image?"
        assert len(_mms_buffer) == 0

        # Reset config
        configure_voicemeup_mms(timeout_seconds=5.0, on_timeout=None)

    def test_standalone_mms_image_passes_through(self) -> None:
        """MMS image without prior .mms.html should pass through."""
        payload = self._make_mms_image_payload()
        result = parse_voicemeup_webhook(payload, channel_id="sms")

        assert result is not None
        assert isinstance(result.content, MediaContent)
        assert result.content.url == "https://clients.voicemeup.com/file/249fad7f.mms.jpg"
