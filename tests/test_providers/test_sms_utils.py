"""Tests for SMS utility functions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from roomkit.models.event import (
    CompositeContent,
    MediaContent,
    TextContent,
)
from roomkit.providers.sms.meta import (
    WebhookMeta,
    build_inbound_content,
    extract_media_urls,
    extract_sinch_meta,
    extract_sms_meta,
    extract_telnyx_meta,
    extract_text_body,
    extract_twilio_meta,
    extract_voicemeup_meta,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestExtractMediaUrls:
    def test_text_content(self) -> None:
        content = TextContent(body="hello")
        assert extract_media_urls(content) == []

    def test_media_content(self) -> None:
        content = MediaContent(url="https://example.com/img.jpg", mime_type="image/jpeg")
        assert extract_media_urls(content) == ["https://example.com/img.jpg"]

    def test_composite_content(self) -> None:
        content = CompositeContent(
            parts=[
                TextContent(body="hello"),
                MediaContent(url="https://example.com/a.jpg", mime_type="image/jpeg"),
                MediaContent(url="https://example.com/b.png", mime_type="image/png"),
            ]
        )
        assert extract_media_urls(content) == [
            "https://example.com/a.jpg",
            "https://example.com/b.png",
        ]


class TestExtractTextBody:
    def test_text_content(self) -> None:
        assert extract_text_body(TextContent(body="hello")) == "hello"

    def test_media_with_caption(self) -> None:
        content = MediaContent(
            url="https://example.com/img.jpg", mime_type="image/jpeg", caption="Look!"
        )
        assert extract_text_body(content) == "Look!"

    def test_media_without_caption(self) -> None:
        content = MediaContent(url="https://example.com/img.jpg", mime_type="image/jpeg")
        assert extract_text_body(content) == ""

    def test_composite_content(self) -> None:
        content = CompositeContent(
            parts=[
                TextContent(body="hello"),
                MediaContent(
                    url="https://example.com/a.jpg", mime_type="image/jpeg", caption="pic"
                ),
            ]
        )
        assert extract_text_body(content) == "hello pic"

    def test_empty_text(self) -> None:
        assert extract_text_body(TextContent(body="")) == ""


class TestBuildInboundContent:
    def test_text_only(self) -> None:
        content = build_inbound_content("hello", [])
        assert isinstance(content, TextContent)
        assert content.body == "hello"

    def test_single_media_no_text(self) -> None:
        media = [{"url": "https://example.com/img.jpg", "mime_type": "image/jpeg"}]
        content = build_inbound_content("", media)
        assert isinstance(content, MediaContent)
        assert content.url == "https://example.com/img.jpg"
        assert content.mime_type == "image/jpeg"

    def test_single_media_with_text(self) -> None:
        media = [{"url": "https://example.com/img.jpg", "mime_type": "image/jpeg"}]
        content = build_inbound_content("Check this", media)
        assert isinstance(content, MediaContent)
        assert content.caption == "Check this"

    def test_multiple_media(self) -> None:
        content = build_inbound_content(
            "Photos",
            [
                {"url": "https://example.com/a.jpg", "mime_type": "image/jpeg"},
                {"url": "https://example.com/b.png", "mime_type": "image/png"},
            ],
        )
        assert isinstance(content, CompositeContent)
        assert len(content.parts) == 3  # text + 2 media
        assert isinstance(content.parts[0], TextContent)
        assert isinstance(content.parts[1], MediaContent)

    def test_multiple_media_no_text(self) -> None:
        content = build_inbound_content(
            "",
            [
                {"url": "https://example.com/a.jpg", "mime_type": "image/jpeg"},
                {"url": "https://example.com/b.png", "mime_type": "image/png"},
            ],
        )
        assert isinstance(content, CompositeContent)
        assert len(content.parts) == 2  # 2 media, no text

    def test_missing_mime_type_defaults(self) -> None:
        media = [{"url": "https://example.com/file", "mime_type": None}]
        content = build_inbound_content("", media)
        assert isinstance(content, MediaContent)
        assert content.mime_type == "application/octet-stream"

    def test_skips_entries_without_url(self) -> None:
        content = build_inbound_content("hi", [{"url": None, "mime_type": "image/jpeg"}])
        assert isinstance(content, TextContent)
        assert content.body == "hi"


# ---------------------------------------------------------------------------
# Provider meta extractors
# ---------------------------------------------------------------------------


class TestExtractVoicemeupMeta:
    def test_extract_full_payload(self) -> None:
        payload = {
            "message": "Hello from user",
            "source_number": "+15145551111",
            "destination_number": "+15145552222",
            "sms_hash": "hash-abc",
            "datetime_transmission": "2026-01-28T12:00:00Z",
        }
        meta = extract_voicemeup_meta(payload)

        assert meta.provider == "voicemeup"
        assert meta.sender == "+15145551111"
        assert meta.recipient == "+15145552222"
        assert meta.body == "Hello from user"
        assert meta.external_id == "hash-abc"
        assert meta.timestamp == datetime(2026, 1, 28, 12, 0, 0, tzinfo=UTC)
        assert meta.raw is payload
        assert meta.media_urls == []

    def test_extract_minimal_payload(self) -> None:
        payload = {}
        meta = extract_voicemeup_meta(payload)

        assert meta.provider == "voicemeup"
        assert meta.sender == ""
        assert meta.recipient == ""
        assert meta.body == ""
        assert meta.external_id is None
        assert meta.timestamp is None
        assert meta.media_urls == []

    def test_extract_with_attachment(self) -> None:
        """Real VoiceMeUp MMS payload uses 'attachment' and 'attachment_mime_type'."""
        payload = {
            "message": "See this",
            "source_number": "+15145551111",
            "destination_number": "+15145552222",
            "attachment": "https://clients.voicemeup.com/file/abc.mms.jpg",
            "attachment_mime_type": "image/jpeg",
        }
        meta = extract_voicemeup_meta(payload)

        assert len(meta.media_urls) == 1
        assert meta.media_urls[0]["url"] == "https://clients.voicemeup.com/file/abc.mms.jpg"
        assert meta.media_urls[0]["mime_type"] == "image/jpeg"

    def test_extract_with_attachment_legacy_fields(self) -> None:
        """Backwards compatibility with attachment_url/attachment_type field names."""
        payload = {
            "message": "See this",
            "source_number": "+15145551111",
            "destination_number": "+15145552222",
            "attachment_url": "https://cdn.voicemeup.com/img.jpg",
            "attachment_type": "image/jpeg",
        }
        meta = extract_voicemeup_meta(payload)

        assert len(meta.media_urls) == 1
        assert meta.media_urls[0]["url"] == "https://cdn.voicemeup.com/img.jpg"
        assert meta.media_urls[0]["mime_type"] == "image/jpeg"


class TestExtractTelnyxMeta:
    def test_extract_full_payload(self) -> None:
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
        meta = extract_telnyx_meta(payload)

        assert meta.provider == "telnyx"
        assert meta.sender == "+15145551111"
        assert meta.recipient == "+15145552222"
        assert meta.body == "Hello from user"
        assert meta.external_id == "msg-uuid-abc"
        assert meta.timestamp == datetime(2026, 1, 28, 12, 0, 0, tzinfo=UTC)
        assert meta.raw is payload
        assert meta.media_urls == []
        assert meta.direction == "inbound"
        assert meta.event_type == "message.received"
        assert meta.is_inbound is True

    def test_extract_outbound_payload(self) -> None:
        """Outbound webhooks should be extractable but marked as not inbound."""
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
        meta = extract_telnyx_meta(payload)

        assert meta.provider == "telnyx"
        assert meta.direction == "outbound"
        assert meta.event_type == "message.sent"
        assert meta.is_inbound is False

    def test_extract_empty_to_list(self) -> None:
        payload = {
            "data": {
                "payload": {
                    "id": "msg-123",
                    "from": {"phone_number": "+15145551111"},
                    "to": [],
                    "text": "Test",
                },
            }
        }
        meta = extract_telnyx_meta(payload)

        assert meta.recipient == ""

    def test_extract_with_media(self) -> None:
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-media",
                    "direction": "inbound",
                    "from": {"phone_number": "+15145551111"},
                    "to": [{"phone_number": "+15145552222"}],
                    "text": "Check this",
                    "media": [
                        {"url": "https://telnyx.com/a.jpg", "content_type": "image/jpeg"},
                        {"url": "https://telnyx.com/b.png", "content_type": "image/png"},
                    ],
                },
            }
        }
        meta = extract_telnyx_meta(payload)

        assert len(meta.media_urls) == 2
        assert meta.media_urls[0]["url"] == "https://telnyx.com/a.jpg"
        assert meta.media_urls[0]["mime_type"] == "image/jpeg"


class TestExtractTwilioMeta:
    def test_extract_full_payload(self) -> None:
        payload = {
            "MessageSid": "SM123456",
            "From": "+15145551111",
            "To": "+15145552222",
            "Body": "Hello from user",
            "AccountSid": "AC123",
        }
        meta = extract_twilio_meta(payload)

        assert meta.provider == "twilio"
        assert meta.sender == "+15145551111"
        assert meta.recipient == "+15145552222"
        assert meta.body == "Hello from user"
        assert meta.external_id == "SM123456"
        assert meta.timestamp is None  # Twilio doesn't include timestamp
        assert meta.raw is payload
        assert meta.media_urls == []

    def test_extract_minimal_payload(self) -> None:
        payload = {}
        meta = extract_twilio_meta(payload)

        assert meta.provider == "twilio"
        assert meta.sender == ""
        assert meta.recipient == ""
        assert meta.body == ""
        assert meta.external_id is None
        assert meta.media_urls == []

    def test_extract_with_media(self) -> None:
        payload = {
            "MessageSid": "SM789",
            "From": "+15145551111",
            "To": "+15145552222",
            "Body": "MMS message",
            "NumMedia": "2",
            "MediaUrl0": "https://api.twilio.com/media/img0.jpg",
            "MediaContentType0": "image/jpeg",
            "MediaUrl1": "https://api.twilio.com/media/img1.png",
            "MediaContentType1": "image/png",
        }
        meta = extract_twilio_meta(payload)

        assert len(meta.media_urls) == 2
        assert meta.media_urls[0]["url"] == "https://api.twilio.com/media/img0.jpg"
        assert meta.media_urls[0]["mime_type"] == "image/jpeg"
        assert meta.media_urls[1]["url"] == "https://api.twilio.com/media/img1.png"


class TestExtractSinchMeta:
    def test_extract_full_payload(self) -> None:
        payload = {
            "id": "msg-uuid-abc",
            "from": "+15145551111",
            "to": "12345",
            "body": "Hello from user",
            "received_at": "2026-01-28T12:00:00Z",
        }
        meta = extract_sinch_meta(payload)

        assert meta.provider == "sinch"
        assert meta.sender == "+15145551111"
        assert meta.recipient == "12345"
        assert meta.body == "Hello from user"
        assert meta.external_id == "msg-uuid-abc"
        assert meta.timestamp == datetime(2026, 1, 28, 12, 0, 0, tzinfo=UTC)
        assert meta.raw is payload
        assert meta.media_urls == []

    def test_extract_minimal_payload(self) -> None:
        payload = {}
        meta = extract_sinch_meta(payload)

        assert meta.provider == "sinch"
        assert meta.sender == ""
        assert meta.recipient == ""
        assert meta.body == ""
        assert meta.external_id is None
        assert meta.timestamp is None
        assert meta.media_urls == []

    def test_extract_with_media(self) -> None:
        payload = {
            "id": "msg-mms",
            "from": "+15145551111",
            "to": "12345",
            "body": "MMS",
            "media": [
                {"url": "https://sinch.com/img.jpg", "mimeType": "image/jpeg"},
            ],
        }
        meta = extract_sinch_meta(payload)

        assert len(meta.media_urls) == 1
        assert meta.media_urls[0]["url"] == "https://sinch.com/img.jpg"
        assert meta.media_urls[0]["mime_type"] == "image/jpeg"


class TestExtractSmsMeta:
    def test_dispatch_voicemeup(self) -> None:
        payload = {"source_number": "+15145551111", "message": "Hi"}
        meta = extract_sms_meta("voicemeup", payload)

        assert meta.provider == "voicemeup"
        assert meta.sender == "+15145551111"

    def test_dispatch_telnyx(self) -> None:
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-1",
                    "direction": "inbound",
                    "from": {"phone_number": "+15145551111"},
                    "to": [{"phone_number": "+15145552222"}],
                    "text": "Hi",
                },
            }
        }
        meta = extract_sms_meta("telnyx", payload)

        assert meta.provider == "telnyx"
        assert meta.sender == "+15145551111"
        assert meta.is_inbound is True

    def test_dispatch_twilio(self) -> None:
        payload = {"From": "+15145551111", "To": "+15145552222", "Body": "Hi"}
        meta = extract_sms_meta("twilio", payload)

        assert meta.provider == "twilio"
        assert meta.sender == "+15145551111"

    def test_dispatch_sinch(self) -> None:
        payload = {"from": "+15145551111", "to": "12345", "body": "Hi"}
        meta = extract_sms_meta("sinch", payload)

        assert meta.provider == "sinch"
        assert meta.sender == "+15145551111"

    def test_case_insensitive(self) -> None:
        payload = {"source_number": "+15145551111"}
        meta = extract_sms_meta("VoiceMeUp", payload)

        assert meta.provider == "voicemeup"

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown SMS provider: unknown"):
            extract_sms_meta("unknown", {})


class TestWebhookMetaDataclass:
    def test_dataclass_fields(self) -> None:
        meta = WebhookMeta(
            provider="test",
            sender="+15145551111",
            recipient="+15145552222",
            body="Hello",
            external_id="id-123",
            timestamp=datetime(2026, 1, 28, 12, 0, 0, tzinfo=UTC),
            raw={"key": "value"},
        )

        assert meta.provider == "test"
        assert meta.sender == "+15145551111"
        assert meta.recipient == "+15145552222"
        assert meta.body == "Hello"
        assert meta.external_id == "id-123"
        assert meta.timestamp is not None
        assert meta.raw == {"key": "value"}
        assert meta.media_urls == []

    def test_to_inbound(self) -> None:
        meta = WebhookMeta(
            provider="twilio",
            sender="+15145551111",
            recipient="+15145552222",
            body="Hello from user",
            external_id="SM123",
            timestamp=datetime(2026, 1, 28, 12, 0, 0, tzinfo=UTC),
            raw={"key": "value"},
        )

        inbound = meta.to_inbound(channel_id="sms-channel")

        assert inbound.channel_id == "sms-channel"
        assert inbound.sender_id == "+15145551111"
        assert isinstance(inbound.content, TextContent)
        assert inbound.content.body == "Hello from user"
        assert inbound.external_id == "SM123"
        assert inbound.idempotency_key == "SM123"
        assert inbound.metadata["provider"] == "twilio"
        assert inbound.metadata["recipient"] == "+15145552222"
        assert inbound.metadata["timestamp"] == "2026-01-28T12:00:00+00:00"

    def test_to_inbound_no_timestamp(self) -> None:
        meta = WebhookMeta(
            provider="twilio",
            sender="+15145551111",
            recipient="+15145552222",
            body="Hello",
            external_id=None,
            timestamp=None,
            raw={},
        )

        inbound = meta.to_inbound(channel_id="sms-channel")

        assert inbound.metadata["timestamp"] is None
        assert inbound.external_id is None
        assert inbound.idempotency_key is None

    def test_to_inbound_with_media(self) -> None:
        meta = WebhookMeta(
            provider="twilio",
            sender="+15145551111",
            recipient="+15145552222",
            body="Check this out",
            external_id="SM456",
            timestamp=None,
            raw={},
            media_urls=[{"url": "https://example.com/img.jpg", "mime_type": "image/jpeg"}],
        )

        inbound = meta.to_inbound(channel_id="sms-channel")

        assert isinstance(inbound.content, MediaContent)
        assert inbound.content.url == "https://example.com/img.jpg"
        assert inbound.content.caption == "Check this out"

    def test_to_inbound_media_only(self) -> None:
        meta = WebhookMeta(
            provider="telnyx",
            sender="+15145551111",
            recipient="+15145552222",
            body="",
            external_id="msg-1",
            timestamp=None,
            raw={},
            media_urls=[{"url": "https://example.com/img.jpg", "mime_type": "image/jpeg"}],
        )

        inbound = meta.to_inbound(channel_id="sms-channel")

        assert isinstance(inbound.content, MediaContent)
        assert inbound.content.url == "https://example.com/img.jpg"
        assert inbound.content.caption is None

    def test_to_inbound_multiple_media(self) -> None:
        meta = WebhookMeta(
            provider="twilio",
            sender="+15145551111",
            recipient="+15145552222",
            body="Photos",
            external_id="SM789",
            timestamp=None,
            raw={},
            media_urls=[
                {"url": "https://example.com/a.jpg", "mime_type": "image/jpeg"},
                {"url": "https://example.com/b.png", "mime_type": "image/png"},
            ],
        )

        inbound = meta.to_inbound(channel_id="sms-channel")

        assert isinstance(inbound.content, CompositeContent)
        assert len(inbound.content.parts) == 3  # text + 2 media

    def test_to_inbound_rejects_outbound(self) -> None:
        """to_inbound() raises ValueError for outbound status webhooks."""
        meta = WebhookMeta(
            provider="telnyx",
            sender="+15145552222",
            recipient="+15145551111",
            body="Outbound message",
            external_id="msg-out",
            timestamp=None,
            raw={},
            direction="outbound",
            event_type="message.sent",
        )

        assert meta.is_inbound is False

        with pytest.raises(ValueError, match="Cannot convert outbound webhook"):
            meta.to_inbound(channel_id="sms-channel")
