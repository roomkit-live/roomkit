"""Tests for Telnyx RCS provider."""

from __future__ import annotations

import httpx
import pytest

from roomkit.models.event import MediaContent, TextContent
from roomkit.providers.telnyx.rcs import (
    TelnyxRCSConfig,
    TelnyxRCSProvider,
    parse_telnyx_rcs_webhook,
)
from tests.conftest import make_event


class TestTelnyxRCSProvider:
    """Tests for TelnyxRCSProvider."""

    @pytest.fixture
    def config(self) -> TelnyxRCSConfig:
        return TelnyxRCSConfig(
            api_key="test-api-key",
            agent_id="test-agent-id",
            messaging_profile_id="test-profile-id",
        )

    @pytest.fixture
    def provider(self, config: TelnyxRCSConfig) -> TelnyxRCSProvider:
        return TelnyxRCSProvider(config)

    def test_sender_id(self, provider: TelnyxRCSProvider) -> None:
        """sender_id returns the agent_id."""
        assert provider.sender_id == "test-agent-id"

    async def test_send_text_success(self, config: TelnyxRCSConfig) -> None:
        """Send text message successfully."""
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "data": {
                        "id": "msg-123",
                        "type": "RCS",
                    }
                },
            )
        )
        provider = TelnyxRCSProvider(config)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello via RCS")
        result = await provider.send(event, to="+15145551234")

        assert result.success is True
        assert result.provider_message_id == "msg-123"
        assert result.channel_used == "rcs"
        assert result.fallback is False

    async def test_send_with_media(self, config: TelnyxRCSConfig) -> None:
        """Send message with media."""
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"data": {"id": "msg-456", "type": "RCS"}},
            )
        )
        provider = TelnyxRCSProvider(config)
        provider._client = httpx.AsyncClient(transport=transport)

        from roomkit.models.event import CompositeContent, EventSource, RoomEvent

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="rcs", channel_type="rcs"),
            content=CompositeContent(
                parts=[
                    TextContent(body="Check this out"),
                    MediaContent(url="https://example.com/image.jpg", mime_type="image/jpeg"),
                ]
            ),
        )
        result = await provider.send(event, to="+15145551234")

        assert result.success is True

    async def test_send_empty_message(self, provider: TelnyxRCSProvider) -> None:
        """Empty message returns error."""
        event = make_event(body="")
        result = await provider.send(event, to="+15145551234")

        assert result.success is False
        assert result.error == "empty_message"

    async def test_send_fallback_occurred(self, config: TelnyxRCSConfig) -> None:
        """Detect when SMS fallback occurred."""
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "data": {
                        "id": "msg-789",
                        "type": "SMS",  # Fallback to SMS
                    }
                },
            )
        )
        provider = TelnyxRCSProvider(config)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello")
        result = await provider.send(event, to="+15145551234")

        assert result.success is True
        assert result.channel_used == "sms"
        assert result.fallback is True

    async def test_send_timeout(self, config: TelnyxRCSConfig) -> None:
        """Timeout returns error."""

        def raise_timeout(req: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timeout")

        transport = httpx.MockTransport(raise_timeout)
        provider = TelnyxRCSProvider(config)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello")
        result = await provider.send(event, to="+15145551234")

        assert result.success is False
        assert result.error == "timeout"

    async def test_send_auth_error(self, config: TelnyxRCSConfig) -> None:
        """401 returns auth_error."""
        transport = httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": "Unauthorized"})
        )
        provider = TelnyxRCSProvider(config)
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello")
        result = await provider.send(event, to="+15145551234")

        assert result.success is False
        assert result.error == "auth_error"

    async def test_check_capability_supported(self, config: TelnyxRCSConfig) -> None:
        """check_capability returns True when RCS is supported."""
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"data": {"rcs_enabled": True}},
            )
        )
        provider = TelnyxRCSProvider(config)
        provider._client = httpx.AsyncClient(transport=transport)

        result = await provider.check_capability("+15145551234")
        assert result is True

    async def test_check_capability_not_supported(self, config: TelnyxRCSConfig) -> None:
        """check_capability returns False when RCS is not supported."""
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"data": {"rcs_enabled": False}},
            )
        )
        provider = TelnyxRCSProvider(config)
        provider._client = httpx.AsyncClient(transport=transport)

        result = await provider.check_capability("+15145551234")
        assert result is False


class TestParseTelnyxRCSWebhook:
    """Tests for parse_telnyx_rcs_webhook."""

    def test_parse_text_message(self) -> None:
        """Parse a simple text message."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-abc123",
                    "direction": "inbound",
                    "text": "Hello from user",
                    "type": "RCS",
                    "from": {
                        "phone_number": "+15145551111",
                    },
                    "to": {
                        "agent_id": "my-agent",
                        "agent_name": "My Business",
                    },
                    "received_at": "2025-01-28T10:30:00Z",
                },
            }
        }

        message = parse_telnyx_rcs_webhook(payload, channel_id="rcs-channel")

        assert message.channel_id == "rcs-channel"
        assert message.sender_id == "+15145551111"
        assert message.external_id == "msg-abc123"
        assert isinstance(message.content, TextContent)
        assert message.content.body == "Hello from user"
        assert message.metadata["agent_id"] == "my-agent"
        assert message.metadata["type"] == "RCS"

    def test_parse_message_with_media(self) -> None:
        """Parse a message with media attachment."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-media123",
                    "direction": "inbound",
                    "text": "Check this image",
                    "type": "RCS",
                    "from": {"phone_number": "+15145551111"},
                    "to": {"agent_id": "my-agent"},
                    "media": [
                        {
                            "url": "https://example.com/image.jpg",
                            "content_type": "image/jpeg",
                        }
                    ],
                },
            }
        }

        message = parse_telnyx_rcs_webhook(payload, channel_id="rcs")

        # Single media with text becomes MediaContent with caption
        assert isinstance(message.content, MediaContent)
        assert message.content.url == "https://example.com/image.jpg"
        assert message.content.caption == "Check this image"

    def test_parse_user_file_transfer(self) -> None:
        """Parse a message with user file transfer."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-file123",
                    "direction": "inbound",
                    "text": "",
                    "type": "RCS",
                    "from": {"phone_number": "+15145551111"},
                    "to": {"agent_id": "my-agent"},
                    "user_file": {
                        "payload": {
                            "file_name": "document.pdf",
                            "file_uri": "https://example.com/files/document.pdf",
                            "mime_type": "application/pdf",
                        }
                    },
                },
            }
        }

        message = parse_telnyx_rcs_webhook(payload, channel_id="rcs")

        assert isinstance(message.content, MediaContent)
        assert message.content.url == "https://example.com/files/document.pdf"
        assert message.content.mime_type == "application/pdf"

    def test_parse_suggestion_response(self) -> None:
        """Parse a suggestion response (button click)."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-suggest123",
                    "direction": "inbound",
                    "text": "Yes, I'm interested",
                    "type": "RCS",
                    "from": {"phone_number": "+15145551111"},
                    "to": {"agent_id": "my-agent"},
                    "suggestion_response": {
                        "postback_data": "interested_yes",
                        "text": "Yes, I'm interested",
                    },
                },
            }
        }

        message = parse_telnyx_rcs_webhook(payload, channel_id="rcs")

        assert message.metadata["suggestion_response"] == {
            "postback_data": "interested_yes",
            "text": "Yes, I'm interested",
        }

    def test_parse_location_sharing(self) -> None:
        """Parse a location sharing message."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "id": "msg-loc123",
                    "direction": "inbound",
                    "text": "",
                    "type": "RCS",
                    "from": {"phone_number": "+15145551111"},
                    "to": {"agent_id": "my-agent"},
                    "location": {
                        "latitude": 45.5017,
                        "longitude": -73.5673,
                    },
                },
            }
        }

        message = parse_telnyx_rcs_webhook(payload, channel_id="rcs")

        assert message.metadata["location"] == {
            "latitude": 45.5017,
            "longitude": -73.5673,
        }
