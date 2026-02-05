"""Tests for the generic HTTP webhook provider."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import httpx

from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.providers.http.config import HTTPProviderConfig
from roomkit.providers.http.provider import WebhookHTTPProvider
from roomkit.providers.http.webhook import parse_http_webhook


def _make_event(body: str = "hello") -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ch1", channel_type=ChannelType.WEBHOOK),
        content=TextContent(body=body),
    )


class TestWebhookHTTPProvider:
    async def test_send_success(self) -> None:
        config = HTTPProviderConfig(webhook_url="https://example.com/hook")
        provider = WebhookHTTPProvider(config)

        mock_response = httpx.Response(
            200,
            json={"success": True, "message_id": "msg-001"},
            request=httpx.Request("POST", "https://example.com/hook"),
        )
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.send(_make_event(), to="user-123")

        assert result.success is True
        assert result.provider_message_id == "msg-001"

        call_kwargs = provider._client.post.call_args
        body = json.loads(call_kwargs.kwargs["content"])
        assert body["recipient_id"] == "user-123"
        assert body["channel_id"] == "ch1"
        assert body["room_id"] == "r1"
        assert body["content"]["type"] == "text"
        assert body["content"]["body"] == "hello"

    async def test_send_with_headers(self) -> None:
        config = HTTPProviderConfig(
            webhook_url="https://example.com/hook",
            headers={"Authorization": "Bearer tok123"},
        )
        provider = WebhookHTTPProvider(config)

        mock_response = httpx.Response(
            200,
            json={"success": True},
            request=httpx.Request("POST", "https://example.com/hook"),
        )
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        await provider.send(_make_event(), to="user-123")

        call_kwargs = provider._client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer tok123"
        assert headers["Content-Type"] == "application/json"

    async def test_send_with_hmac_signature(self) -> None:
        secret = "my-secret-key"
        config = HTTPProviderConfig(
            webhook_url="https://example.com/hook",
            secret=secret,
        )
        provider = WebhookHTTPProvider(config)

        mock_response = httpx.Response(
            200,
            json={"success": True},
            request=httpx.Request("POST", "https://example.com/hook"),
        )
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        await provider.send(_make_event(), to="user-123")

        call_kwargs = provider._client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        body = call_kwargs.kwargs["content"]

        expected_sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        assert headers["X-RoomKit-Signature"] == expected_sig

    async def test_send_empty_message(self) -> None:
        config = HTTPProviderConfig(webhook_url="https://example.com/hook")
        provider = WebhookHTTPProvider(config)

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.WEBHOOK),
            content=TextContent(body=""),
        )
        result = await provider.send(event, to="user-123")

        assert result.success is False
        assert result.error == "empty_message"

    async def test_send_timeout(self) -> None:
        config = HTTPProviderConfig(webhook_url="https://example.com/hook")
        provider = WebhookHTTPProvider(config)

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        result = await provider.send(_make_event(), to="user-123")

        assert result.success is False
        assert result.error == "timeout"

    async def test_send_http_error(self) -> None:
        config = HTTPProviderConfig(webhook_url="https://example.com/hook")
        provider = WebhookHTTPProvider(config)

        mock_response = httpx.Response(
            500,
            request=httpx.Request("POST", "https://example.com/hook"),
        )
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=mock_response.request, response=mock_response
            )
        )

        result = await provider.send(_make_event(), to="user-123")

        assert result.success is False
        assert result.error == "http_500"

    async def test_config_defaults(self) -> None:
        config = HTTPProviderConfig(webhook_url="https://example.com/hook")
        assert config.webhook_url == "https://example.com/hook"
        assert config.secret is None
        assert config.timeout == 30.0
        assert config.headers == {}


class TestParseHTTPWebhook:
    def test_parse_webhook(self) -> None:
        payload = {
            "sender_id": "user-123",
            "body": "Hello!",
            "external_id": "msg-456",
            "metadata": {"source": "web"},
        }
        msg = parse_http_webhook(payload, channel_id="http1")

        assert msg.channel_id == "http1"
        assert msg.sender_id == "user-123"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello!"
        assert msg.external_id == "msg-456"
        assert msg.idempotency_key == "msg-456"
        assert msg.metadata == {"source": "web"}

    def test_parse_webhook_minimal(self) -> None:
        payload = {"sender_id": "user-123", "body": "Hi"}
        msg = parse_http_webhook(payload, channel_id="http1")

        assert msg.sender_id == "user-123"
        assert msg.content.body == "Hi"  # type: ignore[union-attr]
        assert msg.external_id is None
        assert msg.metadata == {}

    def test_parse_webhook_with_metadata(self) -> None:
        payload = {
            "sender_id": "user-999",
            "body": "Full payload",
            "external_id": "ext-1",
            "metadata": {"priority": "high", "tags": ["urgent"]},
        }
        msg = parse_http_webhook(payload, channel_id="http1")

        assert msg.metadata["priority"] == "high"
        assert msg.metadata["tags"] == ["urgent"]
        assert msg.external_id == "ext-1"
