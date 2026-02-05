"""Tests for the SendGrid provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from roomkit.models.event import EventSource, RichContent, RoomEvent
from roomkit.providers.sendgrid import SendGridConfig, SendGridProvider
from tests.conftest import make_event


def _config(**overrides: Any) -> SendGridConfig:
    defaults: dict[str, Any] = {
        "api_key": "sg-secret-key",
        "from_email": "noreply@example.com",
    }
    defaults.update(overrides)
    return SendGridConfig(**defaults)


class _MockTransport(httpx.AsyncBaseTransport):
    """Returns a 202 Accepted with X-Message-Id header (SendGrid success)."""

    def __init__(self, message_id: str = "msg-123", status_code: int = 202) -> None:
        self._message_id = message_id
        self._status_code = status_code
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status_code,
            headers={"X-Message-Id": self._message_id},
            request=request,
        )


class _ErrorTransport(httpx.AsyncBaseTransport):
    """Returns a JSON error response."""

    def __init__(self, status_code: int = 400, error_message: str = "Bad request") -> None:
        self._status_code = status_code
        self._error_message = error_message

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = {"errors": [{"message": self._error_message}]}
        resp = httpx.Response(
            self._status_code,
            json=body,
            request=request,
        )
        resp.raise_for_status()
        return resp  # pragma: no cover


class _RateLimitTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request)


class _TimeoutTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")


class TestSendGridConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.from_email == "noreply@example.com"
        assert cfg.base_url == "https://api.sendgrid.com/v3/mail/send"
        assert cfg.timeout == 30.0
        assert cfg.from_name is None

    def test_custom_values(self) -> None:
        cfg = _config(from_name="RoomKit", timeout=10.0)
        assert cfg.from_name == "RoomKit"
        assert cfg.timeout == 10.0


class TestSendGridProvider:
    @pytest.mark.asyncio
    async def test_send_success(self) -> None:
        transport = _MockTransport("msg-42")
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello!")
        result = await provider.send(event, to="user@example.com", subject="Hi")

        assert result.success is True
        assert result.provider_message_id == "msg-42"

        # Verify JSON payload structure
        body = json.loads(transport.requests[0].content.decode())
        assert body["personalizations"] == [{"to": [{"email": "user@example.com"}]}]
        assert body["from"] == {"email": "noreply@example.com"}
        assert body["subject"] == "Hi"
        assert body["content"] == [{"type": "text/plain", "value": "Hello!"}]

    @pytest.mark.asyncio
    async def test_send_bearer_auth(self) -> None:
        transport = _MockTransport()
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="user@example.com")

        auth_header = transport.requests[0].headers["authorization"]
        assert auth_header == "Bearer sg-secret-key"

    @pytest.mark.asyncio
    async def test_send_passes_from_name(self) -> None:
        transport = _MockTransport()
        provider = SendGridProvider(_config(from_name="RoomKit"))
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="user@example.com")

        body = json.loads(transport.requests[0].content.decode())
        assert body["from"] == {"email": "noreply@example.com", "name": "RoomKit"}

    @pytest.mark.asyncio
    async def test_send_overrides_from_address(self) -> None:
        transport = _MockTransport()
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="a@b.com", from_="custom@example.com")

        body = json.loads(transport.requests[0].content.decode())
        assert body["from"]["email"] == "custom@example.com"

    @pytest.mark.asyncio
    async def test_send_empty_message(self) -> None:
        provider = SendGridProvider(_config())
        event = make_event(body="")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "empty_message"

    @pytest.mark.asyncio
    async def test_send_rich_content_as_html(self) -> None:
        transport = _MockTransport()
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type="email"),
            content=RichContent(body="<h1>Hello</h1>"),
        )
        result = await provider.send(event, to="user@example.com")

        assert result.success is True
        body = json.loads(transport.requests[0].content.decode())
        assert body["content"] == [{"type": "text/html", "value": "<h1>Hello</h1>"}]

    @pytest.mark.asyncio
    async def test_send_api_error(self) -> None:
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(
            transport=_ErrorTransport(400, "Invalid email address"),
        )

        event = make_event(body="test")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "Invalid email address"

    @pytest.mark.asyncio
    async def test_send_rate_limited(self) -> None:
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(transport=_RateLimitTransport())

        event = make_event(body="test")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "rate_limited"

    @pytest.mark.asyncio
    async def test_send_timeout(self) -> None:
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_send_http_status_error(self) -> None:
        provider = SendGridProvider(_config())
        provider._client = httpx.AsyncClient(
            transport=_ErrorTransport(503, "Service unavailable"),
        )

        event = make_event(body="fail")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "Service unavailable"
