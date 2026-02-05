"""Tests for the Elastic Email provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from roomkit.models.event import RichContent
from roomkit.providers.elasticemail import ElasticEmailConfig, ElasticEmailProvider
from tests.conftest import make_event


def _config(**overrides: Any) -> ElasticEmailConfig:
    defaults: dict[str, Any] = {
        "api_key": "secret-key",
        "from_email": "noreply@example.com",
    }
    defaults.update(overrides)
    return ElasticEmailConfig(**defaults)


def _success_response(txn_id: str = "txn-123") -> dict[str, Any]:
    return {"success": True, "data": {"transactionid": txn_id}}


def _error_response(error: str = "Unauthorized") -> dict[str, Any]:
    return {"success": False, "error": error}


class _MockTransport(httpx.AsyncBaseTransport):
    """Captures requests and returns a canned JSON response."""

    def __init__(self, response_data: dict[str, Any]) -> None:
        self._data = response_data
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self._data, request=request)


class _TimeoutTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")


class _StatusErrorTransport(httpx.AsyncBaseTransport):
    def __init__(self, status_code: int = 500) -> None:
        self._status_code = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        resp = httpx.Response(self._status_code, request=request, text="error")
        resp.raise_for_status()
        return resp  # pragma: no cover


class TestElasticEmailConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.from_email == "noreply@example.com"
        assert cfg.is_transactional is True
        assert cfg.base_url == "https://api.elasticemail.com/v2/email/send"
        assert cfg.timeout == 30.0
        assert cfg.from_name is None

    def test_custom_values(self) -> None:
        cfg = _config(from_name="RoomKit", is_transactional=False)
        assert cfg.from_name == "RoomKit"
        assert cfg.is_transactional is False


class TestElasticEmailProvider:
    @pytest.mark.asyncio
    async def test_send_success(self) -> None:
        transport = _MockTransport(_success_response("txn-42"))
        provider = ElasticEmailProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello!")
        result = await provider.send(event, to="user@example.com", subject="Hi")

        assert result.success is True
        assert result.provider_message_id == "txn-42"

        body = transport.requests[0].content.decode()
        assert "noreply%40example.com" in body or "noreply@example.com" in body
        assert "user%40example.com" in body or "user@example.com" in body

    @pytest.mark.asyncio
    async def test_send_passes_from_name(self) -> None:
        transport = _MockTransport(_success_response())
        provider = ElasticEmailProvider(_config(from_name="RoomKit"))
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="user@example.com")

        body = transport.requests[0].content.decode()
        assert "fromName=RoomKit" in body

    @pytest.mark.asyncio
    async def test_send_empty_message(self) -> None:
        provider = ElasticEmailProvider(_config())
        event = make_event(body="")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "empty_message"

    @pytest.mark.asyncio
    async def test_send_api_error(self) -> None:
        transport = _MockTransport(_error_response("Invalid API key"))
        provider = ElasticEmailProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "Invalid API key"

    @pytest.mark.asyncio
    async def test_send_timeout(self) -> None:
        provider = ElasticEmailProvider(_config())
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_send_http_status_error(self) -> None:
        provider = ElasticEmailProvider(_config())
        provider._client = httpx.AsyncClient(transport=_StatusErrorTransport(503))

        event = make_event(body="fail")
        result = await provider.send(event, to="user@example.com")

        assert result.success is False
        assert result.error == "http_503"

    @pytest.mark.asyncio
    async def test_send_rich_content_as_html(self) -> None:
        transport = _MockTransport(_success_response())
        provider = ElasticEmailProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        from roomkit.models.event import EventSource, RoomEvent

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type="email"),
            content=RichContent(body="<h1>Hello</h1>"),
        )
        result = await provider.send(event, to="user@example.com")

        assert result.success is True
        body = transport.requests[0].content.decode()
        assert "bodyHtml" in body

    @pytest.mark.asyncio
    async def test_send_overrides_from_address(self) -> None:
        transport = _MockTransport(_success_response())
        provider = ElasticEmailProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hi")
        await provider.send(event, to="a@b.com", from_="custom@example.com")

        body = transport.requests[0].content.decode()
        assert "custom%40example.com" in body or "custom@example.com" in body
