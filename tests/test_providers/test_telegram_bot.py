"""Tests for the Telegram Bot provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from roomkit.models.event import (
    AudioContent,
    LocationContent,
    VideoContent,
)
from roomkit.providers.telegram import (
    TelegramBotProvider,
    TelegramConfig,
)
from tests.conftest import make_event, make_media_event


def _config(**overrides: Any) -> TelegramConfig:
    defaults: dict[str, Any] = {"bot_token": "123456:ABC-DEF"}
    defaults.update(overrides)
    return TelegramConfig(**defaults)


def _ok_response(message_id: int = 42) -> dict[str, Any]:
    return {
        "ok": True,
        "result": {
            "message_id": message_id,
            "chat": {"id": 100},
        },
    }


def _error_response(error_code: int = 401, description: str = "Unauthorized") -> dict[str, Any]:
    return {"ok": False, "error_code": error_code, "description": description}


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


class TestTelegramConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.timeout == 30.0

    def test_base_url(self) -> None:
        cfg = _config(bot_token="111:AAA")
        assert cfg.base_url == "https://api.telegram.org/bot111:AAA"


class TestTelegramBotProvider:
    @pytest.mark.asyncio
    async def test_send_text_success(self) -> None:
        transport = _MockTransport(_ok_response(42))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="Hello!")
        result = await provider.send(event, to="12345")

        assert result.success is True
        assert result.provider_message_id == "42"

        req = transport.requests[0]
        body = json.loads(req.content)
        assert body["chat_id"] == "12345"
        assert body["text"] == "Hello!"
        assert "/sendMessage" in str(req.url)

    @pytest.mark.asyncio
    async def test_send_empty_message(self) -> None:
        provider = TelegramBotProvider(_config())
        event = make_event(body="")
        result = await provider.send(event, to="12345")

        assert result.success is False
        assert result.error == "empty_message"

    @pytest.mark.asyncio
    async def test_send_photo(self) -> None:
        transport = _MockTransport(_ok_response(43))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/photo.jpg",
            mime_type="image/jpeg",
            caption="A photo",
        )
        result = await provider.send(event, to="12345")

        assert result.success is True
        req = transport.requests[0]
        body = json.loads(req.content)
        assert body["photo"] == "https://example.com/photo.jpg"
        assert body["caption"] == "A photo"
        assert "/sendPhoto" in str(req.url)

    @pytest.mark.asyncio
    async def test_send_document(self) -> None:
        transport = _MockTransport(_ok_response(44))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_media_event(
            url="https://example.com/file.pdf",
            mime_type="application/pdf",
        )
        result = await provider.send(event, to="12345")

        assert result.success is True
        req = transport.requests[0]
        body = json.loads(req.content)
        assert body["document"] == "https://example.com/file.pdf"
        assert "/sendDocument" in str(req.url)

    @pytest.mark.asyncio
    async def test_send_location(self) -> None:
        transport = _MockTransport(_ok_response(45))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        from roomkit.models.enums import ChannelType
        from roomkit.models.event import EventSource, RoomEvent

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.TELEGRAM),
            content=LocationContent(latitude=48.8566, longitude=2.3522),
        )
        result = await provider.send(event, to="12345")

        assert result.success is True
        req = transport.requests[0]
        body = json.loads(req.content)
        assert body["latitude"] == 48.8566
        assert body["longitude"] == 2.3522
        assert "/sendLocation" in str(req.url)

    @pytest.mark.asyncio
    async def test_send_video(self) -> None:
        transport = _MockTransport(_ok_response(46))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        from roomkit.models.enums import ChannelType
        from roomkit.models.event import EventSource, RoomEvent

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.TELEGRAM),
            content=VideoContent(url="https://example.com/video.mp4"),
        )
        result = await provider.send(event, to="12345")

        assert result.success is True
        req = transport.requests[0]
        body = json.loads(req.content)
        assert body["video"] == "https://example.com/video.mp4"
        assert "/sendVideo" in str(req.url)

    @pytest.mark.asyncio
    async def test_send_audio(self) -> None:
        transport = _MockTransport(_ok_response(47))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        from roomkit.models.enums import ChannelType
        from roomkit.models.event import EventSource, RoomEvent

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.TELEGRAM),
            content=AudioContent(url="https://example.com/audio.mp3"),
        )
        result = await provider.send(event, to="12345")

        assert result.success is True
        req = transport.requests[0]
        body = json.loads(req.content)
        assert body["audio"] == "https://example.com/audio.mp3"
        assert "/sendAudio" in str(req.url)

    @pytest.mark.asyncio
    async def test_send_api_error(self) -> None:
        transport = _MockTransport(_error_response(401, "Unauthorized"), status=401)
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = make_event(body="test")
        result = await provider.send(event, to="12345")

        assert result.success is False
        assert result.error == "telegram_401"
        assert result.metadata["description"] == "Unauthorized"

    @pytest.mark.asyncio
    async def test_send_timeout(self) -> None:
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=_TimeoutTransport())

        event = make_event(body="timeout")
        result = await provider.send(event, to="12345")

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        provider = TelegramBotProvider(_config())
        await provider.close()
        # Should not raise


class TestTelegramSignatureVerification:
    """Tests for TelegramBotProvider.verify_signature()."""

    def test_verify_valid_token(self) -> None:
        provider = TelegramBotProvider(_config(webhook_secret="my-secret"))

        assert provider.verify_signature(b"ignored", "my-secret") is True

    def test_verify_invalid_token(self) -> None:
        provider = TelegramBotProvider(_config(webhook_secret="my-secret"))

        assert provider.verify_signature(b"ignored", "wrong-token") is False

    def test_verify_no_webhook_secret(self) -> None:
        provider = TelegramBotProvider(_config())

        with pytest.raises(ValueError, match="webhook_secret must be provided"):
            provider.verify_signature(b"ignored", "any-token")
