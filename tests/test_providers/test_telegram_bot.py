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


class _MethodTransport(httpx.AsyncBaseTransport):
    """Succeeds for every Bot API method except those named in ``errors``."""

    def __init__(self, errors: dict[str, int] | None = None) -> None:
        self._errors = errors or {}
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, code in self._errors.items():
            if f"/{method}" in str(request.url):
                return httpx.Response(code, json=_error_response(code), request=request)
        return httpx.Response(200, json=_ok_response(70), request=request)


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


class TestTelegramMarkdown:
    """The model emits CommonMark; it must reach Telegram as formatting, not syntax."""

    @pytest.mark.asyncio
    async def test_markdown_renders_to_entities(self) -> None:
        transport = _MockTransport(_ok_response(50))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        md = "## Titre\n\n| A | B |\n|---|---|\n| x | y |\n\n**gras** et du texte."
        result = await provider.send(make_event(body=md), to="999")

        assert result.success is True
        assert len(transport.requests) == 1
        body = json.loads(transport.requests[0].content)
        assert body["chat_id"] == "999"
        # Markdown markers are gone — formatting rides on entities, not syntax.
        assert "##" not in body["text"]
        assert "**" not in body["text"]
        # Entities (not parse_mode) carry the formatting, so Telegram never re-parses.
        assert "parse_mode" not in body
        types = {e["type"] for e in body["entities"]}
        assert "pre" in types  # the table became a monospace block
        assert types & {"bold", "underline"}  # the heading/bold became real formatting

    @pytest.mark.asyncio
    async def test_long_message_splits_into_chunks(self) -> None:
        transport = _MockTransport(_ok_response(51))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        long_body = "\n\n".join(f"Paragraphe {i} " + "mot " * 40 for i in range(120))
        result = await provider.send(make_event(body=long_body), to="1")

        assert result.success is True
        assert result.provider_message_id == "51"
        assert len(transport.requests) > 1
        for req in transport.requests:
            assert "/sendMessage" in str(req.url)
            assert len(json.loads(req.content)["text"]) <= 4096

    @pytest.mark.asyncio
    async def test_falls_back_to_plain_when_formatter_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = _MockTransport(_ok_response(52))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        async def _unavailable(_text: str) -> None:
            return None

        monkeypatch.setattr(provider, "_telegramify", _unavailable)

        md = "## Titre\n\nLe **corps**."
        result = await provider.send(make_event(body=md), to="7")

        assert result.success is True
        body = json.loads(transport.requests[0].content)
        assert body["text"] == md  # raw markdown preserved, nothing dropped
        assert "entities" not in body

    @pytest.mark.asyncio
    async def test_short_code_block_stays_inline(self) -> None:
        transport = _MockTransport(_ok_response(54))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        md = "Exemple:\n```python\nprint('hi')\n```\nVoilà."
        result = await provider.send(make_event(body=md), to="5")

        assert result.success is True
        # A small snippet renders inline, not as a file download.
        assert len(transport.requests) == 1
        req = transport.requests[0]
        assert "/sendDocument" not in str(req.url)
        body = json.loads(req.content)
        assert any(e["type"] in {"pre", "code"} for e in body["entities"])

    @pytest.mark.asyncio
    async def test_large_code_block_sent_as_document(self) -> None:
        transport = _MockTransport(_ok_response(55))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        code = "\n".join(f"line_{i} = {i}" for i in range(80))
        md = f"Code:\n```python\n{code}\n```\nFin."
        result = await provider.send(make_event(body=md), to="6")

        assert result.success is True
        doc_reqs = [r for r in transport.requests if "/sendDocument" in str(r.url)]
        assert len(doc_reqs) == 1
        # A large dump is uploaded as multipart file bytes, not a JSON text body.
        assert "multipart/form-data" in doc_reqs[0].headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_rich_content_renders_entities_with_keyboard(self) -> None:
        from roomkit.models.enums import ChannelType
        from roomkit.models.event import EventSource, RichContent, RoomEvent

        transport = _MockTransport(_ok_response(53))
        provider = TelegramBotProvider(_config())
        provider._client = httpx.AsyncClient(transport=transport)

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.TELEGRAM),
            content=RichContent(
                body="**Approuver** la demande ?",
                buttons=[{"text": "Oui", "callback_data": "ok"}],
            ),
        )
        result = await provider.send(event, to="3")

        assert result.success is True
        body = json.loads(transport.requests[0].content)
        assert "**" not in body["text"]
        assert any(e["type"] == "bold" for e in body["entities"])
        assert body["reply_markup"]["inline_keyboard"][0][0]["text"] == "Oui"


class TestTelegramRichMessages:
    """Opt-in Bot API 10.1 Rich Messages, with fallback to entity formatting."""

    TABLE_MD = "## T\n\n| A | B |\n|---|---|\n| 1 | 2 |"

    @pytest.mark.asyncio
    async def test_uses_send_rich_message_when_enabled(self) -> None:
        transport = _MethodTransport()
        provider = TelegramBotProvider(_config(rich_messages=True))
        provider._client = httpx.AsyncClient(transport=transport)

        result = await provider.send(make_event(body=self.TABLE_MD), to="9")

        assert result.success is True
        urls = [str(r.url) for r in transport.requests]
        assert any("/sendRichMessage" in u for u in urls)
        assert not any("/sendMessage" in u for u in urls)
        body = json.loads(transport.requests[0].content)
        # The table reaches Telegram as a native rich block, not monospace text.
        assert "<table>" in body["rich_message"]["html"]

    @pytest.mark.asyncio
    async def test_default_does_not_use_rich_messages(self) -> None:
        transport = _MethodTransport()
        provider = TelegramBotProvider(_config())  # rich_messages defaults False
        provider._client = httpx.AsyncClient(transport=transport)

        result = await provider.send(make_event(body=self.TABLE_MD), to="9")

        assert result.success is True
        assert not any("/sendRichMessage" in str(r.url) for r in transport.requests)

    @pytest.mark.asyncio
    async def test_falls_back_to_entities_on_api_error(self) -> None:
        transport = _MethodTransport(errors={"sendRichMessage": 400})
        provider = TelegramBotProvider(_config(rich_messages=True))
        provider._client = httpx.AsyncClient(transport=transport)

        result = await provider.send(make_event(body=self.TABLE_MD), to="9")

        assert result.success is True
        urls = [str(r.url) for r in transport.requests]
        assert any("/sendRichMessage" in u for u in urls)  # tried rich first
        assert any("/sendMessage" in u for u in urls)  # then fell back, nothing lost

    @pytest.mark.asyncio
    async def test_falls_back_when_converter_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import telegramify_markdown

        def _boom(_text: str, **_kw: Any) -> list[Any]:
            raise RuntimeError("converter down")

        monkeypatch.setattr(telegramify_markdown, "telegramify_rich", _boom)
        transport = _MethodTransport()
        provider = TelegramBotProvider(_config(rich_messages=True))
        provider._client = httpx.AsyncClient(transport=transport)

        result = await provider.send(make_event(body=self.TABLE_MD), to="9")

        assert result.success is True
        urls = [str(r.url) for r in transport.requests]
        assert not any("/sendRichMessage" in u for u in urls)
        assert any("/sendMessage" in u for u in urls)


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
