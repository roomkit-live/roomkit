"""Telegram Bot provider — sends messages via the Telegram Bot API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import (
    AudioContent,
    LocationContent,
    MediaContent,
    RoomEvent,
    TextContent,
    VideoContent,
)
from roomkit.providers.telegram.base import TelegramProvider
from roomkit.providers.telegram.config import TelegramConfig

if TYPE_CHECKING:
    import httpx


class TelegramBotProvider(TelegramProvider):
    """Send messages via the Telegram Bot API."""

    def __init__(self, config: TelegramConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for TelegramBotProvider. "
                "Install it with: pip install roomkit[telegram]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(
            timeout=config.timeout,
        )

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        content = event.content
        if isinstance(content, LocationContent):
            return await self._send_location(to, content)
        if isinstance(content, VideoContent):
            return await self._send_video(to, content)
        if isinstance(content, AudioContent):
            return await self._send_audio(to, content)
        if isinstance(content, MediaContent):
            return await self._send_media(to, content)
        return await self._send_text(to, event)

    async def _send_text(self, to: str, event: RoomEvent) -> ProviderResult:
        text = self._extract_text(event)
        if not text:
            return ProviderResult(success=False, error="empty_message")
        return await self._api_call(
            "sendMessage",
            {"chat_id": to, "text": text},
        )

    async def _send_media(self, to: str, content: MediaContent) -> ProviderResult:
        mime = content.mime_type
        if mime.startswith("image/"):
            method = "sendPhoto"
            payload: dict[str, Any] = {
                "chat_id": to,
                "photo": content.url,
            }
            if content.caption:
                payload["caption"] = content.caption
        else:
            method = "sendDocument"
            payload = {
                "chat_id": to,
                "document": content.url,
            }
            if content.caption:
                payload["caption"] = content.caption
        return await self._api_call(method, payload)

    async def _send_location(self, to: str, content: LocationContent) -> ProviderResult:
        return await self._api_call(
            "sendLocation",
            {
                "chat_id": to,
                "latitude": content.latitude,
                "longitude": content.longitude,
            },
        )

    async def _send_video(self, to: str, content: VideoContent) -> ProviderResult:
        payload: dict[str, Any] = {
            "chat_id": to,
            "video": content.url,
        }
        return await self._api_call("sendVideo", payload)

    async def _send_audio(self, to: str, content: AudioContent) -> ProviderResult:
        payload: dict[str, Any] = {
            "chat_id": to,
            "audio": content.url,
        }
        return await self._api_call("sendAudio", payload)

    async def _api_call(self, method: str, payload: dict[str, Any]) -> ProviderResult:
        url = f"{self._config.base_url}/{method}"
        try:
            import time

            t0 = time.monotonic()
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            send_ms = (time.monotonic() - t0) * 1000
            data = resp.json()

            from roomkit.telemetry.noop import NoopTelemetryProvider

            _tel = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            _tel.record_metric(
                "roomkit.delivery.send_ms",
                send_ms,
                unit="ms",
                attributes={"provider": "TelegramBotProvider"},
            )
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            return self._parse_error(exc)
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        result = data.get("result", {})
        return ProviderResult(
            success=True,
            provider_message_id=str(result.get("message_id", "")),
        )

    @staticmethod
    def _parse_error(exc: Any) -> ProviderResult:
        """Extract a Telegram Bot API error when available."""
        try:
            body = exc.response.json()
            error_code = body.get("error_code", exc.response.status_code)
            description = body.get("description", "")
            return ProviderResult(
                success=False,
                error=f"telegram_{error_code}",
                metadata={"description": description},
            )
        except Exception:
            return ProviderResult(
                success=False,
                error=f"http_{exc.response.status_code}",
            )

    @staticmethod
    def _extract_text(event: RoomEvent) -> str:
        content = event.content
        if isinstance(content, TextContent):
            return content.body
        if hasattr(content, "body"):
            return str(content.body)
        return ""

    async def close(self) -> None:
        await self._client.aclose()
