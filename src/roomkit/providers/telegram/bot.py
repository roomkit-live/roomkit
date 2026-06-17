"""Telegram Bot provider — sends messages via the Telegram Bot API."""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import (
    AudioContent,
    LocationContent,
    MediaContent,
    RichContent,
    RoomEvent,
    VideoContent,
)
from roomkit.providers.telegram.base import TelegramProvider
from roomkit.providers.telegram.config import TelegramConfig
from roomkit.providers.utils import extract_event_text as _extract_event_text

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
        if isinstance(content, RichContent):
            return await self._send_rich(to, content)
        return await self._send_text(to, event)

    async def _send_text(self, to: str, event: RoomEvent) -> ProviderResult:
        text = self._extract_text(event)
        if not text:
            return ProviderResult(success=False, error="empty_message")
        return await self._api_call(
            "sendMessage",
            {"chat_id": to, "text": text},
        )

    async def _send_rich(self, to: str, content: RichContent) -> ProviderResult:
        """Send rich text with an optional inline keyboard.

        ``content.buttons`` is a list of ``{"text", "callback_data"}`` (or
        ``{"text", "url"}``) dicts; each becomes a single-button row in
        Telegram's ``inline_keyboard``. Buttons missing both an action are
        dropped. Plain text falls back to ``body`` when ``plain_text`` is unset.
        """
        text = content.plain_text or content.body
        if not text:
            return ProviderResult(success=False, error="empty_message")
        payload: dict[str, Any] = {"chat_id": to, "text": text}
        keyboard = self._inline_keyboard(content.buttons)
        if keyboard:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        return await self._api_call("sendMessage", payload)

    @staticmethod
    def _inline_keyboard(buttons: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        rows: list[list[dict[str, Any]]] = []
        for b in buttons:
            if not isinstance(b, dict) or not b.get("text"):
                continue
            btn: dict[str, Any] = {"text": b["text"]}
            if b.get("callback_data"):
                btn["callback_data"] = b["callback_data"]
            elif b.get("url"):
                btn["url"] = b["url"]
            else:
                continue
            rows.append([btn])
        return rows

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

    def verify_signature(
        self,
        payload: bytes,  # noqa: ARG002
        signature: str,
    ) -> bool:
        """Verify a Telegram webhook secret token.

        Telegram's ``setWebhook`` accepts a ``secret_token`` parameter.
        On each webhook request Telegram sends the token in the
        ``X-Telegram-Bot-Api-Secret-Token`` header.  Verification is a
        constant-time comparison of that header value against the
        configured :attr:`TelegramConfig.webhook_secret`.

        Args:
            payload: Raw request body bytes (unused).
            signature: Value of the ``X-Telegram-Bot-Api-Secret-Token`` header.

        Returns:
            True if the token matches, False otherwise.

        Raises:
            ValueError: If ``webhook_secret`` was not provided in config.
        """
        if not self._config.webhook_secret:
            raise ValueError(
                "webhook_secret must be provided in TelegramConfig for signature verification"
            )
        return hmac.compare_digest(
            signature,
            self._config.webhook_secret.get_secret_value(),
        )

    @staticmethod
    def _extract_text(event: RoomEvent) -> str:
        return _extract_event_text(event)

    async def close(self) -> None:
        await self._client.aclose()
