"""Generic HTTP webhook provider — POSTs events to a configurable URL."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.http.base import HTTPProvider
from roomkit.providers.http.config import HTTPProviderConfig

if TYPE_CHECKING:
    import httpx


class WebhookHTTPProvider(HTTPProvider):
    """HTTP provider that POSTs JSON payloads to a webhook URL."""

    def __init__(self, config: HTTPProviderConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for WebhookHTTPProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(
            timeout=config.timeout,
        )

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        text = self._extract_text(event)
        if not text:
            return ProviderResult(success=False, error="empty_message")

        payload = self._build_payload(event, to, text)
        body = json.dumps(payload)
        headers = self._build_headers(body)

        try:
            resp = await self._client.post(
                self._config.webhook_url,
                content=body,
                headers=headers,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            return ProviderResult(
                success=False,
                error=f"http_{exc.response.status_code}",
            )
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        return self._parse_response(data)

    def _build_payload(self, event: RoomEvent, to: str, text: str) -> dict[str, Any]:
        return {
            "recipient_id": to,
            "channel_id": event.source.channel_id,
            "room_id": event.room_id,
            "content": {"type": "text", "body": text},
            "metadata": event.metadata or {},
        }

    def _build_headers(self, body: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self._config.headers,
        }
        if self._config.secret is not None:
            signature = hmac.new(
                self._config.secret.get_secret_value().encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-RoomKit-Signature"] = signature
        return headers

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ProviderResult:
        return ProviderResult(
            success=True,
            provider_message_id=data.get("message_id"),
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
