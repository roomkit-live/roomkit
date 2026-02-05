"""Elastic Email provider — sends emails via the Elastic Email v2 API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RichContent, RoomEvent, TextContent
from roomkit.providers.elasticemail.config import ElasticEmailConfig
from roomkit.providers.email.base import EmailProvider

if TYPE_CHECKING:
    import httpx


class ElasticEmailProvider(EmailProvider):
    """Send-only email provider using the Elastic Email REST API."""

    def __init__(self, config: ElasticEmailConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for ElasticEmailProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(
            timeout=config.timeout,
        )

    async def send(
        self,
        event: RoomEvent,
        to: str,
        from_: str | None = None,
        subject: str | None = None,
    ) -> ProviderResult:
        body_text, body_html = self._extract_body(event)
        if not body_text and not body_html:
            return ProviderResult(success=False, error="empty_message")

        data: dict[str, Any] = {
            "apikey": self._config.api_key.get_secret_value(),
            "from": from_ or self._config.from_email,
            "to": to,
            "subject": subject or "",
            "bodyText": body_text,
            "bodyHtml": body_html,
            "isTransactional": str(self._config.is_transactional).lower(),
        }
        if self._config.from_name:
            data["fromName"] = self._config.from_name

        try:
            resp = await self._client.post(self._config.base_url, data=data)
            resp.raise_for_status()
            result = resp.json()
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            return ProviderResult(
                success=False,
                error=f"http_{exc.response.status_code}",
            )
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        return self._parse_response(result)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ProviderResult:
        if data.get("success"):
            return ProviderResult(
                success=True,
                provider_message_id=data.get("data", {}).get("transactionid"),
            )
        return ProviderResult(
            success=False,
            error=data.get("error", "unknown_error"),
        )

    @staticmethod
    def _extract_body(event: RoomEvent) -> tuple[str, str]:
        """Return ``(body_text, body_html)`` from the event content."""
        content = event.content
        if isinstance(content, RichContent):
            return "", content.body
        if isinstance(content, TextContent):
            return content.body, ""
        if hasattr(content, "body"):
            return str(content.body), ""
        return "", ""

    async def close(self) -> None:
        await self._client.aclose()
