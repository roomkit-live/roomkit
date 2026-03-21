"""Shared HTTP error handling for delivery providers."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import ProviderResult


def handle_http_error(exc: Exception, httpx_module: Any) -> ProviderResult:
    """Convert an httpx exception to a ProviderResult.

    Provides consistent error categorization across all delivery providers:
    - TimeoutException → "timeout"
    - HTTPStatusError → "http_{status_code}"
    - Other HTTPError → str(exc)
    """
    if isinstance(exc, httpx_module.TimeoutException):
        return ProviderResult(success=False, error="timeout")
    if isinstance(exc, httpx_module.HTTPStatusError):
        return ProviderResult(
            success=False,
            error=f"http_{exc.response.status_code}",
        )
    if isinstance(exc, httpx_module.HTTPError):
        return ProviderResult(success=False, error=str(exc))
    return ProviderResult(success=False, error=str(exc))
