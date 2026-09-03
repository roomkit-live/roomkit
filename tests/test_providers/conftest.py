"""Shared fixtures for the provider tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.providers.gemini import sdk as gemini_sdk


@pytest.fixture(autouse=True)
async def _close_genai_http_clients(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Close the httpx client of every Gemini provider a test builds.

    The chat, image and Vertex providers hand the SDK a real ``httpx.AsyncClient``
    in ``__init__``, mocked ``google.genai`` or not, and most tests here never
    call ``close()``: that client opens no socket, but it is still a resource
    left to the garbage collector. Recorded at the one place they are built.
    """
    built: list[gemini_sdk.GenaiClient] = []
    build = gemini_sdk._build_client

    def record(*args: Any, **kwargs: Any) -> gemini_sdk.GenaiClient:
        result = build(*args, **kwargs)
        built.append(result)
        return result

    monkeypatch.setattr(gemini_sdk, "_build_client", record)
    yield
    for result in built:
        if not result.http.is_closed:
            await result.http.aclose()
