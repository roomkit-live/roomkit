"""Fakes shared by the connect/read timeout tests.

``tests/test_providers/test_http_timeouts.py`` covers every HTTP client under
``roomkit.providers``; ``tests/test_http_timeouts.py`` the ones outside it.
Both read back the ``timeout`` a client construction actually received.
"""

from __future__ import annotations

from typing import Any

import httpx


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"data": []}


class RecordingAsyncClient:
    """Stand-in for ``httpx.AsyncClient`` that records its constructor kwargs.

    For the clients opened inside a method (a throwaway ``GET /models``, the
    avatar's ``POST /start``), where nothing holds the client afterwards.
    """

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).calls.append(kwargs)

    async def __aenter__(self) -> RecordingAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()

    async def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()


async def read_and_close(client: httpx.AsyncClient) -> httpx.Timeout:
    try:
        return client.timeout
    finally:
        await client.aclose()
