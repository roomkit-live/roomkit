"""The google-genai client RoomKit's Gemini providers share.

Built here rather than in each provider because its timeouts have to be set
in one specific way for the connect/read split to survive the SDK:

* A per-request ``timeout`` is flattened by google-genai to its largest
  value (``BaseSDK._coerce_timeout_ms`` keeps ``max(connect, read, write,
  pool)`` and hands httpx one float again), so the split has to live on the
  httpx client itself.
* ``HttpOptions.async_client_args`` is not the place for it either: the SDK
  reuses those args as ``aiohttp.ClientSession.request()`` kwargs whenever
  aiohttp is importable (the ``twilio`` and ``gradium`` extras pull it in),
  and its Files API path then fails with "multiple values for keyword
  argument 'timeout'".
* Handing the SDK its own ``httpx.AsyncClient`` (``httpx_async_client``) is
  the SDK's own switch for "httpx, not aiohttp": every call, the Files API
  included, goes through that client with its timeout as the default, and
  httpx keeps applying the environment proxies and ``SSL_CERT_FILE`` it
  applies to any client it builds.

The SDK does not close a client it was given, so the caller owns both
objects and closes them together.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.providers.utils import HTTPTimeouts, http_timeout

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger("roomkit.providers.gemini.sdk")


def build_genai_client(
    api_key: str, timeouts: HTTPTimeouts, *, provider: str
) -> tuple[Any, httpx.AsyncClient]:
    """Return the ``genai.Client`` for *api_key* and the httpx client it runs on.

    ``timeouts.timeout`` is the read/write/pool budget, ``connect_timeout``
    bounds the TCP connect. *provider* names the caller in the ImportError
    raised when google-genai is not installed.
    """
    try:
        import httpx
        from google import genai
    except ImportError as exc:
        raise ImportError(
            f"google-genai is required for {provider}. "
            "Install it with: pip install roomkit[gemini]"
        ) from exc

    timeout = http_timeout(timeouts)
    # ``follow_redirects`` matches the client the SDK would build itself.
    http = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    client = genai.Client(
        api_key=api_key,
        http_options=genai.types.HttpOptions(
            # The sync client is built by the SDK regardless; same budget.
            client_args={"timeout": timeout},
            httpx_async_client=http,
        ),
    )
    return client, http


async def close_genai_client(client: Any, http: httpx.AsyncClient | None) -> None:
    """Close a pair from :func:`build_genai_client`; either may be ``None``.

    The SDK's own close is best effort (a transport already gone is not worth
    an exception); the httpx client it was given is closed regardless, since
    the SDK never closes one it did not build.
    """
    try:
        if client is not None:
            await client.aio.aclose()
    except Exception:  # pragma: no cover - transport already gone
        logger.debug("genai client close failed", exc_info=True)
    finally:
        if http is not None:
            await http.aclose()
