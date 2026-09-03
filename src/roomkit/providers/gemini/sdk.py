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
* Even then the SDK's classic request path (streamed generation,
  ``models.list``, the Files API) hands httpx ``timeout=None`` unless
  ``HttpOptions.timeout`` is set, and httpx reads an explicit ``None`` as
  "no timeout at all" rather than "the client's default"; only the
  Interactions API path leaves the default in place. A request hook on the
  client puts the budget back on any request that names none. One that
  names its own (the Files API: one flat value the SDK spreads over the
  connect too) keeps its read budget and takes the client's connect, the
  part of the budget the SDK cannot split.

The SDK does not close a client it was given, so the caller owns both
objects and closes them together.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.providers.utils import HTTPTimeouts, http_timeout

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger("roomkit.providers.gemini.sdk")


def _restore_client_timeout(timeout: httpx.Timeout) -> Callable[[httpx.Request], Awaitable[None]]:
    """A request hook keeping the client's budget on what the SDK sends.

    A request that names no budget at all (``timeout=None``, the SDK's
    classic path) gets the client's whole budget. One that names its own
    keeps its read budget and has its connect capped at the client's: the
    SDK's per-request value is one float spread over the connect as well,
    where ``connect_timeout`` is the ceiling by definition. A request already
    at or under it (the Interactions path, which leaves the client default in
    place) is not touched.
    """
    budget = timeout.as_dict()
    ceiling = budget["connect"]

    async def restore(request: httpx.Request) -> None:
        current = request.extensions.get("timeout")
        if current is None or all(value is None for value in current.values()):
            request.extensions["timeout"] = dict(budget)
            return
        connect = current.get("connect")
        if ceiling is not None and (connect is None or connect > ceiling):
            request.extensions["timeout"] = {**current, "connect": ceiling}

    return restore


def build_genai_client(
    timeouts: HTTPTimeouts, *, provider: str, **client_kwargs: Any
) -> tuple[Any, httpx.AsyncClient]:
    """Return a ``genai.Client`` and the httpx client it runs on.

    ``timeouts.timeout`` is the read/write/pool budget, ``connect_timeout``
    bounds the TCP connect. *provider* names the caller in the ImportError
    raised when google-genai is not installed. *client_kwargs* reach
    ``genai.Client`` as they are: ``api_key`` for the Developer API;
    ``vertexai``, ``project``, ``location`` and ``credentials`` for Vertex,
    where the identity is never a key on the request.
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
    # ``follow_redirects`` matches the client the SDK would build itself. Built
    # first because the SDK takes it as an argument; should ``genai.Client``
    # then reject the key, a client that never sent a request holds nothing
    # to release.
    http = httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"request": [_restore_client_timeout(timeout)]},
    )
    client = genai.Client(
        **client_kwargs,
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
    an exception) and takes two calls, ``aclose`` leaving the sync httpx
    client the SDK builds regardless to ``close``. The httpx client it was
    given is closed whatever happens, since the SDK never closes one it did
    not build.
    """
    try:
        if client is not None:
            await client.aio.aclose()
            client.close()
    except Exception:
        logger.debug("genai client close failed", exc_info=True)
    finally:
        if http is not None:
            await http.aclose()
