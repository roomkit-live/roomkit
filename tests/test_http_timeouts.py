"""Every HTTP client outside ``roomkit.providers`` splits connect from read too.

The providers' own clients are covered by
``tests/test_providers/test_http_timeouts.py``; this file walks the sites the
first pass left out (RMK-149): Grok TTS, OpenAI vision, the WebSocket avatar
(three clients), the SSE source and Gemini TTS/STT. Each case builds the
object with distinctive values and reads back the ``timeout`` its client
actually received, so the test fails the day one passes the float again.

Gemini is read through the real SDK: google-genai flattens a per-request
``httpx.Timeout`` to its largest value, so the split only survives on the
httpx client the SDK builds from ``HttpOptions``, and that is what is read.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from roomkit.sources import sse as sse_module
from roomkit.sources.sse import SSESource
from roomkit.video.avatar.websocket import WebSocketAvatarProvider
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider
from roomkit.voice.stt.gemini import GeminiSTTConfig, GeminiSTTProvider
from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider
from roomkit.voice.tts.grok import GrokTTSConfig, GrokTTSProvider
from tests.http_timeout_fakes import RecordingAsyncClient, read_and_close

# Distinctive on purpose: equal values would let a swapped connect/read pass.
TIMEOUT = 42.0
CONNECT = 3.0
_TIMEOUTS: dict[str, float] = {"timeout": TIMEOUT, "connect_timeout": CONNECT}
_AVATAR_URL = "http://avatar.example:8765"
_SSE_URL = "https://stream.example/events"

Builder = Callable[[], Awaitable[Any]]


async def _never_emit(message: Any) -> Any:
    raise AssertionError("the connect is made to fail before any event is read")


# ---------------------------------------------------------------------------
# Builders: one per client construction site outside ``providers/``
# ---------------------------------------------------------------------------


async def _grok_tts() -> Any:
    provider = GrokTTSProvider(GrokTTSConfig(api_key="k", **_TIMEOUTS))
    return await read_and_close(provider._get_client())


async def _openai_vision() -> Any:
    mod = MagicMock()
    with patch.dict("sys.modules", {"openai": mod}):
        OpenAIVisionProvider(OpenAIVisionConfig(**_TIMEOUTS))._get_client()
    return mod.AsyncOpenAI.call_args.kwargs["timeout"]


async def _avatar_start() -> Any:
    """The async client behind ``POST /start``."""
    provider = WebSocketAvatarProvider(_AVATAR_URL, **_TIMEOUTS)
    RecordingAsyncClient.calls.clear()
    with patch("httpx.AsyncClient", RecordingAsyncClient):
        await provider.start(b"png")
        timeout = RecordingAsyncClient.calls[-1]["timeout"]
        await provider.stop()
    return timeout


async def _avatar_sync() -> Any:
    """The sync client kept for the thread-pool calls (idle frame, restart)."""
    provider = WebSocketAvatarProvider(_AVATAR_URL, **_TIMEOUTS)
    with patch("httpx.AsyncClient", RecordingAsyncClient):
        await provider.start(b"png")
        assert provider._http is not None
        timeout = provider._http.timeout
        await provider.stop()
    return timeout


async def _avatar_stop() -> Any:
    """The async client behind ``POST /stop``."""
    provider = WebSocketAvatarProvider(_AVATAR_URL, **_TIMEOUTS)
    with patch("httpx.AsyncClient", RecordingAsyncClient):
        await provider.start(b"png")
        RecordingAsyncClient.calls.clear()
        await provider.stop()
    return RecordingAsyncClient.calls[-1]["timeout"]


async def _sse() -> Any:
    pytest.importorskip("httpx_sse")
    source = SSESource(_SSE_URL, channel_id="sse", **_TIMEOUTS)
    RecordingAsyncClient.calls.clear()
    with (
        patch("httpx.AsyncClient", RecordingAsyncClient),
        patch.object(sse_module, "aconnect_sse", side_effect=RuntimeError("stop here")),
        pytest.raises(RuntimeError, match="stop here"),
    ):
        await source.start(_never_emit)
    return RecordingAsyncClient.calls[-1]["timeout"]


async def _gemini_httpx_timeout(provider: GeminiTTSProvider | GeminiSTTProvider) -> Any:
    pytest.importorskip("google.genai")
    client = provider._get_client()
    try:
        # The SDK builds this client from ``HttpOptions.async_client_args``
        # and hands every interactions request ``USE_CLIENT_DEFAULT``.
        return client._api_client._async_httpx_client.timeout
    finally:
        await provider.close()


async def _gemini_tts() -> Any:
    config = GeminiTTSConfig(api_key="k", **_TIMEOUTS)
    return await _gemini_httpx_timeout(GeminiTTSProvider(config))


async def _gemini_stt() -> Any:
    config = GeminiSTTConfig(api_key="k", **_TIMEOUTS)
    return await _gemini_httpx_timeout(GeminiSTTProvider(config))


# (builder, expected read budget): the SSE source leaves its read side
# unbounded on purpose, so the stream survives idle periods between events.
CASES: dict[str, tuple[Builder, float | None]] = {
    "grok-tts": (_grok_tts, TIMEOUT),
    "openai-vision": (_openai_vision, TIMEOUT),
    "websocket-avatar-start": (_avatar_start, TIMEOUT),
    "websocket-avatar-sync": (_avatar_sync, TIMEOUT),
    "websocket-avatar-stop": (_avatar_stop, TIMEOUT),
    "sse": (_sse, None),
    "gemini-tts": (_gemini_tts, TIMEOUT),
    "gemini-stt": (_gemini_stt, TIMEOUT),
}


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("build", "read"), list(CASES.values()), ids=list(CASES))
async def test_client_timeout_splits_connect_from_read(build: Builder, read: float | None) -> None:
    timeout = await build()

    assert isinstance(timeout, httpx.Timeout), f"a bare {type(timeout).__name__} was passed"
    assert timeout.connect == CONNECT
    assert timeout.read == read
    assert timeout.write == TIMEOUT
    assert timeout.pool == TIMEOUT


# ---------------------------------------------------------------------------
# The default: the SDKs' own 5 s, on every config and constructor
# ---------------------------------------------------------------------------

CONFIGS: list[tuple[type[Any], dict[str, Any]]] = [
    (GrokTTSConfig, {"api_key": "k"}),
    (OpenAIVisionConfig, {}),
    (GeminiTTSConfig, {"api_key": "k"}),
    (GeminiSTTConfig, {"api_key": "k"}),
]


@pytest.mark.parametrize(("config_cls", "kwargs"), CONFIGS, ids=[c.__name__ for c, _ in CONFIGS])
def test_connect_timeout_defaults_to_five_seconds(
    config_cls: type[Any], kwargs: dict[str, Any]
) -> None:
    config = config_cls(**kwargs)

    assert config.connect_timeout == 5.0
    assert config.timeout > config.connect_timeout


def test_constructor_keywords_default_to_five_seconds() -> None:
    assert WebSocketAvatarProvider(_AVATAR_URL)._connect_timeout == 5.0
    assert SSESource(_SSE_URL, channel_id="sse")._connect_timeout == 5.0
