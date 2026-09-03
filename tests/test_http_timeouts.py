"""Every HTTP client outside ``roomkit.providers`` splits connect from read too.

The providers' own clients are covered by
``tests/test_providers/test_http_timeouts.py``; this file walks the sites the
first pass left out (RMK-149): Grok TTS, OpenAI vision, the WebSocket avatar
(three clients), the SSE source and Gemini TTS/STT. Each case builds the
object with distinctive values and reads back the ``timeout`` its client
actually received, so the test fails the day one passes the float again.

Gemini is read through the real SDK: google-genai flattens a per-request
``httpx.Timeout`` to its largest value, so the split only survives on the
httpx client RoomKit hands the SDK, and that client is what the SDK's
requests are driven through here.
"""

from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
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
        # The httpx client RoomKit hands the SDK (``HttpOptions.httpx_async_client``);
        # every request the SDK sends without its own timeout inherits it.
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


# ---------------------------------------------------------------------------
# Gemini, driven through the real SDK down to the httpx transport
# ---------------------------------------------------------------------------


class _RecordingTransport:
    """An ``httpx.AsyncClient`` whose transport records what each request
    carried, for the client RoomKit builds inside ``build_genai_client``."""

    seen: list[dict[str, Any]] = []

    @classmethod
    def client_class(cls) -> type[httpx.AsyncClient]:
        def handler(request: httpx.Request) -> httpx.Response:
            cls.seen.append(dict(request.extensions["timeout"]))
            return httpx.Response(200, json={}, headers={"content-type": "application/json"})

        class Client(httpx.AsyncClient):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(transport=httpx.MockTransport(handler), **kwargs)

        return Client


class TestGeminiThroughTheSDK:
    """The SDK flattens a per-request timeout and, with aiohttp importable,
    reuses ``async_client_args`` as aiohttp request kwargs. Handing it the
    httpx client is what keeps the split on the interactions path and keeps
    the Files API on httpx at all; both are asserted at the transport."""

    async def test_interactions_request_carries_the_split(self) -> None:
        pytest.importorskip("google.genai")
        _RecordingTransport.seen.clear()
        provider = GeminiTTSProvider(GeminiTTSConfig(api_key="k", **_TIMEOUTS))
        with patch("httpx.AsyncClient", _RecordingTransport.client_class()):
            # An empty interaction has no audio; the request went out regardless.
            with pytest.raises(RuntimeError):
                await provider.synthesize("hello")
            await provider.close()

        assert _RecordingTransport.seen == [
            {"connect": CONNECT, "read": TIMEOUT, "write": TIMEOUT, "pool": TIMEOUT}
        ]

    async def test_files_api_upload_goes_through_httpx_with_the_flat_budget(self) -> None:
        pytest.importorskip("google.genai")
        _RecordingTransport.seen.clear()
        config = GeminiSTTConfig(api_key="k", max_inline_bytes=16, **_TIMEOUTS)
        provider = GeminiSTTProvider(config)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "meeting.wav"
            wav.write_bytes(b"RIFF" + b"\x00" * 200)
            with patch("httpx.AsyncClient", _RecordingTransport.client_class()):
                # The fake upload answer carries no upload URL; the first
                # request of the resumable upload is what is asserted.
                with pytest.raises(Exception, match=r"[Uu]pload"):
                    await provider.transcribe_recording(wav)
                await provider.close()

        # The SDK's classic path takes one flat value per call (milliseconds),
        # which is the read budget; it cannot split the connect.
        assert _RecordingTransport.seen[:1] == [
            {"connect": TIMEOUT, "read": TIMEOUT, "write": TIMEOUT, "pool": TIMEOUT}
        ]
