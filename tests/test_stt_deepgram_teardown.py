"""Cancelling a Deepgram stream must not surface as an unhandled error.

The sender task closes the stream in a ``finally``. On the cancellation path
the websocket is already gone, so the close raises — and an exception raised
in a ``finally`` during cancellation *replaces* the ``CancelledError`` the
caller is suppressing, which is how a Ctrl+C ends up printing a websocket
traceback at interpreter shutdown.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.voice.base import AudioChunk
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider


class _DeadConnection:
    """A connection whose socket died before the stream was closed."""

    def __init__(self) -> None:
        self.close_attempts = 0

    def on(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def start_listening(self) -> None:
        await asyncio.sleep(3600)

    async def send_media(self, _data: bytes) -> None:
        return None

    async def send_close_stream(self) -> None:
        self.close_attempts += 1
        raise ConnectionError("no close frame received or sent")

    async def __aenter__(self) -> _DeadConnection:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _FakeDeepgramModule:
    def __init__(self, connection: _DeadConnection) -> None:
        self._connection = connection

    def AsyncDeepgramClient(self, **_kwargs: Any) -> Any:  # noqa: N802 — SDK name
        connection = self._connection

        class _V1:
            @staticmethod
            def connect(**_opts: Any) -> _DeadConnection:
                return connection

        class _Listen:
            v1 = _V1()

        class _Client:
            listen = _Listen()

        return _Client()


@pytest.fixture
def provider() -> tuple[DeepgramSTTProvider, _DeadConnection]:
    connection = _DeadConnection()
    prov = DeepgramSTTProvider.__new__(DeepgramSTTProvider)
    prov._config = DeepgramConfig(api_key="test-key")
    prov._dg = _FakeDeepgramModule(connection)
    prov._client = None
    return prov, connection


async def test_cancelling_the_stream_raises_only_cancelled_error(
    provider: tuple[DeepgramSTTProvider, _DeadConnection],
) -> None:
    prov, connection = provider

    async def audio() -> Any:
        yield AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)
        await asyncio.sleep(3600)

    async def consume() -> None:
        async for _result in prov.transcribe_stream(audio()):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()

    # The caller suppresses CancelledError; anything else escapes to the event
    # loop and is reported as an unhandled exception during shutdown.
    with pytest.raises(asyncio.CancelledError):
        await task

    assert connection.close_attempts == 1, "the close was not attempted"
