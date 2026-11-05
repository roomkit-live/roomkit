"""Rime STT over its actual WebSocket envelope, against a local service."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from roomkit.voice.base import AudioChunk
from roomkit.voice.stt import RimeSTTConfig, RimeSTTError, RimeSTTProvider
from tests.voice.rime_wire import receive, send, server


def provider(url: str, **kwargs: Any) -> RimeSTTProvider:
    return RimeSTTProvider(RimeSTTConfig(api_key="test-key", websocket_url=url, **kwargs))


async def chunks(*items: AudioChunk) -> AsyncIterator[AudioChunk]:
    for item in items:
        yield item


async def test_partials_arrive_before_input_ends_and_final_uses_reported_language() -> None:
    heard_partial = asyncio.Event()
    requests = []

    async def handler(ws: ServerConnection) -> None:
        requests.append(await receive(ws))
        await send(ws, started={"requestId": "stt-1"})
        assert "audio" in await receive(ws)
        await send(ws, hypothesis={"text": "Hola", "revision": "1"})
        assert await receive(ws) == {"end": {}}
        await send(ws, hypothesis={"text": "Hola mundo", "revision": "2"})
        await send(ws, done={"text": "Hola mundo", "language": {"tag": "es"}})
        await ws.wait_closed()

    async def source() -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"\0\0" * 160)
        await asyncio.wait_for(heard_partial.wait(), 1)

    async with server(handler) as url:
        stt = provider(url, language="en", mode="verbatim", context_terms=["RoomKit"])
        results = []
        async for result in stt.transcribe_stream(source(), language="es"):
            results.append(result)
            if not result.is_final:
                heard_partial.set()
        await stt.close()
    assert [(r.text, r.is_final) for r in results] == [
        ("Hola", False),
        ("Hola mundo", False),
        ("Hola mundo", True),
    ]
    assert results[-1].language == "es"
    assert results[-1].confidence is None and not results[-1].segments
    assert requests[0]["start"] == {
        "outputContract": "STREAMING_OUTPUT_CONTRACT_REVISED_HYPOTHESES",
        "language": "es",
        "mode": "TRANSCRIPTION_MODE_VERBATIM",
        "contextTerms": ["RoomKit"],
    }


async def test_pcm_is_downmixed_resampled_chunked_and_stops_at_final() -> None:
    received = bytearray()

    async def handler(ws: ServerConnection) -> None:
        await receive(ws)
        await send(ws, started={})
        while True:
            event = await receive(ws)
            if "end" in event:
                break
            data = base64.b64decode(event["audio"])
            assert 0 < len(data) <= 65536 and len(data) % 2 == 0
            received.extend(data)
        await send(ws, done={})
        await ws.wait_closed()

    async with server(handler) as url:
        stt = provider(url)
        source = chunks(
            AudioChunk(data=b"\x01\0\x03\0" * 800, sample_rate=8000, channels=2, is_final=True),
            AudioChunk(data=b"\x09\0" * 16000),
        )
        results = [result async for result in stt.transcribe_stream(source)]
        await stt.close()
    assert received == b"\x02\0" * 1600
    assert results[-1].is_final and results[-1].text == "" and results[-1].language is None


@pytest.mark.parametrize("terminal", ["error", "close", "timeout"])
async def test_partial_transcript_is_not_promoted_after_failure(terminal: str) -> None:
    async def handler(ws: ServerConnection) -> None:
        await receive(ws)
        await send(ws, started={})
        assert await receive(ws) == {"end": {}}
        await send(ws, hypothesis={"text": "unfinished"})
        if terminal == "error":
            await send(
                ws, error={"kind": "unavailable", "message": "offline", "requestId": "stt-2"}
            )
        elif terminal == "close":
            await ws.close()
        await ws.wait_closed()

    async with server(handler) as url:
        stt = provider(url, final_timeout_s=0.01)
        results = []
        with pytest.raises((RimeSTTError, ConnectionClosed, TimeoutError)):
            async for result in stt.transcribe_stream(chunks()):
                results.append(result)
        await stt.close()
    assert results and all(not result.is_final for result in results)


async def test_input_failure_is_propagated_without_waiting_for_timeout() -> None:
    async def handler(ws: ServerConnection) -> None:
        await receive(ws)
        await send(ws, started={})
        await ws.wait_closed()

    async def broken() -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"\0\0")
        raise ValueError("microphone failed")

    async with server(handler) as url:
        stt = provider(url)
        with pytest.raises(ValueError, match="microphone failed"):
            _ = [r async for r in stt.transcribe_stream(broken())]
        await stt.close()


@pytest.mark.parametrize("language", ["fr", "auto", ""])
async def test_unsupported_language_is_refused_before_connecting(language: str) -> None:
    stt = provider("ws://127.0.0.1:1")
    with pytest.raises(ValueError, match="language"):
        await stt.transcribe(AudioChunk(data=b"\0\0"), language=language)


async def test_complete_transcription_reuses_stream_without_leaking_language_override() -> None:
    languages = []

    async def handler(ws: ServerConnection) -> None:
        languages.append((await receive(ws))["start"]["language"])
        await send(ws, started={})
        while "end" not in await receive(ws):
            pass
        await send(ws, done={"text": "hello"})
        await ws.wait_closed()

    async with server(handler) as url:
        stt = provider(url)
        assert (await stt.transcribe(AudioChunk(data=b"\0\0"), language="es")).text == "hello"
        await stt.transcribe(AudioChunk(data=b"\0\0"))
        await stt.close()
    assert languages == ["es", "en"]
