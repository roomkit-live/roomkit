"""Rime TTS contract, including concurrent input, cancellation and split samples."""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from websockets.asyncio.server import ServerConnection

from roomkit.providers.utils import parse_data_uri
from roomkit.voice.tts import RimeTTSConfig, RimeTTSError, RimeTTSProvider
from tests.voice.rime_wire import receive, send, server


def provider(url: str, **kwargs: Any) -> RimeTTSProvider:
    return RimeTTSProvider(RimeTTSConfig(api_key="test-key", websocket_url=url, **kwargs))


async def test_streams_text_and_audio_concurrently_without_losing_spaces() -> None:
    heard_audio = asyncio.Event()
    messages = []

    async def handler(ws: ServerConnection) -> None:
        await send(ws, ready={"protocol": 1})
        start = await receive(ws)
        messages.append(start)
        context = start["contextId"]
        messages.append(await receive(ws))
        await send(ws, contextId=context, audio=base64.b64encode(b"\x01\x02\x03").decode())
        messages.append(await receive(ws))
        await send(ws, contextId=context, audio=base64.b64encode(b"\x04").decode())
        messages.append(await receive(ws))
        await send(ws, contextId=context, done={})
        await ws.wait_closed()

    async def text() -> AsyncIterator[str]:
        yield "Hello "
        await asyncio.wait_for(heard_audio.wait(), 1)
        yield "world."

    async with server(handler) as url:
        tts = provider(url, model="mistv3", sample_rate=24000, time_scale_factor=0.8)
        chunks = []
        async for chunk in tts.synthesize_stream_input(text(), voice="cove"):
            chunks.append(chunk)
            if chunk.data:
                heard_audio.set()
        await tts.close()
    assert b"".join(chunk.data for chunk in chunks) == b"\x01\x02\x03\x04"
    assert all(chunk.sample_rate == 24000 and chunk.format == "pcm_s16le" for chunk in chunks)
    assert chunks[-1].is_final
    assert messages[0]["start"] == {
        "speaker": "cove",
        "language": "en",
        "audioParameters": {
            "audioFormat": "audio/pcm",
            "samplingRate": 24000,
            "timeScaleFactor": 0.8,
        },
        "splitStrategy": "SPLIT_STRATEGY_ACCUMULATE_SENTENCE",
    }
    assert [message.get("text") for message in messages[1:3]] == ["Hello ", "world."]
    assert "end" in messages[-1]
    assert len({message["contextId"] for message in messages}) == 1


async def test_complete_synthesis_is_a_wav_with_duration_and_transcript() -> None:
    async def handler(ws: ServerConnection) -> None:
        await send(ws, ready={"protocol": 1})
        request = await receive(ws)
        assert request["start"]["text"] == "Bonjour."
        assert request["start"]["language"] == "fr"
        await send(ws, audio=base64.b64encode(b"\x01\x00" * 160).decode())
        await send(ws, done={})
        await ws.wait_closed()

    async with server(handler) as url:
        tts = provider(url, language="fr", voice_id="aurelie")
        result = await tts.synthesize("Bonjour.")
        await tts.close()
    mime, data = parse_data_uri(result.url)
    assert mime == "audio/wav"
    assert result.transcript == "Bonjour."
    assert result.duration_seconds == 0.01
    with wave.open(io.BytesIO(data)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert audio.readframes(160) == b"\x01\x00" * 160


@pytest.mark.parametrize(
    ("terminal", "match"),
    [
        (
            {"error": {"kind": "unavailable", "message": "Try later", "requestId": "req-1"}},
            "Try later",
        ),
        ({"cancelled": {}}, "cancelled"),
        ({"audio": "AQ==", "done": {}}, "incomplete PCM"),
    ],
)
async def test_failure_is_never_reported_as_success(terminal: dict[str, Any], match: str) -> None:
    async def handler(ws: ServerConnection) -> None:
        await send(ws, ready={"protocol": 1})
        await receive(ws)
        await send(ws, **terminal)
        await ws.wait_closed()

    async with server(handler) as url:
        tts = provider(url)
        with pytest.raises(RimeTTSError, match=match) as failure:
            await tts.synthesize("Hello.")
        await tts.close()
    if "error" in terminal:
        assert failure.value.retryable
        assert failure.value.request_id == "req-1"


async def test_source_failure_closes_the_connection_and_propagates() -> None:
    closed = asyncio.Event()

    async def handler(ws: ServerConnection) -> None:
        await send(ws, ready={"protocol": 1})
        await receive(ws)
        await ws.wait_closed()
        closed.set()

    async def broken() -> AsyncIterator[str]:
        yield "Hello "
        raise ValueError("text source failed")

    async with server(handler) as url:
        tts = provider(url)
        with pytest.raises(ValueError, match="text source failed"):
            _ = [chunk async for chunk in tts.synthesize_stream_input(broken())]
        await asyncio.wait_for(closed.wait(), 1)
        await tts.close()


async def test_closing_a_reader_cancels_only_its_synthesis() -> None:
    closed = asyncio.Queue[None]()

    async def handler(ws: ServerConnection) -> None:
        await send(ws, ready={"protocol": 1})
        await receive(ws)
        await send(ws, audio="AQACAA==")
        await ws.wait_closed()
        closed.put_nowait(None)

    async with server(handler) as url:
        tts = provider(url)
        first, second = tts.synthesize_stream("One."), tts.synthesize_stream("Two.")
        assert (await anext(first)).data == (await anext(second)).data
        await first.aclose()
        await asyncio.wait_for(closed.get(), 1)
        assert closed.empty()
        await second.aclose()
        await asyncio.wait_for(closed.get(), 1)
        await tts.close()


async def test_voice_catalog_filters_model_language_gender_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        {
            "speaker": "aurelie",
            "modelId": "coda",
            "lang": "fra",
            "gender": "Female",
            "genre": ["Warm"],
        },
        {"speaker": "other", "modelId": "mistv3", "lang": "fra", "gender": "Female"},
        {"speaker": "astra", "modelId": "coda", "lang": "eng", "gender": "Female"},
    ]
    real = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json=entries)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(respond), **kw)
    )
    voices = await provider("ws://unused").list_voices(
        language="fr", gender="female", query="warm"
    )
    assert [voice.id for voice in voices] == ["aurelie"]
