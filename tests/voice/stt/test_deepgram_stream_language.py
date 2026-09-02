"""A Deepgram stream opens with the language it was given, and reports what it hears.

``transcribe_stream`` imports the SDK's event types inside the function, so
the stream path needs the optional extra installed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("deepgram")

from deepgram.core.events import EventType  # noqa: E402

from roomkit.voice.base import AudioChunk  # noqa: E402
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider  # noqa: E402


class _Connection:
    """A connection that delivers one result, then closes."""

    def __init__(self, message: Any) -> None:
        self._message = message
        self._handlers: dict[Any, Any] = {}

    def on(self, event: Any, handler: Any) -> None:
        self._handlers[event] = handler

    async def start_listening(self) -> None:
        self._handlers[EventType.MESSAGE](self._message)
        self._handlers[EventType.CLOSE](None)

    async def send_media(self, _data: bytes) -> None:
        return None

    async def send_close_stream(self) -> None:
        return None

    async def __aenter__(self) -> _Connection:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _FakeDeepgramModule:
    def __init__(self, message: Any) -> None:
        self._message = message
        self.connect_opts: list[dict[str, Any]] = []

    def AsyncDeepgramClient(self, **_kwargs: Any) -> Any:  # noqa: N802 — SDK name
        module = self

        class _V1:
            @staticmethod
            def connect(**opts: Any) -> _Connection:
                module.connect_opts.append(opts)
                return _Connection(module._message)

        class _Listen:
            v1 = _V1()

        class _Client:
            listen = _Listen()

        return _Client()


def _results_message(transcript: str, *word_languages: str | None) -> SimpleNamespace:
    words = [
        SimpleNamespace(word=w, language=lang)
        for w, lang in zip(transcript.split(), word_languages, strict=False)
    ]
    alt = SimpleNamespace(transcript=transcript, confidence=0.9, words=words)
    return SimpleNamespace(channel=SimpleNamespace(alternatives=[alt]), is_final=True)


def _provider(
    message: Any, language: str = "multi"
) -> tuple[DeepgramSTTProvider, _FakeDeepgramModule]:
    module = _FakeDeepgramModule(message)
    prov = DeepgramSTTProvider.__new__(DeepgramSTTProvider)
    prov._config = DeepgramConfig(api_key="test-key", model="nova-3", language=language)
    prov._dg = module
    prov._client = None
    return prov, module


async def _audio() -> Any:
    yield AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)


async def _collect(prov: DeepgramSTTProvider, **kwargs: Any) -> list[Any]:
    results = []
    async for result in prov.transcribe_stream(_audio(), **kwargs):
        results.append(result)
    return results


async def test_stream_opens_with_the_override_and_reports_the_language() -> None:
    prov, module = _provider(_results_message("bonjour le monde", "fr", "fr", "fr"))

    results = await asyncio.wait_for(_collect(prov, language="fr-CA"), timeout=5)

    assert module.connect_opts[-1]["language"] == "fr-CA"
    assert [r.text for r in results] == ["bonjour le monde"]
    assert results[0].language == "fr"


async def test_stream_without_override_uses_the_config() -> None:
    prov, module = _provider(_results_message("bonjour", None))

    results = await asyncio.wait_for(_collect(prov), timeout=5)

    assert module.connect_opts[-1]["language"] == "multi"
    assert results[0].language is None
