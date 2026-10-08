"""Tests for AzureMAISTTProvider (Microsoft MAI-Transcribe-2-Streaming).

The streaming tests run the provider against a real WebSocket server on
localhost, scripted with the events Microsoft documents for the realtime
transcription endpoint (``/mai/v1/realtime``, Learn page of 2026-09-30), so
the handshake, the close codes and the frame types go through the actual
protocol stack. No test here has met the live service.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from roomkit import HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.enums import EventType
from roomkit.models.event import AudioContent
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.azure_mai import (
    AzureMAISTTConfig,
    AzureMAISTTError,
    AzureMAISTTProvider,
)
from roomkit.voice.stt.azure_mai_protocol import (
    Transcript,
    language_code,
    realtime_url,
    to_result,
)

_EVENT = "conversation.item.input_audio_transcription."

# The example Microsoft gives for one utterance, then its completion.
_UTTERANCE = [
    {"type": _EVENT + "delta", "item_id": "item_1", "content_index": 0, "delta": "Hello"},
    {"type": _EVENT + "intermediate", "item_id": "item_1", "intermediate": " world"},
    {"type": _EVENT + "intermediate", "item_id": "item_1", "intermediate": " there"},
    {"type": _EVENT + "delta", "item_id": "item_1", "content_index": 0, "delta": " there!"},
    {"type": "input_audio_buffer.committed"},
    {"type": _EVENT + "completed", "item_id": "item_1", "transcript": "Hello there!"},
]

_PCM_16K = b"\x01\x00" * 1600  # 100 ms of 16 kHz mono

Handler = Callable[[ServerConnection], Awaitable[None]]


class _Record:
    """What the scripted server received."""

    def __init__(self) -> None:
        self.request_path = ""
        self.headers: dict[str, str] = {}
        self.session_update: dict[str, Any] = {}
        self.audio = bytearray()
        self.appends = 0
        self.after_audio: list[dict[str, Any]] = []
        self.connections = 0
        self.closed = asyncio.Event()


async def _accept(ws: ServerConnection, record: _Record) -> None:
    """Open the session the way the service does, recording the client's settings."""
    record.connections += 1
    record.request_path = ws.request.path if ws.request else ""
    record.headers = dict(ws.request.headers) if ws.request else {}
    await ws.send(json.dumps({"type": "session.created", "session": {"id": "sess_1"}}))
    record.session_update = json.loads(await ws.recv())
    await ws.send(json.dumps({"type": "session.updated", "session": {"id": "sess_1"}}))


async def _read_until_commit(ws: ServerConnection, record: _Record) -> None:
    async for message in ws:
        event = json.loads(message)
        if event["type"] == "input_audio_buffer.append":
            record.audio += base64.b64decode(event["audio"])
            record.appends += 1
            continue
        record.after_audio.append(event)
        if event["type"] == "input_audio_buffer.commit":
            return


def _scripted(record: _Record, events: list[dict[str, Any]]) -> Handler:
    """Open the session, read the audio up to the commit, answer ``events``."""

    async def handler(ws: ServerConnection) -> None:
        try:
            await _accept(ws, record)
            await _read_until_commit(ws, record)
            for event in events:
                await ws.send(json.dumps(event))
            await ws.wait_closed()
        finally:
            record.closed.set()

    return handler


@contextlib.asynccontextmanager
async def _server(handler: Handler, **kwargs: Any) -> AsyncIterator[str]:
    async with serve(handler, "127.0.0.1", 0, **kwargs) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"http://127.0.0.1:{port}"


def _provider(endpoint: str, **config: Any) -> AzureMAISTTProvider:
    return AzureMAISTTProvider(AzureMAISTTConfig(endpoint=endpoint, api_key="test-key", **config))


async def _chunks(*chunks: AudioChunk) -> AsyncIterator[AudioChunk]:
    for chunk in chunks:
        yield chunk


async def _drain(stream: AsyncIterator[TranscriptionResult]) -> list[TranscriptionResult]:
    """Every result of a stream, failing rather than hanging if it never ends."""

    async def read() -> list[TranscriptionResult]:
        return [result async for result in stream]

    return await asyncio.wait_for(read(), timeout=5)


async def _collect(
    provider: AzureMAISTTProvider, *chunks: AudioChunk, language: str | None = None
) -> list[TranscriptionResult]:
    return await _drain(provider.transcribe_stream(_chunks(*chunks), language=language))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://res.services.ai.azure.com",
            "https://res.services.ai.azure.com/",
            "https://res.services.ai.azure.com/mai/v1/realtime?intent=transcription",
            "wss://res.services.ai.azure.com/mai/v1/realtime",
        ],
    )
    def test_the_resource_url_gives_the_transcription_socket(self, endpoint: str) -> None:
        assert realtime_url(endpoint) == (
            "wss://res.services.ai.azure.com/mai/v1/realtime?intent=transcription"
        )

    def test_another_path_is_refused(self) -> None:
        with pytest.raises(ValueError, match="resource root"):
            realtime_url("https://res.services.ai.azure.com/openai/v1")

    def test_a_plain_connection_is_refused_off_loopback(self) -> None:
        with pytest.raises(ValueError, match="https"):
            realtime_url("http://res.services.ai.azure.com")
        assert realtime_url("http://127.0.0.1:8080").startswith("ws://127.0.0.1:8080/")

    def test_a_url_that_is_no_resource_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Foundry resource URL"):
            AzureMAISTTConfig(endpoint="res.services.ai.azure.com", api_key="k")

    def test_the_key_is_required_and_stays_out_of_repr(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            AzureMAISTTConfig(endpoint="https://res.services.ai.azure.com", api_key="")
        config = AzureMAISTTConfig(endpoint="https://res.services.ai.azure.com", api_key="s3cret")
        assert "s3cret" not in repr(config)

    def test_a_key_no_header_can_carry_is_refused_without_echoing_it(self) -> None:
        with pytest.raises(ValueError, match="non-ASCII") as info:
            AzureMAISTTConfig(endpoint="https://res.services.ai.azure.com", api_key="key…")
        assert "key…" not in str(info.value)

    def test_a_blank_deployment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="deployment"):
            AzureMAISTTConfig(
                endpoint="https://res.services.ai.azure.com", api_key="k", deployment=" "
            )

    @pytest.mark.parametrize(
        ("language", "code"),
        [("fr-CA", "fr"), ("fr", "fr"), ("zh_CN", "zh"), ("yue", "yue"), ("EN-us", "en")],
    )
    def test_a_language_is_sent_as_its_code(self, language: str, code: str) -> None:
        assert language_code(language) == code

    @pytest.mark.parametrize("language", ["klingon", "no", "x"])
    def test_an_unsupported_language_is_refused(self, language: str) -> None:
        with pytest.raises(ValueError, match="does not support"):
            AzureMAISTTConfig(
                endpoint="https://res.services.ai.azure.com", api_key="k", language=language
            )

    def test_capabilities(self) -> None:
        provider = _provider("https://res.services.ai.azure.com")
        assert provider.supports_streaming is True
        assert provider.supports_language_override is True
        assert provider.supports_diarization is False
        assert provider.name == "AzureMAISTT"

    def test_lazy_getters(self) -> None:
        from roomkit.voice import get_azure_mai_stt_config, get_azure_mai_stt_provider

        assert get_azure_mai_stt_provider() is AzureMAISTTProvider
        assert get_azure_mai_stt_config() is AzureMAISTTConfig


# ---------------------------------------------------------------------------
# Event mapping
# ---------------------------------------------------------------------------


class TestEventMapping:
    def test_deltas_and_intermediates_build_the_partial(self) -> None:
        transcript = Transcript()

        results = [to_result(event, transcript) for event in _UTTERANCE[:4]]

        assert [r.text if r else None for r in results] == [
            "Hello",
            "Hello world",
            "Hello there",
            "Hello there!",
        ]
        assert not any(r.is_final for r in results if r)

    def test_an_unchanged_guess_is_not_repeated(self) -> None:
        transcript = Transcript()
        to_result({"type": _EVENT + "intermediate", "intermediate": "Hello"}, transcript)

        # The service firms up exactly what it had guessed.
        assert to_result({"type": _EVENT + "delta", "delta": "Hello"}, transcript) is None

    def test_completed_is_the_final_even_when_empty(self) -> None:
        final = to_result({"type": _EVENT + "completed", "transcript": " Hi. "}, Transcript())
        empty = to_result({"type": _EVENT + "completed", "transcript": ""}, Transcript())

        assert final == TranscriptionResult(text="Hi.", is_final=True)
        assert empty == TranscriptionResult(text="", is_final=True)

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "input_audio_buffer.committed"},
            {"type": "session.updated"},
            {"type": _EVENT + "intermediate", "intermediate": "  "},
        ],
    )
    def test_events_without_text_for_roomkit_are_dropped(self, event: dict[str, Any]) -> None:
        assert to_result(event, Transcript()) is None

    def test_an_error_event_raises_with_the_service_codes(self) -> None:
        event = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_value",
                "message": "Unknown deployment",
            },
        }

        with pytest.raises(AzureMAISTTError, match="Unknown deployment") as info:
            to_result(event, Transcript())

        assert info.value.code == "invalid_value"
        assert info.value.error_type == "invalid_request_error"
        assert info.value.retryable is False

    def test_a_failed_transcription_raises_and_a_server_error_is_retryable(self) -> None:
        event = {"type": _EVENT + "failed", "error": {"type": "server_error", "message": "boom"}}

        with pytest.raises(AzureMAISTTError) as info:
            to_result(event, Transcript())

        assert info.value.retryable is True


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_one_utterance(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _UTTERANCE)) as url:
            results = await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert record.request_path == "/mai/v1/realtime?intent=transcription"
        assert record.headers["api-key"] == "test-key"
        assert record.session_update == {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "transcription": {"model": "MAI-Transcribe-2-Streaming", "language": None},
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        }
        assert bytes(record.audio) == _PCM_16K
        assert record.after_audio == [{"type": "input_audio_buffer.commit"}]
        assert [(r.text, r.is_final) for r in results] == [
            ("Hello", False),
            ("Hello world", False),
            ("Hello there", False),
            ("Hello there!", False),
            ("Hello there!", True),
        ]

    async def test_the_configured_language_and_a_per_stream_override(self) -> None:
        sent: list[Any] = []
        for override in (None, "es-MX"):
            record = _Record()
            async with _server(_scripted(record, _UTTERANCE[-1:])) as url:
                await _collect(
                    _provider(url, language="fr-CA", deployment="mai-stt"),
                    AudioChunk(data=_PCM_16K, sample_rate=16000),
                    language=override,
                )
            sent.append(record.session_update["session"]["audio"]["input"]["transcription"])

        assert sent == [
            {"model": "mai-stt", "language": "fr"},
            {"model": "mai-stt", "language": "es"},
        ]

    async def test_an_unsupported_override_opens_no_connection(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _UTTERANCE)) as url:
            with pytest.raises(ValueError, match="does not support"):
                await _collect(
                    _provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000), language="tlh"
                )

        assert record.connections == 0

    async def test_24khz_goes_through_untouched(self) -> None:
        record = _Record()
        pcm = b"\x01\x00" * 2400
        async with _server(_scripted(record, _UTTERANCE[-1:])) as url:
            await _collect(_provider(url), AudioChunk(data=pcm, sample_rate=24000))

        assert record.session_update["session"]["audio"]["input"]["format"]["rate"] == 24000
        assert bytes(record.audio) == pcm

    async def test_other_rates_are_resampled_to_16khz(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _UTTERANCE[-1:])) as url:
            await _collect(_provider(url), AudioChunk(data=b"\x01\x00" * 4800, sample_rate=48000))

        assert record.session_update["session"]["audio"]["input"]["format"]["rate"] == 16000
        assert len(record.audio) == 1600 * 2

    async def test_a_long_chunk_leaves_in_appends_of_100_ms(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _UTTERANCE[-1:])) as url:
            await _collect(_provider(url), AudioChunk(data=_PCM_16K * 10, sample_rate=16000))

        assert record.appends == 10
        assert bytes(record.audio) == _PCM_16K * 10

    async def test_a_stream_the_service_finalises_to_nothing_yields_no_final(self) -> None:
        record = _Record()
        events = [
            {"type": _EVENT + "intermediate", "intermediate": "Hm"},
            {"type": _EVENT + "completed", "transcript": ""},
        ]
        async with _server(_scripted(record, events)) as url:
            results = await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert [(r.text, r.is_final) for r in results] == [("Hm", False)]

    async def test_audio_after_a_final_chunk_is_not_sent(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _UTTERANCE[-1:])) as url:
            await _collect(
                _provider(url),
                AudioChunk(data=_PCM_16K, sample_rate=16000),
                AudioChunk(data=_PCM_16K, sample_rate=16000, is_final=True),
                AudioChunk(data=b"\x09\x00" * 1600, sample_rate=16000),
            )

        assert bytes(record.audio) == _PCM_16K * 2

    async def test_no_audio_opens_no_connection(self) -> None:
        record = _Record()
        async with _server(_scripted(record, [])) as url:
            results = await _collect(
                _provider(url), AudioChunk(data=b"", sample_rate=16000, is_final=True)
            )

        assert results == []
        assert record.connections == 0

    async def test_a_refused_session_raises_the_service_error(self) -> None:
        async def handler(ws: ServerConnection) -> None:
            await ws.send(json.dumps({"type": "session.created"}))
            await ws.recv()
            await ws.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": {"code": "DeploymentNotFound", "message": "No such deployment"},
                    }
                )
            )
            await ws.wait_closed()

        async with _server(handler) as url:
            with pytest.raises(AzureMAISTTError, match="No such deployment") as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.code == "DeploymentNotFound"

    async def test_a_session_that_never_opens_times_out_retryable(self) -> None:
        async def handler(ws: ServerConnection) -> None:
            await ws.wait_closed()

        async with _server(handler) as url:
            with pytest.raises(AzureMAISTTError) as info:
                await _collect(
                    _provider(url, handshake_timeout_s=0.2),
                    AudioChunk(data=_PCM_16K, sample_rate=16000),
                )

        assert info.value.retryable is True

    async def test_a_failure_mid_stream_raises(self) -> None:
        record = _Record()
        failed = {"type": _EVENT + "failed", "error": {"type": "server_error", "message": "boom"}}
        async with _server(_scripted(record, [_UTTERANCE[0], failed])) as url:
            with pytest.raises(AzureMAISTTError, match="boom") as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.retryable is True

    async def test_a_session_closed_before_the_final_raises_retryable(self) -> None:
        async def handler(ws: ServerConnection) -> None:
            await _accept(ws, _Record())
            await _read_until_commit(ws, _Record())
            await ws.close(1000)

        async with _server(handler) as url:
            with pytest.raises(AzureMAISTTError, match="before the final") as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.retryable is True

    async def test_a_final_that_never_comes_ends_the_stream(self) -> None:
        record = _Record()
        async with _server(_scripted(record, [])) as url:
            with pytest.raises(AzureMAISTTError, match="no final transcript within") as info:
                await _collect(
                    _provider(url, final_timeout_s=0.2),
                    AudioChunk(data=_PCM_16K, sample_rate=16000),
                )

        assert info.value.retryable is True
        await asyncio.wait_for(record.closed.wait(), timeout=2)

    @pytest.mark.parametrize(
        ("status", "retryable"),
        [(HTTPStatus.SERVICE_UNAVAILABLE, True), (HTTPStatus.UNAUTHORIZED, False)],
    )
    async def test_upgrade_refused_by_http_status(
        self, status: HTTPStatus, retryable: bool
    ) -> None:
        async def handler(ws: ServerConnection) -> None:  # pragma: no cover - never reached
            await ws.close()

        def refuse(connection: ServerConnection, request: Any) -> Any:
            return connection.respond(status, "refused\n")

        async with _server(handler, process_request=refuse) as url:
            with pytest.raises(AzureMAISTTError) as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.status_code == status
        assert info.value.retryable is retryable

    async def test_closing_after_the_final_closes_the_socket(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _UTTERANCE)) as url:
            await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))
            await asyncio.wait_for(record.closed.wait(), timeout=2)

    async def test_a_failing_audio_source_still_commits(self) -> None:
        record = _Record()

        async def broken() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=_PCM_16K, sample_rate=16000)
            raise RuntimeError("microphone unplugged")

        async with _server(_scripted(record, _UTTERANCE[-1:])) as url:
            results = await _drain(_provider(url).transcribe_stream(broken()))

        assert record.after_audio == [{"type": "input_audio_buffer.commit"}]
        assert results == [TranscriptionResult(text="Hello there!")]


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    async def test_a_clip_is_one_committed_stream(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _UTTERANCE)) as url:
            result = await _provider(url).transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert result == TranscriptionResult(text="Hello there!")
        assert record.after_audio == [{"type": "input_audio_buffer.commit"}]

    async def test_a_frame_is_converted_to_16khz_mono(self) -> None:
        record = _Record()
        frame = AudioFrame(data=b"\x01\x00\x01\x00" * 4800, sample_rate=48000, channels=2)
        async with _server(_scripted(record, _UTTERANCE[-1:])) as url:
            await _provider(url).transcribe(frame)

        assert record.session_update["session"]["audio"]["input"]["format"]["rate"] == 16000
        assert len(record.audio) == 1600 * 2

    async def test_a_clip_without_words_is_empty(self) -> None:
        record = _Record()
        events = [{"type": _EVENT + "completed", "transcript": ""}]
        async with _server(_scripted(record, events)) as url:
            result = await _provider(url).transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert result == TranscriptionResult(text="")

    async def test_audio_content_is_refused(self) -> None:
        provider = _provider("https://res.services.ai.azure.com")

        with pytest.raises(ValueError, match="does not take AudioContent"):
            await provider.transcribe(AudioContent(url="https://example.com/a.wav"))


# ---------------------------------------------------------------------------
# Behind a VoiceChannel
# ---------------------------------------------------------------------------


class TestVoiceChannelWithVAD:
    async def test_each_utterance_is_committed_and_reaches_the_room(self) -> None:
        record = _Record()
        transcripts: list[str] = []
        utterance: list[VADEvent | None] = [
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x00\x00" * 1600),
        ]
        async with _server(_scripted(record, _UTTERANCE)) as url:
            stt = _provider(url)
            backend = MockVoiceBackend()
            channel = VoiceChannel(
                "voice-1",
                stt=stt,
                backend=backend,
                pipeline=AudioPipelineConfig(vad=MockVADProvider(events=utterance)),
            )
            kit = RoomKit(stt=stt, voice=backend)
            kit.register_channel(channel)

            @kit.hook(HookTrigger.ON_TRANSCRIPTION)
            async def on_transcription(event: Any, ctx: Any) -> HookResult:
                transcripts.append(event.text)
                return HookResult.allow()

            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice-1")
            session = await kit.join(room.id, "voice-1", participant_id="user-1")
            assert not channel._continuous_stt

            for _ in range(3):
                await backend.simulate_audio_received(session, AudioFrame(data=_PCM_16K))
            await _eventually(lambda: bool(transcripts))
            events = await kit.store.list_events(room.id, offset=0, limit=20)
            await channel.close()

        assert transcripts == ["Hello there!"]
        assert record.connections == 1
        assert record.after_audio == [{"type": "input_audio_buffer.commit"}]
        assert [
            e.content.body
            for e in events
            if e.type == EventType.MESSAGE and e.source.channel_id == "voice-1"
        ] == ["Hello there!"]


async def _eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)
