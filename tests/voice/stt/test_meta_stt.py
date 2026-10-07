"""Tests for MetaSTTProvider (Meta Muse Voice Transcribe).

The streaming tests run the provider against a real WebSocket server on
localhost, scripted with the frames the live service sent on 2026-09-27, so
the close codes and the frame types go through the actual protocol stack.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import email
import email.policy
import io
import json
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from websockets.asyncio.server import ServerConnection, serve

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.enums import EventType
from roomkit.models.event import AudioContent
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, SpeakerSegment, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.meta import MetaSTTConfig, MetaSTTError, MetaSTTProvider
from roomkit.voice.stt.meta_protocol import Turns, speaker_label, to_result

_ACK = {"sessionId": "voyager/duplex/TEST"}
_END_OF_STREAM = {"type": "transcript", "transcript": "", "final": True, "audioProcessedMs": 0}

# One turn as the live service sent it in DIARIZATION mode; ENDPOINTING sends
# the same frames without the speaker event, which it would ignore anyway.
_ENDPOINTING_TURN = [
    {"type": "speechStart", "audioProcessedMs": 60, "turnId": 0},
    {"type": "transcript", "transcript": " Bonjour", "final": False, "audioProcessedMs": 1200},
    {"type": "transcript", "transcript": " Bonjour Julie,", "final": False},
    {"type": "speaker", "label": "A", "audioProcessedMs": 4560},
    {"type": "speechEnd", "audioProcessedMs": 4060, "turnId": 0},
    {"type": "speechComplete", "turnId": 0, "transcript": " Bonjour Julie, ça va? "},
    _END_OF_STREAM,
]

Handler = Callable[[ServerConnection], Awaitable[None]]


class _Record:
    """What the scripted server received."""

    def __init__(self) -> None:
        self.handshake: dict[str, Any] = {}
        self.audio = bytearray()
        self.controls: list[dict[str, Any]] = []
        self.connections = 0
        self.closed = asyncio.Event()


def _scripted(record: _Record, events: list[dict[str, Any]], ack: Any = _ACK) -> Handler:
    """Accept, read the audio up to ``endStream``, answer ``events``, close 1000."""

    async def handler(ws: ServerConnection) -> None:
        record.connections += 1
        try:
            record.handshake = json.loads(await ws.recv())
            await ws.send(json.dumps(ack))
            async for message in ws:
                if isinstance(message, bytes):
                    record.audio += message
                    continue
                record.controls.append(json.loads(message))
                if record.controls[-1] == {"type": "endStream"}:
                    break
            for event in events:
                await ws.send(json.dumps(event))
            await ws.close(1000, "No more transcript segments")
        finally:
            record.closed.set()

    return handler


@contextlib.asynccontextmanager
async def _server(handler: Handler) -> AsyncIterator[str]:
    async with serve(handler, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


def _provider(url: str, **config: Any) -> MetaSTTProvider:
    return MetaSTTProvider(MetaSTTConfig(api_key="test-key", realtime_url=url, **config))


async def _chunks(*chunks: AudioChunk) -> AsyncIterator[AudioChunk]:
    for chunk in chunks:
        yield chunk


async def _drain(stream: AsyncIterator[TranscriptionResult]) -> list[TranscriptionResult]:
    """Every result of a stream, failing rather than hanging if it never ends."""

    async def read() -> list[TranscriptionResult]:
        return [result async for result in stream]

    return await asyncio.wait_for(read(), timeout=5)


async def _collect(provider: MetaSTTProvider, *chunks: AudioChunk) -> list[TranscriptionResult]:
    return await _drain(provider.transcribe_stream(_chunks(*chunks)))


_PCM_16K = b"\x01\x00" * 1600  # 100 ms of 16 kHz mono


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    @pytest.mark.parametrize(
        ("mode", "diarizing"),
        [("DIARIZATION", True), ("ENDPOINTING", False), ("PUSH_TO_TALK", False)],
    )
    def test_only_diarization_mode_labels_speakers(self, mode: Any, diarizing: bool) -> None:
        assert _provider("ws://unused", mode=mode).supports_diarization is diarizing

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ENDPOINTING"):
            MetaSTTConfig(api_key="k", mode="BOGUS")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bias", [["fr"], ["french"], ["French", "Klingon"]])
    def test_language_bias_takes_documented_names_only(self, bias: list[str]) -> None:
        # The service accepts these silently and the bias does nothing.
        with pytest.raises(ValueError, match="language names"):
            MetaSTTConfig(api_key="k", language_bias=bias)

    def test_documented_names_are_accepted_and_the_key_stays_out_of_repr(self) -> None:
        config = MetaSTTConfig(api_key="secret-key", language_bias=["French", "English"])
        assert "secret-key" not in repr(config)

    def test_api_key_is_required(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            MetaSTTConfig(api_key="")

    def test_a_key_no_header_can_carry_is_refused_without_echoing_it(self) -> None:
        # A placeholder "…" pasted as the key failed later, as a UnicodeEncodeError
        # from the HTTP client on the REST path.
        with pytest.raises(ValueError, match="non-ASCII") as info:
            MetaSTTConfig(api_key="sk-secret…")
        assert "sk-secret" not in str(info.value)

    def test_language_override_is_not_claimed(self) -> None:
        # languageBias biases, it does not pin (RFC §12.2).
        assert _provider("ws://unused").supports_language_override is False


# ---------------------------------------------------------------------------
# Event mapping
# ---------------------------------------------------------------------------


class TestEventMapping:
    def test_speech_start_signals_speech(self) -> None:
        result = to_result({"type": "speechStart", "turnId": 0})
        assert result is not None
        assert result.is_speech_start and not result.is_final and result.text == ""

    def test_partial_is_stripped(self) -> None:
        result = to_result({"type": "transcript", "transcript": " Bonjour ", "final": False})
        assert result == TranscriptionResult(text="Bonjour", is_final=False)

    def test_speech_complete_is_a_final(self) -> None:
        result = to_result({"type": "speechComplete", "transcript": " Oui. ", "turnId": 1})
        assert result == TranscriptionResult(text="Oui.", is_final=True)

    def test_push_to_talk_final_transcript_is_a_final(self) -> None:
        result = to_result({"type": "transcript", "transcript": "Oui.", "final": True})
        assert result == TranscriptionResult(text="Oui.", is_final=True)

    @pytest.mark.parametrize(
        "event",
        [
            _END_OF_STREAM,
            {"type": "speechEnd", "turnId": 0},
            {"type": "speaker", "label": "A"},
            {"type": "audioProgress", "audioProcessedMs": 80},
            _ACK,
        ],
    )
    def test_events_without_text_for_roomkit_are_dropped(self, event: dict[str, Any]) -> None:
        assert to_result(event) is None

    @pytest.mark.parametrize(
        ("value", "label"),
        [("A", "A"), (0, "0"), (" B ", "B"), ("unknown", None), ("", None), (None, None)],
    )
    def test_labels_are_strings_and_unattributed_is_none(
        self, value: Any, label: str | None
    ) -> None:
        assert speaker_label(value) == label

    def test_diarized_turn_carries_its_speaker_and_offsets(self) -> None:
        turn = Turns()
        results = [to_result(event, turn) for event in _ENDPOINTING_TURN]
        final = [r for r in results if r is not None and r.is_final]

        assert final == [
            TranscriptionResult(
                text="Bonjour Julie, ça va?",
                segments=[SpeakerSegment("A", "Bonjour Julie, ça va?", 60, 4060)],
            )
        ]
        assert final[0].speaker == "A"

    def test_a_label_is_never_carried_into_the_next_turn(self) -> None:
        turn = Turns()
        for event in _ENDPOINTING_TURN:
            to_result(event, turn)
        # A turn the service could not attribute: no speaker event at all.
        to_result({"type": "speechStart", "audioProcessedMs": 5000, "turnId": 1}, turn)
        result = to_result({"type": "speechComplete", "transcript": "Oui.", "turnId": 1}, turn)

        assert result is not None
        assert result.segments == [SpeakerSegment(None, "Oui.", 5000, None)]
        assert result.speaker is None

    def test_overlapping_turns_keep_their_own_speaker_and_offsets(self) -> None:
        # A change of voice ends a turn without a pause: turn 1 may start
        # before turn 0 completes, and the speaker event names no turn.
        turns = Turns()
        events = [
            {"type": "speechStart", "audioProcessedMs": 0, "turnId": 0},
            {"type": "transcript", "transcript": "Bonjour", "final": False},
            {"type": "speechStart", "audioProcessedMs": 3000, "turnId": 1},
            {"type": "speaker", "label": "A"},
            {"type": "speechEnd", "audioProcessedMs": 2900, "turnId": 0},
            {"type": "speechComplete", "transcript": "Bonjour", "turnId": 0},
            {"type": "speaker", "label": "B"},
            {"type": "speechEnd", "audioProcessedMs": 5000, "turnId": 1},
            {"type": "speechComplete", "transcript": "Oui", "turnId": 1},
        ]
        finals = [r for e in events if (r := to_result(e, turns)) is not None and r.is_final]

        assert [r.segments for r in finals] == [
            [SpeakerSegment("A", "Bonjour", 0, 2900)],
            [SpeakerSegment("B", "Oui", 3000, 5000)],
        ]

    def test_an_empty_completion_still_closes_its_turn(self) -> None:
        turns = Turns()
        for event in [
            {"type": "speechStart", "audioProcessedMs": 0, "turnId": 0},
            {"type": "speaker", "label": "A"},
            {"type": "speechEnd", "audioProcessedMs": 900, "turnId": 0},
        ]:
            to_result(event, turns)
        assert to_result({"type": "speechComplete", "transcript": " ", "turnId": 0}, turns) is None

        to_result({"type": "speechStart", "audioProcessedMs": 1000, "turnId": 1}, turns)
        # A completion naming no turn takes the oldest open one: turn 1, not A's.
        result = to_result({"type": "speechComplete", "transcript": "Oui."}, turns)
        assert result is not None
        assert result.segments == [SpeakerSegment(None, "Oui.", 1000, None)]

    def test_partials_carry_no_speaker(self) -> None:
        result = to_result({"type": "transcript", "transcript": "Bon", "final": False}, Turns())
        assert result is not None and result.segments == []

    def test_error_frame_raises_with_meta_codes(self) -> None:
        with pytest.raises(MetaSTTError) as info:
            to_result(
                {
                    "type": "error",
                    "message": "Billing verification failed",
                    "errorType": "billing_error",
                    "errorCode": "billing_not_configured",
                }
            )
        assert info.value.code == "billing_not_configured"
        assert info.value.error_type == "billing_error"
        assert info.value.retryable is False


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_endpointing_turn(self) -> None:
        record = _Record()
        async with _server(_scripted(record, _ENDPOINTING_TURN)) as url:
            provider = _provider(url, keywords=["RoomKit"], language_bias=["French"])
            results = await _collect(
                provider,
                AudioChunk(data=_PCM_16K, sample_rate=16000),
                AudioChunk(data=_PCM_16K, sample_rate=16000, is_final=True),
            )

        assert record.handshake == {
            "authorization": {"accessToken": "Bearer test-key"},
            "model": "muse-voice-transcribe-1.0",
            "mode": "ENDPOINTING",
            "keywords": ["RoomKit"],
            "languageBias": ["French"],
            "audioEncoding": "PCM_16KHZ",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
        }
        assert bytes(record.audio) == _PCM_16K * 2
        assert record.controls == [{"type": "endStream"}]
        assert results == [
            TranscriptionResult(text="", is_final=False, is_speech_start=True),
            TranscriptionResult(text="Bonjour", is_final=False),
            TranscriptionResult(text="Bonjour Julie,", is_final=False),
            # ENDPOINTING: the speaker frame is ignored, no speaker invented.
            TranscriptionResult(text="Bonjour Julie, ça va?", is_final=True),
        ]

    async def test_diarization_stream_names_each_turns_speaker(self) -> None:
        record = _Record()
        second_turn = [
            {"type": "speechStart", "audioProcessedMs": 4300, "turnId": 1},
            {"type": "transcript", "transcript": " Oui", "final": False},
            {"type": "speaker", "label": "B", "audioProcessedMs": 10080},
            {"type": "speechEnd", "audioProcessedMs": 9580, "turnId": 1},
            {"type": "speechComplete", "turnId": 1, "transcript": " Oui, je l'ai lu. "},
        ]
        events = _ENDPOINTING_TURN[:-1] + second_turn + [_END_OF_STREAM]
        async with _server(_scripted(record, events)) as url:
            results = await _collect(
                _provider(url, mode="DIARIZATION"), AudioChunk(data=_PCM_16K, sample_rate=16000)
            )

        assert record.handshake["mode"] == "DIARIZATION"
        finals = [r for r in results if r.is_final]
        assert [(r.speaker, r.text) for r in finals] == [
            ("A", "Bonjour Julie, ça va?"),
            ("B", "Oui, je l'ai lu."),
        ]
        assert finals[1].segments == [SpeakerSegment("B", "Oui, je l'ai lu.", 4300, 9580)]

    async def test_push_to_talk_final_arrives_after_end_of_stream(self) -> None:
        record = _Record()
        events = [
            {"type": "transcript", "transcript": "Bonjour", "final": False},
            {"type": "transcript", "transcript": "Bonjour, une table.", "final": True},
        ]
        async with _server(_scripted(record, events)) as url:
            results = await _collect(
                _provider(url, mode="PUSH_TO_TALK"), AudioChunk(data=_PCM_16K, sample_rate=16000)
            )

        assert record.handshake["mode"] == "PUSH_TO_TALK"
        assert [r.is_final for r in results] == [False, True]
        assert results[-1].text == "Bonjour, une table."

    async def test_a_backlog_is_paced_not_sent_at_once(self) -> None:
        """A stream that opens on a backlog sends a burst, then catches up at a
        bounded speed (RFC §12.2): the service refuses 7 s of audio at once."""
        record = _Record()
        backlog = [AudioChunk(data=_PCM_16K, sample_rate=16000) for _ in range(12)]
        with (
            patch("roomkit.voice.stt.meta._BURST_S", 0.2),
            patch("roomkit.voice.stt.meta._CATCH_UP_SPEED", 10.0),
        ):
            async with _server(_scripted(record, [_END_OF_STREAM])) as url:
                start = time.monotonic()
                await _collect(_provider(url), *backlog)
                elapsed = time.monotonic() - start

        assert bytes(record.audio) == _PCM_16K * 12
        # 1.2 s of audio: 0.2 s at once, then 1 s at ten times real time.
        assert elapsed >= 0.1

    async def test_24khz_goes_through_untouched(self) -> None:
        record = _Record()
        pcm = b"\x02\x00" * 2400
        async with _server(_scripted(record, [_END_OF_STREAM])) as url:
            await _collect(_provider(url), AudioChunk(data=pcm, sample_rate=24000))

        assert record.handshake["audioEncoding"] == "PCM_24KHZ"
        assert bytes(record.audio) == pcm

    async def test_other_rates_are_resampled_to_24khz(self) -> None:
        record = _Record()
        async with _server(_scripted(record, [_END_OF_STREAM])) as url:
            await _collect(_provider(url), AudioChunk(data=b"\x03\x00" * 800, sample_rate=8000))

        assert record.handshake["audioEncoding"] == "PCM_24KHZ"
        assert len(record.audio) == 2400 * 2  # 100 ms at 24 kHz

    async def test_refused_handshake_raises_the_service_error(self) -> None:
        record = _Record()
        ack = {
            "type": "error",
            "message": "Billing verification failed",
            "sessionId": "voyager/duplex/X",
            "errorType": "billing_error",
            "errorCode": "billing_not_configured",
        }
        async with _server(_scripted(record, [], ack=ack)) as url:
            with pytest.raises(MetaSTTError) as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.code == "billing_not_configured"
        assert info.value.retryable is False

    async def test_handshake_closed_1008_without_error_frame(self) -> None:
        async def handler(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.close(1008, "Bad Request")

        async with _server(handler) as url:
            with pytest.raises(MetaSTTError) as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.code == "1008"
        assert info.value.retryable is False

    async def test_server_fault_mid_stream_is_retryable(self) -> None:
        async def handler(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.send(json.dumps(_ACK))
            await ws.recv()
            await ws.close(1011, "Max session duration reached")

        async with _server(handler) as url:
            with pytest.raises(MetaSTTError) as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.code == "1011"
        assert info.value.retryable is True

    async def test_connection_dropped_without_close_frame_is_retryable(self) -> None:
        # A proxy or a network cut kills the TCP stream: no close code at all.
        async def handler(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.send(json.dumps(_ACK))
            await ws.recv()
            ws.transport.abort()

        async with _server(handler) as url:
            with pytest.raises(MetaSTTError) as info:
                await _collect(_provider(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert info.value.code is None
        assert info.value.retryable is True

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

        async with serve(handler, "127.0.0.1", 0, process_request=refuse) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            with pytest.raises(MetaSTTError) as info:
                await _collect(
                    _provider(f"ws://127.0.0.1:{port}"),
                    AudioChunk(data=_PCM_16K, sample_rate=16000),
                )

        assert info.value.status_code == status
        assert info.value.retryable is retryable

    async def test_no_audio_opens_no_connection(self) -> None:
        record = _Record()
        async with _server(_scripted(record, [])) as url:
            results = await _collect(
                _provider(url), AudioChunk(data=b"", sample_rate=16000, is_final=True)
            )

        assert results == []
        assert record.connections == 0

    async def test_closing_after_a_final_closes_the_socket(self) -> None:
        # Continuous mode stops reading at the first final and opens a new
        # stream for the next turn: the old socket must not linger.
        record = _Record()
        audio_open: asyncio.Queue[AudioChunk | None] = asyncio.Queue()

        async def endless() -> AsyncIterator[AudioChunk]:
            while (chunk := await audio_open.get()) is not None:
                yield chunk

        async def handler(ws: ServerConnection) -> None:
            try:
                await ws.recv()
                await ws.send(json.dumps(_ACK))
                await ws.recv()
                await ws.send(json.dumps(_ENDPOINTING_TURN[-2]))
                await ws.wait_closed()
            finally:
                record.closed.set()

        async with _server(handler) as url:
            audio_open.put_nowait(AudioChunk(data=_PCM_16K, sample_rate=16000))
            stream = _provider(url).transcribe_stream(endless())
            final = await asyncio.wait_for(anext(stream), timeout=5)
            await stream.aclose()
            await asyncio.wait_for(record.closed.wait(), timeout=2)

        assert final.is_final and final.text == "Bonjour Julie, ça va?"

    async def test_failing_audio_source_still_ends_the_stream(self) -> None:
        record = _Record()

        async def broken() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=_PCM_16K, sample_rate=16000)
            raise RuntimeError("microphone unplugged")

        events = [{"type": "transcript", "transcript": "Bonjour", "final": True}]
        async with _server(_scripted(record, events)) as url:
            provider = _provider(url, mode="PUSH_TO_TALK")
            results = await _drain(provider.transcribe_stream(broken()))

        assert record.controls == [{"type": "endStream"}]
        assert results == [TranscriptionResult(text="Bonjour", is_final=True)]


# ---------------------------------------------------------------------------
# Batch (REST)
# ---------------------------------------------------------------------------


def _multipart(request: Any) -> dict[str, tuple[str | None, bytes]]:
    """``{part name: (content type, payload)}`` of a multipart request."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = email.message_from_bytes(head + request.content, policy=email.policy.HTTP)
    return {
        part.get_param("name", header="content-disposition"): (
            part.get_content_type(),
            part.get_payload(decode=True),
        )
        for part in message.iter_parts()
    }


@contextlib.contextmanager
def _rest(respond: Callable[[Any], Any]) -> Any:
    real = httpx.AsyncClient
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return respond(request)

    with patch(
        "httpx.AsyncClient",
        side_effect=lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
    ):
        yield seen


class TestBatch:
    async def test_transcribe_uploads_a_wav_and_returns_the_transcript(self) -> None:
        answer = {"sessionId": "s", "transcript": " Bonjour. ", "turns": []}
        with _rest(lambda _: httpx.Response(200, json=answer)) as seen:
            provider = _provider("ws://unused", language_bias=["French"])
            result = await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))
            await provider.close()

        assert result == TranscriptionResult(text="Bonjour.")
        request = seen[0]
        assert str(request.url) == "https://api.meta.ai/v1/asr/transcribe"
        assert request.headers["authorization"] == "Bearer test-key"
        parts = _multipart(request)
        assert json.loads(parts["request"][1]) == {
            "model": "muse-voice-transcribe-1.0",
            "mode": "PUSH_TO_TALK",
            "languageBias": ["French"],
            "audioEncoding": "WAV",
        }
        with wave.open(io.BytesIO(parts["audio"][1])) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            assert wav.readframes(wav.getnframes()) == _PCM_16K

    async def test_diarized_clip_comes_back_as_speaker_segments(self) -> None:
        # The live answer's shape (2026-09-27), plus a turn Meta left unattributed.
        answer = {
            "sessionId": "s",
            "transcript": "Bonjour Julie? Oui. Parfait.",
            "audioDurationMs": 8240,
            "turns": [
                {
                    "turnId": 0,
                    "startMs": 300,
                    "endMs": 4380,
                    "transcript": "Bonjour Julie?",
                    "speaker": "A",
                },
                {
                    "turnId": 1,
                    "startMs": 4380,
                    "endMs": 9420,
                    "transcript": " Oui. ",
                    "speaker": "B",
                },
                {"turnId": 2, "startMs": 9660, "endMs": 11000, "transcript": "Parfait."},
            ],
        }
        with _rest(lambda _: httpx.Response(200, json=answer)) as seen:
            provider = _provider("ws://unused", mode="DIARIZATION")
            result = await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))
            await provider.close()

        assert json.loads(_multipart(seen[0])["request"][1])["mode"] == "DIARIZATION"
        assert result.text == "Bonjour Julie? Oui. Parfait."
        assert result.segments == [
            SpeakerSegment("A", "Bonjour Julie?", 300, 4380),
            SpeakerSegment("B", "Oui.", 4380, 9420),
            SpeakerSegment(None, "Parfait.", 9660, 11000),
        ]

    async def test_diarized_text_without_turns_is_attributed_to_nobody(self) -> None:
        answer = {"sessionId": "s", "transcript": "Bonjour.", "turns": []}
        with _rest(lambda _: httpx.Response(200, json=answer)):
            provider = _provider("ws://unused", mode="DIARIZATION")
            result = await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))
            await provider.close()

        assert result.segments == [SpeakerSegment(None, "Bonjour.")]

    @pytest.mark.parametrize(("status", "retryable"), [(402, False), (429, True), (503, True)])
    async def test_transcribe_maps_error_bodies(self, status: int, retryable: bool) -> None:
        body = {"error": {"code": "billing_not_configured", "type": "billing_error"}}
        with _rest(lambda _: httpx.Response(status, json=body)):
            provider = _provider("ws://unused")
            with pytest.raises(MetaSTTError) as info:
                await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))
            await provider.close()

        assert info.value.status_code == status
        assert info.value.code == "billing_not_configured"
        assert info.value.retryable is retryable

    async def test_transcribe_decodes_a_wav_data_uri_locally(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(8000)
            writer.writeframes(b"\x03\x00" * 800)  # 100 ms at 8 kHz
        url = "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()

        answer = {"sessionId": "s", "transcript": "Allô", "turns": []}
        with _rest(lambda _: httpx.Response(200, json=answer)) as seen:
            provider = _provider("ws://unused")
            result = await provider.transcribe(AudioContent(url=url, mime_type="audio/wav"))
            await provider.close()

        assert result.text == "Allô"
        with wave.open(io.BytesIO(_multipart(seen[0])["audio"][1])) as wav:
            assert wav.getframerate() == 24000  # resampled from 8 kHz
            assert wav.getnframes() == 2400

    async def test_transcribe_refuses_a_data_uri_that_is_not_wav(self) -> None:
        url = "data:audio/ogg;base64," + base64.b64encode(b"OggS").decode()
        with pytest.raises(ValueError, match="WAV only"):
            await _provider("ws://unused").transcribe(AudioContent(url=url))

    async def test_transcribe_refuses_to_fetch_a_url(self) -> None:
        with pytest.raises(ValueError, match="does not fetch URLs"):
            await _provider("ws://unused").transcribe(
                AudioContent(url="https://example.com/a.ogg")
            )


# ---------------------------------------------------------------------------
# VoiceChannel, continuous mode
# ---------------------------------------------------------------------------


async def _eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)


class TestVoiceChannelContinuous:
    async def test_diarized_turns_reach_the_room_on_one_stream(self) -> None:
        # Two turns, two voices, one WebSocket: the channel keeps the stream,
        # so "A" and "B" compare, and each turn is its own message.
        record = _Record()
        turn_b = [
            {"type": "speechStart", "audioProcessedMs": 4300, "turnId": 1},
            {"type": "speaker", "label": "B"},
            {"type": "speechEnd", "audioProcessedMs": 9580, "turnId": 1},
            {"type": "speechComplete", "turnId": 1, "transcript": " Oui, je l'ai lu. "},
        ]

        async def handler(ws: ServerConnection) -> None:
            record.connections += 1
            record.handshake = json.loads(await ws.recv())
            await ws.send(json.dumps(_ACK))
            await ws.recv()
            for event in _ENDPOINTING_TURN[:-1]:
                await ws.send(json.dumps(event))
            await ws.recv()
            for event in turn_b:
                await ws.send(json.dumps(event))
            await ws.wait_closed()

        async with _server(handler) as url:
            stt = _provider(url, mode="DIARIZATION")
            backend = MockVoiceBackend()
            channel = VoiceChannel(
                "voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig()
            )
            kit = RoomKit(stt=stt, voice=backend)
            kit.register_channel(channel)
            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice-1")
            session = await kit.join(room.id, "voice-1", participant_id="owner")

            await backend.simulate_audio_received(session, AudioFrame(data=_PCM_16K * 2))
            await asyncio.sleep(0.3)
            await backend.simulate_audio_received(session, AudioFrame(data=_PCM_16K * 2))

            async def said() -> list[tuple[Any, str]]:
                events = await kit.store.list_events(room.id, offset=0, limit=20)
                return [
                    (e.metadata.get("sender_name"), e.content.body)
                    for e in events
                    if e.source.channel_id == "voice-1" and e.type == EventType.MESSAGE
                ]

            await _eventually(lambda: record.connections >= 1)
            for _ in range(150):
                messages = await said()
                if len(messages) == 2:
                    break
                await asyncio.sleep(0.02)
            await channel.close()

        assert messages == [
            ("Speaker A", "Bonjour Julie, ça va?"),
            ("Speaker B", "Oui, je l'ai lu."),
        ]
        assert record.connections == 1
        assert record.handshake["mode"] == "DIARIZATION"

    async def test_model_endpointing_drives_the_turn(self) -> None:
        # No pipeline VAD: the channel streams everything and the model's
        # speechStart / speechComplete are the turn's only boundaries.
        record = _Record()

        async def handler(ws: ServerConnection) -> None:
            record.connections += 1
            record.handshake = json.loads(await ws.recv())
            await ws.send(json.dumps(_ACK))
            await ws.recv()
            for event in _ENDPOINTING_TURN[:-1]:
                await ws.send(json.dumps(event))
            await ws.wait_closed()

        transcripts: list[str] = []
        speech_starts: list[Any] = []
        async with _server(handler) as url:
            stt = _provider(url)
            backend = MockVoiceBackend()
            channel = VoiceChannel(
                "voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig()
            )
            kit = RoomKit(stt=stt, voice=backend)
            kit.register_channel(channel)

            @kit.hook(HookTrigger.ON_TRANSCRIPTION)
            async def on_transcription(event: Any, ctx: Any) -> HookResult:
                transcripts.append(event.text)
                return HookResult.allow()

            @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
            async def on_speech_start(event: Any, ctx: Any) -> None:
                speech_starts.append(event)

            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice-1")
            session = await kit.join(room.id, "voice-1", participant_id="user-1")
            assert channel._continuous_stt

            await backend.simulate_audio_received(session, AudioFrame(data=_PCM_16K * 2))
            await _eventually(lambda: bool(transcripts))
            await channel.close()

        assert transcripts == ["Bonjour Julie, ça va?"]
        assert speech_starts
        assert record.handshake["mode"] == "ENDPOINTING"
        assert record.handshake["audioEncoding"] == "PCM_16KHZ"
