"""Tests for the Google Gemini batch STT provider.

The fake client mirrors what the live API actually does (verified 2026-08-07):
``interactions.create`` answers with the JSON transcript in ``output_text``, and
``files.upload`` guesses ``audio/x-wav`` for a ``.wav`` — a value the
interactions endpoint rejects, which is why the provider sends its own.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from roomkit.models.event import AudioContent
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.stt.gemini import (
    SUPPORTED_MIME_TYPES,
    GeminiSTTConfig,
    GeminiSTTProvider,
    Transcript,
    TranscriptSegment,
)

TRANSCRIPT = {
    "language": "fr-CA",
    "segments": [
        {"speaker": "Speaker 1", "start": "00:00", "end": "00:04", "text": "On commence."},
        {"speaker": "Speaker 2", "start": "00:04", "end": "00:09", "text": "Je suis prêt."},
    ],
}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeInteraction:
    output_text: str | None
    status: str = "completed"


class _FakeInteractions:
    def __init__(self, payload: str | None) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeInteraction:
        self.calls.append(kwargs)
        return _FakeInteraction(output_text=self._payload)


@dataclass
class _FakeFile:
    uri: str = "https://generativelanguage.googleapis.com/v1beta/files/abc123"
    name: str = "files/abc123"
    mime_type: str = "audio/x-wav"  # what the live upload guesses for .wav


class _FakeFiles:
    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.deleted: list[str] = []
        self.configs: list[Any] = []

    async def upload(self, *, file: str, config: Any = None) -> _FakeFile:
        self.uploaded.append(file)
        self.configs.append(config)
        return _FakeFile()

    async def delete(self, *, name: str, config: Any = None) -> None:
        self.deleted.append(name)
        self.configs.append(config)


class _FakeAio:
    def __init__(self, interactions: _FakeInteractions, files: _FakeFiles) -> None:
        self.interactions = interactions
        self.files = files
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, payload: str | None) -> None:
        self.interactions = _FakeInteractions(payload)
        self.files = _FakeFiles()
        self.aio = _FakeAio(self.interactions, self.files)


def _provider(
    payload: str | None = None, **overrides: Any
) -> tuple[GeminiSTTProvider, _FakeClient]:
    provider = GeminiSTTProvider(GeminiSTTConfig(api_key="test-key", **overrides))
    client = _FakeClient(payload if payload is not None else json.dumps(TRANSCRIPT))
    provider._client = client
    return provider, client


def _wav(tmp_path: Path, size: int = 512) -> Path:
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"RIFF" + b"\x00" * size)
    return path


# ---------------------------------------------------------------------------
# Metadata and configuration
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name(self) -> None:
        provider, _ = _provider()
        assert provider.name == "GeminiSTT"

    def test_streaming_is_not_supported(self) -> None:
        """The API takes a complete recording; a voice channel must transcribe
        on SPEECH_END rather than open a stream."""
        provider, _ = _provider()
        assert provider.supports_streaming is False

    def test_api_key_is_not_in_the_repr(self) -> None:
        assert "super-secret" not in repr(GeminiSTTConfig(api_key="super-secret"))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"api_key": "  "}, "api_key"),
            ({"model": " "}, "model"),
            ({"language": " "}, "language"),
            ({"timeout": 0}, "timeout"),
            ({"timeout": float("nan")}, "timeout"),
            ({"connect_timeout": float("nan")}, "connect_timeout"),
            ({"max_inline_bytes": 0}, "max_inline_bytes"),
        ],
    )
    def test_invalid_config_is_rejected(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            GeminiSTTConfig(**{"api_key": "test-key", **overrides})


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    async def test_audio_and_prompt_are_sent_with_a_json_schema(self, tmp_path: Path) -> None:
        provider, client = _provider()

        await provider.transcribe_recording(_wav(tmp_path))

        call = client.interactions.calls[0]
        assert call["model"] == "gemini-3.6-flash"
        # The split lives on the SDK's httpx client: a per-request timeout
        # would be flattened by google-genai to one float (RMK-149).
        assert "timeout" not in call
        audio, prompt = call["input"]
        assert audio["type"] == "audio"
        assert audio["mime_type"] == "audio/wav"
        assert base64.b64decode(audio["data"]).startswith(b"RIFF")
        assert prompt["type"] == "text"
        assert call["response_format"]["mime_type"] == "application/json"
        assert call["response_format"]["schema"]["required"] == ["language", "segments"]

    async def test_diarization_is_asked_for_by_default(self, tmp_path: Path) -> None:
        provider, client = _provider()

        await provider.transcribe_recording(_wav(tmp_path))

        prompt = client.interactions.calls[0]["input"][1]["text"]
        assert "Speaker 1" in prompt
        assert "order of first appearance" in prompt

    async def test_single_speaker_mode_does_not_ask_for_labels(self, tmp_path: Path) -> None:
        provider, client = _provider(diarize=False)

        await provider.transcribe_recording(_wav(tmp_path))

        prompt = client.interactions.calls[0]["input"][1]["text"]
        assert "one speaker" in prompt
        assert "order of first appearance" not in prompt

    async def test_language_hint_and_extra_prompt_reach_the_model(self, tmp_path: Path) -> None:
        provider, client = _provider(language="fr-CA", prompt="The product is spelled RoomKit.")

        await provider.transcribe_recording(_wav(tmp_path))

        prompt = client.interactions.calls[0]["input"][1]["text"]
        assert "fr-CA" in prompt
        assert "The product is spelled RoomKit." in prompt


# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------


class TestInputPaths:
    async def test_small_file_is_inlined(self, tmp_path: Path) -> None:
        provider, client = _provider()

        await provider.transcribe_recording(_wav(tmp_path))

        assert client.files.uploaded == []
        assert "data" in client.interactions.calls[0]["input"][0]

    async def test_large_file_is_uploaded_then_deleted(self, tmp_path: Path) -> None:
        """A meeting recording does not fit in a request body."""
        provider, client = _provider(max_inline_bytes=64)
        path = _wav(tmp_path, size=4096)

        await provider.transcribe_recording(path)

        assert client.files.uploaded == [str(path)]
        audio = client.interactions.calls[0]["input"][0]
        assert audio["uri"].startswith("https://generativelanguage.googleapis.com/")
        # The upload guesses audio/x-wav, which the endpoint rejects.
        assert audio["mime_type"] == "audio/wav"
        assert client.files.deleted == ["files/abc123"]

    async def test_files_api_calls_carry_the_flat_timeout(self, tmp_path: Path) -> None:
        provider, client = _provider(max_inline_bytes=64, timeout=42.0)

        await provider.transcribe_recording(_wav(tmp_path, size=128))

        # The SDK's classic path sends no timeout at all unless told per call,
        # and its option is one flat value in milliseconds (RMK-149). The
        # delete is cleanup awaited before the transcript: seconds, not minutes.
        assert client.files.configs == [
            {"http_options": {"timeout": 42000}},
            {"http_options": {"timeout": 10000}},
        ]

    async def test_upload_is_deleted_even_when_the_request_fails(self, tmp_path: Path) -> None:
        provider, client = _provider(max_inline_bytes=64)

        async def _boom(**kwargs: Any) -> Any:
            raise RuntimeError("upstream exploded")

        client.interactions.create = _boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="upstream exploded"):
            await provider.transcribe_recording(_wav(tmp_path, size=4096))

        assert client.files.deleted == ["files/abc123"]

    async def test_data_url_is_inlined_with_its_own_mime(self) -> None:
        provider, client = _provider()
        payload = base64.b64encode(b"RIFFfake").decode()

        await provider.transcribe_recording(
            AudioContent(url=f"data:audio/wav;base64,{payload}", mime_type="audio/wav")
        )

        audio = client.interactions.calls[0]["input"][0]
        assert audio["data"] == payload
        assert audio["mime_type"] == "audio/wav"

    async def test_raw_frame_is_sent_as_pcm_with_its_rate(self) -> None:
        provider, client = _provider()

        await provider.transcribe_recording(
            AudioFrame(data=b"\x01\x02" * 100, sample_rate=24000, channels=1)
        )

        audio = client.interactions.calls[0]["input"][0]
        assert audio["mime_type"] == "audio/l16"
        assert audio["sample_rate"] == 24000
        assert audio["channels"] == 1
        assert audio["mime_type"] in SUPPORTED_MIME_TYPES

    async def test_arbitrary_urls_are_refused_rather_than_fetched(self) -> None:
        """Dereferencing a caller-supplied URL would make this an SSRF vector."""
        provider, client = _provider()

        with pytest.raises(ValueError, match="will not fetch"):
            await provider.transcribe_recording(
                AudioContent(url="https://evil.example/x.wav", mime_type="audio/wav")
            )

        assert client.interactions.calls == []

    async def test_files_api_uri_is_passed_through(self) -> None:
        provider, client = _provider()
        uri = "https://generativelanguage.googleapis.com/v1beta/files/xyz"

        await provider.transcribe_recording(AudioContent(url=uri, mime_type="audio/wav"))

        assert client.interactions.calls[0]["input"][0]["uri"] == uri

    async def test_a_path_string_with_a_file_scheme_is_read_from_disk(
        self, tmp_path: Path
    ) -> None:
        """A recorder can report where it wrote in either form."""
        provider, client = _provider()

        await provider.transcribe_recording(f"file://{_wav(tmp_path)}")

        assert client.files.uploaded == []
        assert base64.b64decode(client.interactions.calls[0]["input"][0]["data"]).startswith(
            b"RIFF"
        )

    async def test_unknown_extension_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "recording.xyz"
        path.write_bytes(b"junk")
        provider, _ = _provider()

        with pytest.raises(ValueError, match="mime type"):
            await provider.transcribe_recording(path)

    async def test_missing_file_is_reported_as_such(self, tmp_path: Path) -> None:
        provider, _ = _provider()

        with pytest.raises(FileNotFoundError, match="No such recording"):
            await provider.transcribe_recording(tmp_path / "absent.wav")

    async def test_empty_audio_is_rejected_without_a_round_trip(self) -> None:
        provider, client = _provider()

        with pytest.raises(ValueError, match="empty audio"):
            await provider.transcribe_recording(AudioFrame(data=b"", sample_rate=16000))

        assert client.interactions.calls == []


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class TestResults:
    async def test_segments_carry_speakers_and_timestamps(self, tmp_path: Path) -> None:
        provider, _ = _provider()

        transcript = await provider.transcribe_recording(_wav(tmp_path))

        assert transcript.language == "fr-CA"
        assert transcript.segments == [
            TranscriptSegment("Speaker 1", "00:00", "00:04", "On commence."),
            TranscriptSegment("Speaker 2", "00:04", "00:09", "Je suis prêt."),
        ]

    async def test_blank_segments_are_dropped(self, tmp_path: Path) -> None:
        payload = json.dumps(
            {
                "language": "en",
                "segments": [
                    {"speaker": "Speaker 1", "start": "00:00", "end": "00:01", "text": "  "},
                    {"speaker": "Speaker 1", "start": "00:01", "end": "00:02", "text": "Real."},
                ],
            }
        )
        provider, _ = _provider(payload)

        transcript = await provider.transcribe_recording(_wav(tmp_path))

        assert [s.text for s in transcript.segments] == ["Real."]

    async def test_transcribe_flattens_to_the_abc_shape(self) -> None:
        provider, _ = _provider()
        payload = base64.b64encode(b"RIFFfake").decode()

        result = await provider.transcribe(
            AudioContent(url=f"data:audio/wav;base64,{payload}", mime_type="audio/wav")
        )

        assert result.text == "On commence. Je suis prêt."
        assert result.language == "fr-CA"
        assert result.is_final is True
        assert result.words == []

    async def test_missing_transcript_raises_with_the_status(self, tmp_path: Path) -> None:
        provider, _ = _provider(payload="")

        with pytest.raises(RuntimeError, match="no transcript"):
            await provider.transcribe_recording(_wav(tmp_path))

    async def test_non_json_answer_raises(self, tmp_path: Path) -> None:
        provider, _ = _provider(payload="Sure! Here is the transcript:")

        with pytest.raises(RuntimeError, match="not JSON"):
            await provider.transcribe_recording(_wav(tmp_path))


class TestTranscript:
    def test_text_labels_each_turn_and_plain_text_does_not(self) -> None:
        transcript = Transcript(
            language="en",
            segments=[
                TranscriptSegment("Speaker 1", "00:00", "00:02", "Hello."),
                TranscriptSegment("Speaker 2", "00:02", "00:04", "Hi."),
            ],
        )

        assert transcript.text == "Speaker 1: Hello.\nSpeaker 2: Hi."
        assert transcript.plain_text == "Hello. Hi."


class TestLifecycle:
    async def test_close_shuts_the_pool_and_drops_the_client(self) -> None:
        provider, client = _provider()

        await provider.close()

        assert client.aio.closed is True
        assert provider._client is None
