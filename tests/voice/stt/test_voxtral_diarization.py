"""VoxtralSTTProvider's speaker labels, in a file and live (RFC §12.2.3).

Frames and answers are shaped as the live service sent them on 2026-10-21.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from roomkit import HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.enums import EventType
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, SpeakerSegment, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.stt.voxtral import VoxtralSTTConfig, VoxtralSTTProvider, VoxtralTurnDetection
from roomkit.voice.stt.voxtral_protocol import rest_result
from roomkit.voice.stt.voxtral_transcript import SpeakerTurns
from tests.voice.stt.test_voxtral_stt import (
    _PCM_16K,
    _SPEECH_START,
    Script,
    _collect,
    _done,
    _eventually,
    _multipart,
    _Record,
    _rest,
    _scripted,
    _server,
)


def _segment(speaker: str | None, text: str) -> dict[str, Any]:
    return {
        "type": "transcription.segment",
        "text": text,
        "start": None,
        "end": None,
        "speaker_id": speaker,
    }


def _diarizing(url: str = "ws://unused", **config: Any) -> VoxtralSTTProvider:
    config.setdefault("diarized_pause_ms", 150)
    return VoxtralSTTProvider(
        VoxtralSTTConfig(
            api_key="test-key", realtime_url=url, diarize=True, turn_detection=None, **config
        )
    )


def _said(results: list[TranscriptionResult]) -> list[tuple[str | None, str]]:
    return [(r.speaker, r.text) for r in results if r.is_final]


class TestConfig:
    def test_labels_are_claimed_with_diarize_only(self) -> None:
        assert _diarizing().supports_diarization is True
        assert VoxtralSTTProvider(VoxtralSTTConfig(api_key="k")).supports_diarization is False

    def test_diarize_with_turn_detection_is_refused(self) -> None:
        # The service refuses both in one session ("Diarization is not
        # compatible with turn detection.", code 3051).
        with pytest.raises(ValueError, match="turn_detection=None"):
            VoxtralSTTConfig(api_key="k", diarize=True)
        with pytest.raises(ValueError, match="turn_detection=None"):
            VoxtralSTTConfig(
                api_key="k", diarize=True, turn_detection=VoxtralTurnDetection.aggressive()
            )

    def test_references_need_diarize(self) -> None:
        with pytest.raises(ValueError, match="require diarize"):
            VoxtralSTTConfig(api_key="k", speaker_references={"Julie": b"RIFF"})

    def test_at_most_eight_references(self) -> None:
        refs = {f"voice{i}": b"RIFF" for i in range(9)}
        with pytest.raises(ValueError, match="at most 8"):
            VoxtralSTTConfig(
                api_key="k", diarize=True, turn_detection=None, speaker_references=refs
            )

    def test_a_blank_reference_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="blank"):
            VoxtralSTTConfig(
                api_key="k", diarize=True, turn_detection=None, speaker_references={" ": b"RIFF"}
            )

    def test_the_pause_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="diarized_pause_ms"):
            VoxtralSTTConfig(api_key="k", diarize=True, turn_detection=None, diarized_pause_ms=0)

    @pytest.mark.parametrize("name", ["Speaker Bob", "unknown", "PENDING", "3"])
    def test_a_name_the_room_would_not_show_as_given_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="as given"):
            VoxtralSTTConfig(
                api_key="k", diarize=True, turn_detection=None, speaker_references={name: b"RIFF"}
            )

    @pytest.mark.parametrize("clip", ["julie.wav", b"", None])
    def test_a_reference_must_be_the_clips_bytes(self, clip: Any) -> None:
        with pytest.raises(ValueError, match="bytes"):
            VoxtralSTTConfig(
                api_key="k", diarize=True, turn_detection=None, speaker_references={"Julie": clip}
            )

    def test_the_clips_stay_out_of_repr(self) -> None:
        config = VoxtralSTTConfig(
            api_key="k", diarize=True, turn_detection=None, speaker_references={"Julie": b"RIFF"}
        )
        assert "RIFF" not in repr(config)


class TestSpeakerTurns:
    def test_a_change_of_speaker_ends_the_turn(self) -> None:
        turns = SpeakerTurns(pause_s=1.5)
        results = [
            r
            for event in (
                _segment("speaker_1", "Hi,"),
                _segment("speaker_1", " Julie."),
                _segment("speaker_2", " Yes,"),
            )
            for r in turns.feed(event)
        ]
        assert results == [
            _SPEECH_START,
            TranscriptionResult(text="Hi,", is_final=False),
            TranscriptionResult(text="Hi, Julie.", is_final=False),
            TranscriptionResult(text="Hi, Julie.", segments=[SpeakerSegment("1", "Hi, Julie.")]),
            _SPEECH_START,
            TranscriptionResult(text="Yes,", is_final=False),
        ]

    def test_a_reference_name_is_the_label(self) -> None:
        turns = SpeakerTurns(pause_s=1.5)
        turns.feed(_segment("Sheila", " Friday works for me."))
        assert turns.close()[0].speaker == "Sheila"

    @pytest.mark.parametrize("spelling", [None, "unknown", "UU", "PENDING"])
    def test_unattributed_words_are_nobodys(self, spelling: str | None) -> None:
        """The spellings of nobody are Voxtral's adapter's to name (RFC §12.2.3)."""
        turns = SpeakerTurns(pause_s=1.5)
        turns.feed(_segment(spelling, " Hmm."))
        assert turns.close()[0].segments == [SpeakerSegment(None, "Hmm.")]

    def test_done_gives_the_turn_under_way(self) -> None:
        turns = SpeakerTurns(pause_s=1.5)
        turns.feed(_segment("speaker_2", " Friday."))
        assert _said(turns.feed(_done("Friday."))) == [("2", "Friday.")]

    def test_each_word_pushes_the_pause_back(self) -> None:
        turns = SpeakerTurns(pause_s=1.5)
        assert turns.deadline(10.0, None) is None
        turns.feed(_segment("speaker_1", " Okay,"))
        assert turns.deadline(10.0, None) == 11.5
        turns.feed(_segment("speaker_1", " so"))
        assert turns.deadline(10.4, 11.5) == pytest.approx(11.9)

    def test_empty_segments_are_ignored(self) -> None:
        assert SpeakerTurns(pause_s=1.5).feed(_segment("speaker_1", " ")) == []

    def test_an_event_without_words_keeps_the_pause(self) -> None:
        turns = SpeakerTurns(pause_s=1.5)
        turns.feed(_segment("speaker_1", "Hi,"))
        assert turns.deadline(10.0, None) == 11.5
        turns.feed(_segment("speaker_1", " "))
        assert turns.deadline(10.5, 11.5) == 11.5


class TestFile:
    def test_segments_become_one_per_run_of_a_speaker(self) -> None:
        body = {
            "text": "Hi, Julie. Did you read it? Yes.",
            "segments": [
                {"text": "Hi, Julie.", "start": 0.0, "end": 0.4, "speaker_id": "speaker_1"},
                {"text": " Did you read it?", "start": 0.8, "end": 2.3, "speaker_id": "speaker_1"},
                {"text": " Yes.", "start": 3.1, "end": 3.4, "speaker_id": "speaker_2"},
            ],
        }
        result = rest_result(body, diarized=True)
        assert result.segments == [
            SpeakerSegment("1", "Hi, Julie. Did you read it?", 0, 2300),
            SpeakerSegment("2", "Yes.", 3100, 3400),
        ]

    def test_a_run_joins_its_segments_with_a_space_and_keeps_its_last_end(self) -> None:
        body = {
            "text": "Hi. Yes.",
            "segments": [
                {"text": "Hi.", "start": 0.0, "end": 0.4, "speaker_id": "speaker_1"},
                {"text": "Yes.", "start": 0.6, "end": None, "speaker_id": "speaker_1"},
            ],
        }
        assert rest_result(body, diarized=True).segments == [
            SpeakerSegment("1", "Hi. Yes.", 0, 400)
        ]

    def test_text_without_segments_is_nobodys(self) -> None:
        result = rest_result({"text": "Hello.", "segments": []}, diarized=True)
        assert result.segments == [SpeakerSegment(None, "Hello.")]

    def test_without_diarize_no_segment_is_given(self) -> None:
        body = {"text": "Hi.", "segments": [{"text": "Hi.", "speaker_id": "speaker_1"}]}
        assert rest_result(body).segments == []

    async def test_the_request_asks_for_labelled_segments(self) -> None:
        answer = {"text": "Hi.", "segments": [{"text": "Hi.", "speaker_id": "speaker_1"}]}
        with _rest(lambda _: httpx.Response(200, json=answer)) as seen:
            provider = _diarizing(speaker_references={"Julie": b"RIFF"})
            result = await provider.transcribe(AudioChunk(data=_PCM_16K, sample_rate=16000))
            await provider.close()

        fields = {(p[0], p[3]) for p in _multipart(seen[0]) if not p[1]}
        assert ("diarize", b"true") in fields
        assert ("timestamp_granularities", b"segment") in fields
        # The file endpoint kept its own labels with references (measured): live only.
        assert not any(name == "speaker_references" for name, _ in fields)
        assert result.segments == [SpeakerSegment("1", "Hi.")]


class TestStreaming:
    async def test_the_session_asks_for_labels_and_names_the_references(self) -> None:
        record = _Record()
        async with _server(_scripted(record, on_end=[_done("")])) as url:
            await _collect(
                _diarizing(url, speaker_references={"Julie": b"RIFF-julie"}),
                AudioChunk(data=_PCM_16K, sample_rate=16000),
            )

        session = record.update["session"]
        assert session["diarize"] is True
        assert "turn_detection" not in session
        assert session["speaker_references"] == {"Julie": base64.b64encode(b"RIFF-julie").decode()}

    @pytest.mark.wallclock  # a real pause past diarized_pause_ms ends a turn
    async def test_each_speakers_turn_is_a_final_naming_them(self) -> None:
        # The live dialogue: a few words per segment, no turn events.
        record = _Record()
        script: Script = [
            _segment("speaker_1", "Hi,"),
            _segment("speaker_1", " Julie."),
            0.05,
            _segment("speaker_2", " Yes, I read it"),
            _segment("speaker_2", " this morning."),
            0.4,  # a pause longer than diarized_pause_ms ends speaker 2's turn
            _segment("speaker_2", " Friday works."),
        ]
        async with _server(_scripted(record, on_audio=script, on_end=[_done("")])) as url:
            results = await _collect(_diarizing(url), AudioChunk(data=_PCM_16K, sample_rate=16000))

        assert _said(results) == [
            ("1", "Hi, Julie."),
            ("2", "Yes, I read it this morning."),
            ("2", "Friday works."),
        ]
        assert all(len(r.segments) == 1 for r in results if r.is_final)


class TestVoiceChannel:
    async def test_two_voices_reach_the_room_named_on_one_session(self) -> None:
        record = _Record()
        script: Script = [
            _segment("speaker_1", "Did you read the report?"),
            _segment("speaker_2", " Yes, this morning."),
        ]
        handler = _scripted(record, on_audio=script, on_end=[_done("")])

        async with _server(handler) as url:
            stt = _diarizing(url)
            backend = MockVoiceBackend()
            channel = VoiceChannel(
                "voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig()
            )
            kit = RoomKit(stt=stt, voice=backend)
            kit.register_channel(channel)
            transcripts: list[str] = []

            @kit.hook(HookTrigger.ON_TRANSCRIPTION)
            async def on_transcription(event: Any, ctx: Any) -> HookResult:
                transcripts.append(event.text)
                return HookResult.allow()

            room = await kit.create_room()
            await kit.attach_channel(room.id, "voice-1")
            session = await kit.join(room.id, "voice-1", participant_id="owner")
            await backend.simulate_audio_received(session, AudioFrame(data=_PCM_16K * 2))
            await _eventually(lambda: len(transcripts) == 2)
            events = await kit.store.list_events(room.id, offset=0, limit=20)
            await channel.close()

        said = [
            (e.metadata.get("sender_name"), e.content.body)
            for e in events
            if e.type == EventType.MESSAGE and e.source.channel_id == "voice-1"
        ]
        assert said == [
            ("Speaker 1", "Did you read the report?"),
            ("Speaker 2", "Yes, this morning."),
        ]
        assert record.connections == 1

    def test_behind_a_vad_the_channel_refuses_it(self) -> None:
        with pytest.raises(ValueError, match="labels speakers"):
            VoiceChannel(
                "voice-1",
                stt=_diarizing(),
                backend=MockVoiceBackend(),
                pipeline=AudioPipelineConfig(vad=MockVADProvider()),
            )
