"""Gemini's speaker turns in the shared contract (RFC §12.2.3).

``speaker_segments=True`` puts the turns on ``transcribe()``'s result; it is off
by default because ``diarize`` has always been on and a diarizing STT is
refused by a VoiceChannel behind a VAD.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from roomkit import VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, SpeakerSegment
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.stt.gemini import (
    GeminiSTTConfig,
    GeminiSTTProvider,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

_PROMPTED = Transcript(
    language="fr",
    segments=[
        TranscriptSegment("Speaker 1", "00:00", "00:04", "Bonjour Julie."),
        TranscriptSegment("Speaker 2", "00:04", "00:09", "Oui, je l'ai lu."),
        TranscriptSegment("Speaker 1", "", "", "Parfait."),
    ],
)
_RECOGNISED = Transcript(
    language="",
    segments=[],
    words=[
        TranscriptWord("Bonjour", 0.3, 0.7, "Speaker 1"),
        TranscriptWord("Julie.", 0.7, 1.2, "Speaker 1"),
        TranscriptWord("Oui.", 1.6, 1.9, "Speaker 2"),
    ],
)


class TestTranscriptSpeakerSegments:
    def test_prompted_turns_keep_the_models_seconds(self) -> None:
        assert _PROMPTED.speaker_segments() == [
            SpeakerSegment("1", "Bonjour Julie.", 0, 4000),
            SpeakerSegment("2", "Oui, je l'ai lu.", 4000, 9000),
            SpeakerSegment("1", "Parfait.", None, None),
        ]

    def test_recognised_words_give_turns_timed_to_the_word(self) -> None:
        assert _RECOGNISED.speaker_segments() == [
            SpeakerSegment("1", "Bonjour Julie.", 300, 1200),
            SpeakerSegment("2", "Oui.", 1600, 1900),
        ]

    @pytest.mark.parametrize("spelling", ["UU", "PENDING", "unknown", ""])
    def test_a_turn_attributed_to_nobody_has_no_speaker(self, spelling: str) -> None:
        """The spellings are Gemini's adapter's to name (RFC §12.2.3)."""
        transcript = Transcript(
            language="fr", segments=[TranscriptSegment(spelling, "", "", "Suite.")]
        )
        assert transcript.speaker_segments() == [SpeakerSegment(None, "Suite.", None, None)]

    def test_an_hour_long_offset_and_an_unreadable_one(self) -> None:
        transcript = Transcript(
            language="fr",
            segments=[TranscriptSegment("Speaker 1", "1:02:05", "about then", "Suite.")],
        )
        assert transcript.speaker_segments() == [SpeakerSegment("1", "Suite.", 3_725_000, None)]


def _provider(**config: object) -> GeminiSTTProvider:
    provider = GeminiSTTProvider(GeminiSTTConfig(api_key="k", **config))  # type: ignore[arg-type]
    provider.transcribe_recording = AsyncMock(return_value=_PROMPTED)  # type: ignore[method-assign]
    return provider


class TestProvider:
    def test_segments_need_diarize(self) -> None:
        with pytest.raises(ValueError, match="needs diarize"):
            GeminiSTTConfig(api_key="k", diarize=False, speaker_segments=True)

    @pytest.mark.parametrize(
        ("config", "claimed"),
        [({}, False), ({"diarize": True}, False), ({"speaker_segments": True}, True)],
        ids=["default", "diarize-only", "speaker-segments"],
    )
    def test_only_speaker_segments_claims_diarization(
        self, config: dict[str, object], claimed: bool
    ) -> None:
        assert _provider(**config).supports_diarization is claimed

    async def test_transcribe_carries_the_turns_when_asked(self) -> None:
        result = await _provider(speaker_segments=True).transcribe(
            AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)
        )
        assert [s.speaker for s in result.segments] == ["1", "2", "1"]
        assert result.text == "Bonjour Julie. Oui, je l'ai lu. Parfait."

    async def test_transcribe_is_unchanged_by_default(self) -> None:
        result = await _provider().transcribe(
            AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)
        )
        assert result.segments == []


class TestVoiceChannel:
    def test_the_default_config_still_works_behind_a_vad(self) -> None:
        # Backward compatibility: diarize=True alone never claimed diarization.
        VoiceChannel(
            "voice-1",
            stt=_provider(),
            backend=MockVoiceBackend(),
            pipeline=AudioPipelineConfig(vad=MockVADProvider(events=[])),
        )

    def test_speaker_segments_are_refused_behind_a_vad(self) -> None:
        with pytest.raises(ValueError, match="continuous mode"):
            VoiceChannel(
                "voice-1",
                stt=_provider(speaker_segments=True),
                backend=MockVoiceBackend(),
                pipeline=AudioPipelineConfig(vad=MockVADProvider(events=[])),
            )
