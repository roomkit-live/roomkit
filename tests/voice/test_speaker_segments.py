"""Speaker attribution on transcription results (RFC §12.2.3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from roomkit import MockConferenceBackend, VoiceChannel
from roomkit.channels.conference import ConferenceChannel
from roomkit.voice import speaker_label
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import SpeakerSegment, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.stt.mock import MockSTTProvider


class _DiarizingSTT(MockSTTProvider):
    @property
    def supports_diarization(self) -> bool:
        return True


class _LegacySTT:
    """A duck-typed provider written before ``supports_diarization`` existed."""

    name = "LegacySTT"
    supports_streaming = False

    async def transcribe(self, audio: object) -> TranscriptionResult:
        return TranscriptionResult(text="")


class TestSpeakerLabel:
    @pytest.mark.parametrize(
        ("value", "label"),
        [
            ("Speaker 1", "1"),
            ("speaker_0", "0"),
            ("speaker1", "1"),
            ("Speakerphone", "Speakerphone"),
            ("speakers", "speakers"),
            ("A", "A"),
            (0, "0"),
            (" B ", "B"),
            ("S1", "S1"),
            ("", None),
            ("  ", None),
            (None, None),
        ],
    )
    def test_labels_are_short_strings_and_none_is_none(
        self, value: object, label: str | None
    ) -> None:
        assert speaker_label(value) == label

    @pytest.mark.parametrize("spelling", ["UU", "PENDING", "unknown"])
    def test_the_rule_knows_no_vendor_spelling(self, spelling: str) -> None:
        """RFC §12.2.3: a speaker the provider could not attribute is null
        whatever its vendor spells it, and the spelling is the provider's to
        name: RoomKit's rule reads none of them."""
        assert speaker_label(spelling) == spelling

    @pytest.mark.parametrize("value", ["UU", "uu", " Pending "])
    def test_a_provider_names_its_own_unattributed_spellings(self, value: str) -> None:
        assert speaker_label(value, unattributed=("UU", "PENDING")) is None
        assert speaker_label("A", unattributed=("UU", "PENDING")) == "A"


class TestTranscriptionResultSpeaker:
    def test_no_segments_means_no_speaker(self) -> None:
        result = TranscriptionResult(text="Bonjour")
        assert result.segments == []
        assert result.speaker is None

    def test_one_speaker_across_segments(self) -> None:
        result = TranscriptionResult(
            text="Bonjour. Ça va?",
            segments=[SpeakerSegment("A", "Bonjour."), SpeakerSegment("A", "Ça va?")],
        )
        assert result.speaker == "A"

    def test_mixed_speakers_have_no_single_speaker(self) -> None:
        result = TranscriptionResult(
            text="Bonjour. Oui.",
            segments=[SpeakerSegment("A", "Bonjour."), SpeakerSegment("B", "Oui.")],
        )
        assert result.speaker is None

    def test_an_unattributed_segment_is_not_a_speaker(self) -> None:
        result = TranscriptionResult(text="Oui.", segments=[SpeakerSegment(None, "Oui.")])
        assert result.speaker is None


class TestProviderDefault:
    def test_providers_do_not_diarize_unless_they_say_so(self) -> None:
        assert MockSTTProvider().supports_diarization is False


class TestVoiceChannelAcceptsOnlyAnExplicitClaim:
    # Where a diarizing STT is refused or carried: test_speaker_attribution_channel.

    @pytest.mark.parametrize("stt", [MagicMock(), _LegacySTT()], ids=["mock", "duck-typed"])
    def test_only_an_explicit_claim_counts(self, stt: object) -> None:
        # A mock's auto-attribute and a provider predating the property are
        # not claims to label speakers, even behind a VAD.
        VoiceChannel(
            "voice-1",
            stt=stt,  # type: ignore[arg-type]
            backend=MockVoiceBackend(),
            pipeline=AudioPipelineConfig(vad=MockVADProvider(events=[])),
        )


class TestConferenceRefusesDiarizingSTT:
    # A conference attributes speech by participant track and transcribes each
    # utterance on its own: labels would compare with nothing (RFC §12.2.3).
    def test_at_construction(self) -> None:
        with pytest.raises(ValueError, match="participant track"):
            ConferenceChannel("conf", backend=MockConferenceBackend(), stt=_DiarizingSTT())

    async def test_when_plugged(self) -> None:
        channel = ConferenceChannel("conf", backend=MockConferenceBackend())
        with pytest.raises(ValueError, match="participant track"):
            await channel.plug_stt(_DiarizingSTT())
        assert channel.info["stt_configured"] is False
