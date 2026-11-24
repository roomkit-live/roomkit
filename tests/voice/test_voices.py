"""VoiceInfo, the catalog filters, the dialogue checks and the voice library defaults
shared by every provider (RFC §12.2, §12.2.4)."""

from __future__ import annotations

import pytest

from roomkit.voice import VoiceLibrary
from roomkit.voice.realtime.provider import VoiceInfo as RealtimeVoiceInfo
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.voices import (
    DialogueTurn,
    VoiceInfo,
    check_dialogue,
    dialogue_transcript,
    filter_voices,
)

VOICES = [
    VoiceInfo(id="qc", name="Montreal advisor", language="fr-CA", gender="female"),
    VoiceInfo(id="fr", name="Paris advisor", language="fr-FR", gender="male"),
    VoiceInfo(id="fry", name="Frisian", language="fry", gender="male"),
    VoiceInfo(id="en", name="Kore", language="en-US", gender="female", description="Firm"),
    VoiceInfo(id="any", name="Unknown"),
]


def _ids(voices: list[VoiceInfo]) -> list[str]:
    return [v.id for v in voices]


class TestFilters:
    def test_a_language_prefix_takes_its_regions(self) -> None:
        assert _ids(filter_voices(VOICES, language="fr")) == ["qc", "fr"]

    def test_a_region_does_not_take_its_siblings(self) -> None:
        assert _ids(filter_voices(VOICES, language="fr-CA")) == ["qc"]

    def test_the_prefix_stops_at_the_hyphen(self) -> None:
        """``fr`` is not the start of ``fry``."""
        assert "fry" not in _ids(filter_voices(VOICES, language="fr"))

    def test_gender_is_exact_and_case_blind(self) -> None:
        assert _ids(filter_voices(VOICES, gender="FEMALE")) == ["qc", "en"]

    def test_query_looks_in_the_name_and_the_description(self) -> None:
        assert _ids(filter_voices(VOICES, query="advisor")) == ["qc", "fr"]
        assert _ids(filter_voices(VOICES, query="FIRM")) == ["en"]

    def test_an_unknown_field_matches_no_filter_on_it(self) -> None:
        assert "any" not in _ids(filter_voices(VOICES, language="en"))
        assert "any" not in _ids(filter_voices(VOICES, gender="male"))

    def test_filters_combine(self) -> None:
        assert _ids(filter_voices(VOICES, language="fr", gender="male")) == ["fr"]

    def test_no_filter_keeps_everything(self) -> None:
        assert _ids(filter_voices(VOICES)) == _ids(VOICES)


class TestVoiceInfo:
    def test_the_realtime_import_path_is_the_same_class(self) -> None:
        assert RealtimeVoiceInfo is VoiceInfo


class TestDialogueChecks:
    TURNS = [
        DialogueTurn(speaker="Joe", text="Salut."),
        DialogueTurn(speaker="Jane", text="Bonjour."),
        DialogueTurn(speaker="Joe", text="Ça va ?"),
    ]
    VOICES = {"Joe": "Puck", "Jane": "Kore"}

    def test_speakers_come_back_in_order_of_first_appearance(self) -> None:
        assert check_dialogue(self.TURNS, self.VOICES, max_speakers=2, provider="p") == [
            "Joe",
            "Jane",
        ]

    def test_no_dialogue_capability_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not synthesize dialogue"):
            check_dialogue(self.TURNS, self.VOICES, max_speakers=0, provider="p")

    def test_too_many_speakers_is_refused_before_any_call(self) -> None:
        with pytest.raises(ValueError, match="at most 1 speakers"):
            check_dialogue(self.TURNS, self.VOICES, max_speakers=1, provider="p")

    def test_an_unmapped_speaker_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'Jane' has no voice"):
            check_dialogue(self.TURNS, {"Joe": "Puck"}, max_speakers=2, provider="p")

    def test_an_empty_dialogue_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one turn"):
            check_dialogue([], self.VOICES, max_speakers=2, provider="p")

    def test_the_transcript_is_one_line_per_turn(self) -> None:
        assert dialogue_transcript(self.TURNS) == "Joe: Salut.\nJane: Bonjour.\nJoe: Ça va ?"


class _PlainTTS(TTSProvider):
    async def synthesize(self, text, *, voice=None):  # type: ignore[override]
        raise AssertionError("not called")


class TestTTSProviderDefaults:
    async def test_a_provider_without_a_catalog_lists_nothing(self) -> None:
        assert _PlainTTS.available_voices() == []
        assert await _PlainTTS().list_voices(language="fr") == []

    async def test_the_default_list_filters_the_curated_catalog(self) -> None:
        class Curated(_PlainTTS):
            @classmethod
            def available_voices(cls) -> list[VoiceInfo]:
                return list(VOICES)

        assert _ids(await Curated().list_voices(language="fr-FR")) == ["fr"]

    async def test_a_provider_without_dialogue_refuses_it(self) -> None:
        assert _PlainTTS().max_dialogue_speakers == 0
        with pytest.raises(NotImplementedError):
            await _PlainTTS().synthesize_dialogue(
                [DialogueTurn(speaker="Joe", text="Salut.")], {"Joe": "Puck"}
            )


class TestVoiceLibraryDefaults:
    async def test_the_base_refuses_what_it_does_not_announce(self) -> None:
        class Nothing(VoiceLibrary):
            async def get_voice(self, voice_id: str) -> None:
                return None

            async def delete_voice(self, voice_id: str) -> None:
                return None

        with pytest.raises(NotImplementedError, match="does not design"):
            await Nothing().design_voice("a voice")
        with pytest.raises(NotImplementedError, match="does not replicate"):
            await Nothing().replicate_voice(b"sample", b"consent")
