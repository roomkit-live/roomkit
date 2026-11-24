"""GeminiVoiceLibrary: custom voices designed or replicated (RFC §12.2.4).

The fake mirrors the live API (verified 2026-09-27): a designed voice must be
stored, ``get`` and ``delete`` answer 404 for an id Google does not hold, and a
refused consent arrives as a 500 wrapping Google's ``Consent flow failed``.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roomkit.models.event import AudioContent
from roomkit.voice.tts.audio_utils import wrap_wav
from roomkit.voice.tts.gemini_library import (
    CONSENT_STATEMENTS,
    GeminiVoiceLibrary,
    GeminiVoiceLibraryConfig,
)
from roomkit.voice.tts.library import VoiceConsentError

WAV_24K = wrap_wav(b"\x01\x00" * 24000, 24000, 1)
EXPIRES = datetime(2027, 9, 28, tzinfo=UTC)
CONSENT_DUMP = (
    "Error code: 500 - {'error': {'code': 500, 'message': 'Error translating server response "
    "to JSON', 'details': [{'detail': 'Original error: INVALID_ARGUMENT: Consent flow failed. "
    "Please follow instructions at https://ai.google.dev for troubleshooting.\\nThe recorded "
    "phrase didn\\'t match the text on screen. Please read the prompt exactly as written. "
    "[type.googleapis.com/util.MessageSetPayload='"
)


class _ApiError(Exception):
    def __init__(self, code: int, text: str = "") -> None:
        super().__init__(text or f"{code} error")
        self.code = code


def _created(**fields: Any) -> SimpleNamespace:
    base = {
        "id": "voice_abc",
        "key": None,
        "display_name": "Narratrice",
        "language_code": None,
        "gender": None,
        "accent": None,
        "description": None,
        "persona": None,
        "context": None,
        "region_code": None,
        "model": "models/gemini-3.8-flash-tts",
        "type": "prompted",
        "pitch": None,
        "expire_time": EXPIRES,
        "sample_audio": SimpleNamespace(
            data=base64.b64encode(WAV_24K).decode(), mime_type="audio/wav"
        ),
    }
    base.update(fields)
    return SimpleNamespace(**base)


class _FakeVoices:
    def __init__(self, *, create: Any = None, get: Any = None, delete: Any = None) -> None:
        self._create, self._get, self._delete = create, get, delete
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    async def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        if isinstance(self._create, Exception):
            raise self._create
        return self._create or _created()

    async def get(self, voice_id: str) -> Any:
        if isinstance(self._get, Exception):
            raise self._get
        return self._get or _created(id=voice_id)

    async def delete(self, voice_id: str) -> None:
        self.deleted.append(voice_id)
        if isinstance(self._delete, Exception):
            raise self._delete


def _library(**voices: Any) -> tuple[GeminiVoiceLibrary, _FakeVoices]:
    library = GeminiVoiceLibrary(GeminiVoiceLibraryConfig(api_key="test-key"))
    fake = _FakeVoices(**voices)
    library._client = SimpleNamespace(aio=SimpleNamespace(voices=fake))
    return library, fake


def _audio(wav: bytes = WAV_24K) -> AudioContent:
    return AudioContent(url=f"data:audio/wav;base64,{base64.b64encode(wav).decode()}")


class TestCapabilities:
    def test_it_designs_and_replicates(self) -> None:
        library, _ = _library()
        assert (library.supports_design, library.supports_replication) == (True, True)


class TestDesign:
    async def test_a_description_becomes_a_stored_voice(self) -> None:
        library, fake = _library()

        custom = await library.design_voice("Une narratrice chaleureuse", name="Narratrice")

        assert fake.created == [
            {
                "voice": {
                    "type": "prompted",
                    "prompted": {"input": "Une narratrice chaleureuse"},
                    "display_name": "Narratrice",
                },
                "store": True,
            }
        ]
        assert custom.voice.id == "voice_abc"
        assert custom.voice.attributes["type"] == "prompted"
        assert custom.stored is True
        assert custom.expires_at == EXPIRES
        assert custom.sample is not None
        assert base64.b64decode(custom.sample.url.split(",", 1)[1]) == WAV_24K

    async def test_an_unstored_design_is_refused_before_any_call(self) -> None:
        """Google answers ``Prompted voice creation requires store=true``."""
        library, fake = _library()

        with pytest.raises(ValueError, match="needs store=True"):
            await library.design_voice("Une narratrice", store=False)

        assert fake.created == []

    async def test_a_blank_description_is_refused(self) -> None:
        library, fake = _library()

        with pytest.raises(ValueError, match="description"):
            await library.design_voice("  ")

        assert fake.created == []


class TestReplication:
    async def test_both_recordings_travel_as_wav(self, tmp_path: Path) -> None:
        sample = tmp_path / "sample.wav"
        sample.write_bytes(WAV_24K)
        library, fake = _library(create=_created(type="replicated"))

        custom = await library.replicate_voice(sample, _audio(), name="Sylvain")

        replicated = fake.created[0]["voice"]["replicated"]
        assert base64.b64decode(replicated["source_audio"]["data"]) == WAV_24K
        assert base64.b64decode(replicated["consent_audio"]["data"]) == WAV_24K
        assert replicated["source_audio"]["mime_type"] == "audio/wav"
        assert fake.created[0]["store"] is True
        assert custom.voice.attributes["type"] == "replicated"

    async def test_an_unstored_voice_is_named_by_its_key(self) -> None:
        library, _ = _library(create=_created(id=None, key="voicekey_xyz", type="replicated"))

        custom = await library.replicate_voice(WAV_24K, WAV_24K, store=False)

        assert custom.voice.id == "voicekey_xyz"
        assert custom.stored is False

    async def test_a_refused_consent_is_raised_as_such(self) -> None:
        """Google sends it as a 500 wrapping the real refusal (seen 2026-09-27)."""
        library, _ = _library(create=_ApiError(500, CONSENT_DUMP))

        with pytest.raises(VoiceConsentError) as caught:
            await library.replicate_voice(WAV_24K, WAV_24K)

        assert str(caught.value) == (
            "Google refused the consent: The recorded phrase didn't match the text on "
            "screen. Please read the prompt exactly as written."
        )

    async def test_another_failure_is_raised_as_it_came(self) -> None:
        library, _ = _library(create=_ApiError(429, "quota"))

        with pytest.raises(_ApiError):
            await library.replicate_voice(WAV_24K, WAV_24K)

    @pytest.mark.parametrize(
        ("wav", "message"),
        [
            (wrap_wav(b"\x01\x00" * 16000, 16000, 1), "got 16-bit 16000 Hz"),
            (wrap_wav(b"\x01\x00" * 48000, 24000, 2), "with 2 channel"),
            (b"not a wav", "not a readable WAV"),
        ],
    )
    async def test_a_recording_google_cannot_take_is_refused_before_any_call(
        self, wav: bytes, message: str
    ) -> None:
        library, fake = _library()

        with pytest.raises(ValueError, match=message):
            await library.replicate_voice(wav, WAV_24K)

        assert fake.created == []

    async def test_a_remote_url_is_not_fetched(self) -> None:
        library, fake = _library()

        with pytest.raises(ValueError, match="data: URL"):
            await library.replicate_voice(AudioContent(url="https://example.com/a.wav"), WAV_24K)

        assert fake.created == []

    def test_the_consent_statements_are_googles_words(self) -> None:
        assert CONSENT_STATEMENTS["fr-CA"].startswith("Je suis le propriétaire de cette voix")
        assert CONSENT_STATEMENTS["en-US"].startswith("I am the owner of this voice")


class TestGetAndDelete:
    async def test_a_held_voice_is_read_back(self) -> None:
        library, _ = _library()

        custom = await library.get_voice("voice_abc")

        assert custom is not None
        assert custom.voice.name == "Narratrice"

    async def test_an_unknown_voice_is_none(self) -> None:
        library, _ = _library(get=_ApiError(404))

        assert await library.get_voice("voice_gone") is None

    async def test_another_read_error_is_raised(self) -> None:
        library, _ = _library(get=_ApiError(500))

        with pytest.raises(_ApiError):
            await library.get_voice("voice_abc")

    async def test_deleting_an_unknown_voice_is_not_an_error(self) -> None:
        library, fake = _library(delete=_ApiError(404))

        await library.delete_voice("voice_gone")

        assert fake.deleted == ["voice_gone"]

    async def test_another_delete_error_is_raised(self) -> None:
        library, _ = _library(delete=_ApiError(403))

        with pytest.raises(_ApiError):
            await library.delete_voice("voice_abc")


class TestPublicLoaders:
    def test_get_gemini_voice_library(self) -> None:
        from roomkit.voice import get_gemini_voice_library, get_gemini_voice_library_config

        assert get_gemini_voice_library() is GeminiVoiceLibrary
        assert get_gemini_voice_library_config() is GeminiVoiceLibraryConfig
