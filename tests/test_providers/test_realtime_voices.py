"""Tests for the per-provider realtime voice catalog (available_voices / list_voices).

Curated catalogs are classmethods — no SDK or API key needed, so they are
exercised directly on the class. The ElevenLabs live ``list_voices`` is tested by
building the provider via ``__new__`` and patching the ``elevenlabs`` SDK module,
keeping the test offline while covering the response→VoiceInfo mapping and merge.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

CURATED = [
    OpenAIRealtimeProvider,
    GeminiLiveProvider,
    XAIRealtimeProvider,
    PersonaPlexRealtimeProvider,
    ElevenLabsRealtimeProvider,
]


# --- VoiceInfo + base ABC ------------------------------------------------------


def test_voiceinfo_defaults() -> None:
    v = VoiceInfo(id="x")
    assert v.id == "x"
    assert v.name is None
    assert v.language is None
    assert v.gender is None
    assert v.description is None
    assert v.deprecated is False


def test_base_available_voices_is_empty() -> None:
    assert RealtimeVoiceProvider.available_voices() == []


def test_merge_curated_backfills_and_passes_through() -> None:
    class _Cat(RealtimeVoiceProvider):
        @classmethod
        def available_voices(cls) -> list[VoiceInfo]:
            return [VoiceInfo(id="a", name="A", gender="female", description="warm")]

    merged = {v.id: v for v in _Cat._merge_curated([VoiceInfo(id="a"), VoiceInfo(id="b")])}
    # Known id: metadata filled from the catalog.
    assert merged["a"].name == "A"
    assert merged["a"].gender == "female"
    assert merged["a"].description == "warm"
    # Unknown id: passes through untouched.
    assert merged["b"].name is None


def test_merge_curated_prefers_live_values() -> None:
    class _Cat(RealtimeVoiceProvider):
        @classmethod
        def available_voices(cls) -> list[VoiceInfo]:
            return [VoiceInfo(id="a", name="Curated", gender="male")]

    merged = _Cat._merge_curated([VoiceInfo(id="a", name="Live", gender="female")])[0]
    assert merged.name == "Live"
    assert merged.gender == "female"


# --- Curated catalogs (offline, no SDK, no key) --------------------------------


@pytest.mark.parametrize("provider_cls", CURATED)
def test_curated_catalog_is_nonempty_and_unique(provider_cls: type[RealtimeVoiceProvider]) -> None:
    voices = provider_cls.available_voices()
    assert voices, f"{provider_cls.__name__} has an empty curated voice catalog"
    assert all(isinstance(v, VoiceInfo) for v in voices)
    ids = [v.id for v in voices]
    assert len(ids) == len(set(ids)), f"{provider_cls.__name__} has duplicate voice ids"


def test_openai_known_voice_present() -> None:
    ids = {v.id for v in OpenAIRealtimeProvider.available_voices()}
    assert {"alloy", "verse"} <= ids


def test_personaplex_voice_ids_are_pt_prompts() -> None:
    # The provider passes the voice as the .pt prompt filename.
    assert all(v.id.endswith(".pt") for v in PersonaPlexRealtimeProvider.available_voices())


# --- ElevenLabs live list_voices (fake SDK injected, offline) ------------------


async def test_elevenlabs_list_voices_maps_and_merges() -> None:
    provider = ElevenLabsRealtimeProvider.__new__(ElevenLabsRealtimeProvider)
    provider._config = SimpleNamespace(api_key="test-key")  # type: ignore[attr-defined]

    # A known default (Adam) to exercise the curated merge, plus a custom voice.
    voices = [
        SimpleNamespace(
            voice_id="pNInz6obpgDQGcFmaJgB",
            name="Adam - Deep",
            labels={"gender": "male"},
            description=None,
        ),
        SimpleNamespace(
            voice_id="custom123",
            name="My Clone",
            labels={},
            description="cloned",
        ),
    ]
    fake_client = MagicMock()
    fake_client.voices.get_all.return_value = SimpleNamespace(voices=voices)
    fake_elevenlabs = SimpleNamespace(ElevenLabs=MagicMock(return_value=fake_client))

    with patch.dict("sys.modules", {"elevenlabs": fake_elevenlabs}):
        result = {v.id: v for v in await provider.list_voices()}

    # Live name strips the " - ..." suffix; gender from labels.
    assert result["pNInz6obpgDQGcFmaJgB"].name == "Adam"
    assert result["pNInz6obpgDQGcFmaJgB"].gender == "male"
    # description backfilled from the curated catalog (live had None).
    assert result["pNInz6obpgDQGcFmaJgB"].description is not None
    # Custom voice passes through with its own data.
    assert result["custom123"].name == "My Clone"
    assert result["custom123"].description == "cloned"
