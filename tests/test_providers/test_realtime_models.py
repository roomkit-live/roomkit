"""Tests for the per-provider realtime model catalog (``available_models``).

The speech-to-speech counterpart of ``test_realtime_voices.py``: curated
catalogs are classmethods — no SDK or API key needed, so they are exercised
directly on the class. The one invariant that matters most is that each
provider's *constructor default* appears in its own catalog: a default the
catalog does not know is either a stale catalog or a stale default, and
nothing else would notice which.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.realtime.mock import MockRealtimeAudioVideoProvider, MockRealtimeProvider
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

CATALOGED = [OpenAIRealtimeProvider, GeminiLiveProvider, XAIRealtimeProvider]


# --- base ABC ------------------------------------------------------------------


def test_base_available_models_is_empty() -> None:
    class _Bare(RealtimeVoiceProvider):
        pass

    assert _Bare.available_models() == []


@pytest.mark.parametrize("provider_cls", [DeepgramAgentProvider, ElevenLabsRealtimeProvider])
def test_composed_and_agent_bound_providers_keep_the_empty_default(provider_cls: type) -> None:
    """Deepgram has no end-to-end model id; ElevenLabs binds a dashboard agent."""
    assert provider_cls.available_models() == []


# --- curated catalogs ----------------------------------------------------------


@pytest.mark.parametrize("provider_cls", CATALOGED)
def test_curated_catalog_is_nonempty_and_unique(provider_cls: type) -> None:
    models = provider_cls.available_models()
    assert models
    assert all(isinstance(m, ModelInfo) for m in models)
    ids = [m.id for m in models]
    assert len(ids) == len(set(ids))


def test_openai_default_model_is_in_its_catalog() -> None:
    default = inspect.signature(OpenAIRealtimeProvider.__init__).parameters["model"].default
    assert default in {m.id for m in OpenAIRealtimeProvider.available_models()}


def test_gemini_default_model_is_in_its_catalog() -> None:
    default = inspect.signature(GeminiLiveProvider.__init__).parameters["model"].default
    assert default in {m.id for m in GeminiLiveProvider.available_models()}


def test_xai_default_model_is_in_its_catalog() -> None:
    default = XAIRealtimeConfig.model_fields["model"].default
    assert default in {m.id for m in XAIRealtimeProvider.available_models()}


def test_vision_follows_the_documented_cut() -> None:
    """gpt-realtime-2.1+ reads images; xAI deliberately does not (0.43.0)."""
    by_id = {m.id: m for m in OpenAIRealtimeProvider.available_models()}
    assert by_id["gpt-realtime-2.1"].supports_vision is True
    assert by_id["gpt-realtime-1.5"].supports_vision is False

    assert all(m.supports_vision for m in GeminiLiveProvider.available_models())
    assert all(not m.supports_vision for m in XAIRealtimeProvider.available_models())


def test_retired_preview_lineup_is_flagged_deprecated() -> None:
    deprecated = {m.id for m in OpenAIRealtimeProvider.available_models() if m.deprecated}
    assert deprecated == {"gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"}


# --- release-gate coverage -----------------------------------------------------


def test_the_check_script_names_every_realtime_catalog() -> None:
    """Being unmirrored is fine; being unmentioned is how a catalog goes stale."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_models.py"
    spec = importlib.util.spec_from_file_location("check_models_rt", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec — @dataclass resolves the string annotations
    # `from __future__ import annotations` produces via sys.modules.
    sys.modules["check_models_rt"] = module
    spec.loader.exec_module(module)

    assert {"openai-realtime", "gemini-realtime", "xai-realtime"} <= set(
        module.UNMIRRORED_CATALOGS
    )


# --- model_name ----------------------------------------------------------------


def test_a_self_hosted_single_model_provider_keeps_the_default() -> None:
    """PersonaPlex serves one model and names none: the default answers."""
    provider = PersonaPlexRealtimeProvider()
    assert provider.model_name == provider.name == "PersonaPlexRealtimeProvider"


def test_openai_reports_the_model_it_connects_to() -> None:
    provider = OpenAIRealtimeProvider(api_key="sk-test", model="gpt-realtime-2.1")
    assert provider.model_name == "gpt-realtime-2.1"


def test_xai_reports_the_model_it_connects_to() -> None:
    provider = XAIRealtimeProvider(XAIRealtimeConfig(api_key="xai-test", model="grok-2-audio"))
    assert provider.model_name == "grok-2-audio"


def test_deepgram_reports_its_think_model() -> None:
    """A composed stack names the stage that decides what the agent says."""
    provider = DeepgramAgentProvider(api_key="dg-test")
    default_think = DeepgramAgentConfig.model_fields["think_model"].default
    assert provider.model_name == default_think
    assert provider.model_name != provider.name


def test_an_agent_bound_provider_keeps_the_default() -> None:
    """ElevenLabs binds a dashboard-configured agent: no model id to give."""
    provider = ElevenLabsRealtimeProvider(
        ElevenLabsRealtimeConfig(api_key="el-test", agent_id="agent-1")
    )
    assert provider.model_name == provider.name


def test_anam_names_its_llm_only_for_an_inline_persona() -> None:
    inline = AnamRealtimeProvider(AnamConfig(api_key="ak-test", llm_id="llm-1"))
    assert inline.model_name == "llm-1"

    lab_persona = AnamRealtimeProvider(AnamConfig(api_key="ak-test", persona_id="persona-1"))
    assert lab_persona.model_name == lab_persona.name


def test_the_mock_shows_both_shapes() -> None:
    """A mock that always named a model would lie about half the fleet."""
    assert MockRealtimeProvider().model_name == "MockRealtimeProvider"
    assert MockRealtimeProvider(model="gpt-realtime-2.1").model_name == "gpt-realtime-2.1"


def test_the_av_mock_can_name_a_model_too() -> None:
    """Anam is the avatar shape, and its model id is the conditional one."""
    assert MockRealtimeAudioVideoProvider().model_name == "MockRealtimeAudioVideoProvider"
    assert MockRealtimeAudioVideoProvider(model="llm-1").model_name == "llm-1"
